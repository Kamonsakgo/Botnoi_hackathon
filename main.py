from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Header, Query, Form,Request
import pandas as pd
import uvicorn
from pydantic import BaseModel
from passlib.context import CryptContext
from typing import Dict, Optional
from pymongo import MongoClient
from dotenv import load_dotenv
import jwt
from bson import ObjectId
import gridfs
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import uuid
from datetime import datetime, timedelta
import os  # อันนี้จะรวมเข้ามาแทนที่ที่นำเข้า os หลายครั้ง
from fastapi.responses import JSONResponse
import qrcode
import stripe
import google.generativeai as genai
import warnings
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Load environment variables
load_dotenv()
MONGO_URL = os.getenv("MONGO_URL")
SECRET_KEY = os.getenv("SECRET_KEY")
WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 12
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
genai.configure(api_key= os.getenv("API_KEY"))
# Connect to MongoDB
client = MongoClient(MONGO_URL)
db = client.get_database("Hackathon")
users_collection = db.get_collection("users")
fs = gridfs.GridFS(db)
app = FastAPI(
    title="Hackathon API",
    description="API สำหรับจัดการผู้ใช้และวิเคราะห์ข้อมูลไฟล์ CSV",
    version="1.0"
)

# ✅ เปิดให้ Angular หรือ Frontend อื่นๆ เข้าถึง API ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # เพิ่ม URL ของ Angular app และ ngrok URL
    allow_credentials=True,
    allow_methods=["*"],  # อนุญาตทุก HTTP method (GET, POST, PUT, DELETE ฯลฯ)
    allow_headers=["*"],  # อนุญาตทุก HTTP headers
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(BaseModel):
    username: str
    password: str
    points: int

class UserLogin(BaseModel):
    username: str
    password: str
    points: int

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    user_record = users_collection.find_one({"username": username})
    if not user_record or not verify_password(password, user_record["password"]):
        return None
    return user_record

def create_access_token(username: str):
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    to_encode = {"sub": username, "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(authorization: Optional[str] = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    token = authorization.split("Bearer ")[1]
    return verify_token(token)
    
conversation_context = ""  # ตัวแปร global ที่เก็บบริบท

conversation_context = ""  # ตัวแปร global ที่เก็บบริบท

def ask_gemini(prompt, context):
    global conversation_context  # ใช้ตัวแปร global เพื่อให้สามารถอัปเดตค่า conversation_context ได้
    
    # หากมีบริบทที่ถูกส่งมา เราจะรวมคำถามใหม่กับบริบทเก่า
    if context != "":
        full_prompt = prompt + "\n" + context  # รวมคำถามใหม่กับบริบทที่มีอยู่
    else:
        # ถ้าไม่มีบริบทจากคำถามก่อนหน้า เราจะใช้แค่คำถามใหม่
        full_prompt = prompt
        
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(full_prompt)
    
    # อัปเดต conversation_context
    conversation_context = full_prompt + "\n" + response.text  # เก็บทั้ง prompt และ response
    return response.text, conversation_context  # ส่งกลับคำตอบและบริบทที่อัปเดต


# ✅ Register API
@app.post("/register/", summary="สมัครสมาชิก", description="API สำหรับสมัครสมาชิก โดยต้องส่ง `username` และ `password` เป็น form-data")
async def register(username: str = Form(...), password: str = Form(...)):
    if users_collection.find_one({"username": username}):
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(password)
    users_collection.insert_one({"username": username, "password": hashed_password, "points": 200})
    
    return {"message": "User registered successfully"}

# ✅ Login API
@app.post("/login/", summary="เข้าสู่ระบบ", description="API สำหรับเข้าสู่ระบบ โดยต้องส่ง `username` และ `password` เป็น form-data")
async def login(username: str = Form(...), password: str = Form(...)):
    user_record = authenticate_user(username, password)
    if not user_record:
        raise HTTPException(status_code=400, detail="Invalid username or password")
    
    token = create_access_token(username)
    return {"message": "Login successful", "token": token}

# ✅ Upload File API


# ฟังก์ชันที่ใช้ในการจัดการชื่อไฟล์
def generate_unique_filename(filename: str):
    file_name, file_extension = os.path.splitext(filename)  # แยกชื่อไฟล์ออกจากส่วนขยาย
    unique_suffix = f"_{uuid.uuid4().hex[:8]}"  # ใช้ UUID (8 หลัก) เพื่อให้ชื่อไม่ซ้ำ
    return f"{file_name}{unique_suffix}{file_extension}"

@app.post("/upload/", summary="อัปโหลดไฟล์", description="API สำหรับอัปโหลดไฟล์ CSV โดยต้องใช้ Token Authentication")
async def upload_file(file: UploadFile = File(...), username: str = Depends(get_current_user)):
    try:
        # ตรวจสอบคะแนนของผู้ใช้
        user = users_collection.find_one({"username": username})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if user.get("points", 0) < 100:
            raise HTTPException(status_code=400, detail="Insufficient points to upload file")
        
        # หักคะแนน 100 จากผู้ใช้
        users_collection.update_one(
            {"username": username},
            {"$inc": {"points": -100}}  # หักคะแนน 100
        )

        # สร้างชื่อไฟล์ที่ไม่ซ้ำ โดยเพิ่มเลขหรือตัวอักษรเพื่อให้ไฟล์ไม่ซ้ำ
        unique_filename = generate_unique_filename(file.filename)
        
        # อ่านข้อมูลไฟล์
        file_data = file.file.read()

        # อัปโหลดไฟล์และเก็บข้อมูลใน GridFS
        file_id = fs.put(file_data, filename=unique_filename, username=username, upload_time=datetime.utcnow())

        # ส่งข้อความยืนยัน
        return {"message": "File uploaded successfully", "file_id": str(file_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/dashboard/summary/", summary="สรุปยอดขาย", description="API สำหรับดึงข้อมูลสรุปยอดขายรายวัน รายเดือน รายปี และสินค้าขายดี พร้อมแนวโน้มการเติบโต และพยากรณ์ยอดขายล่วงหน้า")
async def dashboard_summary(
        file_id: str = Query(..., description="File ID in MongoDB"),
        report_type: str = Query("all", description="ประเภทของรายงาน: daily, monthly, yearly, top_products, compare_products, compare_prices,ompare_trends, forecast หรือ all"),
        product_filter: str = Query(None, description="ระบุชื่อสินค้า เช่น 'Bagel' หรือ 'Donut' หากต้องการดูเฉพาะสินค้าใดสินค้าหนึ่ง"),
        time_filter: str = Query("yearly", description="เลือกช่วงเวลาสำหรับการเปรียบเทียบ: daily, monthly, yearly"),
        forecast_periods: int = Query(3, description="จำนวนเดือนที่ต้องการพยากรณ์ยอดขาย"),
        forecast_quantity: bool = Query(False, description="พยากรณ์จำนวนสินค้าที่ขาย"),
        ai_summary: bool = Query(False, description="ใช้ AI สรุปยอดขาย"),
):
    try:
        file = fs.get(ObjectId(file_id))
        df = pd.read_csv(file)
        
        df.columns = df.columns.str.strip()
        
        if "Date" not in df.columns or "Product" not in df.columns or "Quantity" not in df.columns or "Price" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV file must contain 'Date', 'Product', 'Quantity', and 'Price' columns")
        
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df = df.dropna(subset=["Date"])
        
        df["Total Sales"] = df["Quantity"] * df["Price"]
        
        if product_filter:
            df = df[df["Product"] == product_filter]
        
        results = {}
        if time_filter == "daily":
            sales_data = df.groupby([df["Date"].dt.date, "Product"]).agg({"Total Sales": "sum", "Quantity": "sum"}).reset_index()
        elif time_filter == "monthly":
            sales_data = df.groupby([df["Date"].dt.to_period("M"), "Product"]).agg({"Total Sales": "sum", "Quantity": "sum"}).reset_index()
        elif time_filter == "yearly":
            sales_data = df.groupby([df["Date"].dt.to_period("Y"), "Product"]).agg({"Total Sales": "sum", "Quantity": "sum"}).reset_index()
        else:
            raise HTTPException(status_code=400, detail="Invalid time filter. Choose from 'daily', 'monthly', or 'yearly'.")
        
        if report_type in ["all", "daily"]:
            daily_sales = df.groupby(df["Date"].dt.date).agg({"Total Sales": "sum", "Quantity": "sum"}).reset_index()
            daily_sales.rename(columns={"Quantity": "Quantity Sold"}, inplace=True)
            daily_sales["Date"] = daily_sales["Date"].astype(str)
            results["daily_sales"] = daily_sales.to_dict(orient="records")
        
        if report_type in ["all", "monthly"]:
            monthly_sales = df.groupby(df["Date"].dt.to_period("M")).agg({"Total Sales": "sum", "Quantity": "sum"}).reset_index()
            monthly_sales.rename(columns={"Quantity": "Quantity Sold"}, inplace=True)
            monthly_sales["Date"] = monthly_sales["Date"].astype(str)
            monthly_sales = monthly_sales.sort_values(by="Date")
            monthly_sales["Growth Rate (%)"] = monthly_sales["Total Sales"].pct_change() * 100
            monthly_sales = monthly_sales.fillna(0)
            results["monthly_sales"] = monthly_sales.to_dict(orient="records")
        
        if report_type in ["all", "yearly"]:
            yearly_sales = df.groupby(df["Date"].dt.to_period("Y")).agg({"Total Sales": "sum", "Quantity": "sum"}).reset_index()
            yearly_sales.rename(columns={"Quantity": "Quantity Sold"}, inplace=True)
            yearly_sales["Date"] = yearly_sales["Date"].astype(str)
            yearly_sales = yearly_sales.sort_values(by="Date")
            yearly_sales["Growth Rate (%)"] = yearly_sales["Total Sales"].pct_change() * 100
            yearly_sales = yearly_sales.fillna(0)
            results["yearly_sales"] = yearly_sales.to_dict(orient="records")
        
        if report_type in [ "top_products"]:
            top_products = df.groupby("Product").agg({"Total Sales": "sum", "Quantity": "sum"}).reset_index()
            top_products.rename(columns={"Quantity": "Quantity Sold"}, inplace=True)
            top_products = top_products.sort_values(by="Total Sales", ascending=False)
            results["top_products"] = top_products.to_dict(orient="records")
        
        if report_type in [ "compare_products"]:
            compare_products = df.groupby("Product").agg({"Total Sales": "sum", "Quantity": "sum", "Price": "mean"}).reset_index()
            compare_products.rename(columns={"Quantity": "Quantity Sold", "Price": "Average Price"}, inplace=True)
            results["compare_products"] = compare_products.to_dict(orient="records")
        
        if report_type in [ "compare_prices"]:
            compare_prices = df.groupby("Product")["Price"].describe().reset_index()
            results["compare_prices"] = compare_prices.to_dict(orient="records")

        sales_data.rename(columns={"Quantity": "Quantity Sold"}, inplace=True)
        sales_data["Date"] = sales_data["Date"].astype(str)
        results["sales_comparison"] = sales_data.to_dict(orient="records")
        
        if report_type in [ "compare_trends"]:
            trend_data = sales_data.copy()
            results["compare_trends"] = trend_data.to_dict(orient="records")
        
        if report_type in ["all", "forecast"]:
            monthly_sales = df.groupby(df["Date"].dt.to_period("M"))["Total Sales"].sum()
            model = ExponentialSmoothing(monthly_sales, trend="add", seasonal=None, damped_trend=True)
            fit_model = model.fit()
            forecast_values = fit_model.forecast(forecast_periods)
            
            forecast_df = forecast_values.reset_index()
            forecast_df.columns = ["Date", "Forecasted Sales"]
            forecast_df["Date"] = forecast_df["Date"].astype(str)
            results["forecast"] = forecast_df.to_dict(orient="records")
            
            if forecast_quantity:
                monthly_quantity = df.groupby(df["Date"].dt.to_period("M"))["Quantity"].sum()
                quantity_model = ExponentialSmoothing(monthly_quantity, trend="add", seasonal=None, damped_trend=True)
                quantity_fit_model = quantity_model.fit()
                quantity_forecast_values = quantity_fit_model.forecast(forecast_periods)
                
                quantity_forecast_df = quantity_forecast_values.reset_index()
                quantity_forecast_df.columns = ["Date", "Forecasted Quantity"]
                quantity_forecast_df["Date"] = quantity_forecast_df["Date"].astype(str)
                results["forecast_quantity"] = quantity_forecast_df.to_dict(orient="records")
        
        if ai_summary:
                # สรุปผลจาก AI (Gemini)
                prompt = f"กรุณาสรุปข้อมูลยอดขายและแนวโน้มของรายงานต่อไปนี้: {results} ขอสรุปเป็นภาษาไทย แต่สกุลเงินไม่ต้องบอก บอกแค่เลข"
                conversation_context = ""
                summary, updated_context = ask_gemini(prompt, conversation_context)  # เก็บคำตอบและบริบทที่อัปเดตจาก AI
                results["ai_summary"] = summary  # เก็บเฉพาะคำตอบจาก AI (summary)
                results["updated_context"] = updated_context
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing dashboard summary: {str(e)}")



# ให้เลือกจำนวนเงิน 100, 500 หรือ 1000 ผ่าน query parameter
@app.post("/create-payment-intent/")
async def create_payment_intent(amount: int = Query(..., ge=100, le=1000, description="Amount in THB (100, 500, or 1000)"), username: str = Query(..., description="Username of the user")):
    if amount not in [100, 500, 1000]:
        raise HTTPException(status_code=400, detail="Amount must be one of [100, 500, 1000]")

    try:
        # สร้างราคาของสินค้าก่อน (Price) โดยใช้สกุลเงิน THB
        price = stripe.Price.create(
            unit_amount=amount * 100,  # ราคาสินค้าในหน่วย cent (1 THB = 100 cents)
            currency="thb",  # สกุลเงินเป็น THB
            product_data={
                "name": f"Product for QR payment ({amount} THB)"
            }
        )

        # สร้าง Payment Link พร้อมกับ metadata ที่เก็บข้อมูล username
        payment_link = stripe.PaymentLink.create(
            line_items=[{
                "price": price.id,  # ใช้ ID ของ Price ที่เราสร้าง
                "quantity": 1
            }],
            metadata={"username": username, "amount": amount},  # ส่ง username ไปใน metadata
            after_completion={
                "type": "redirect",
                "redirect": {
                    "url": "https://a3c4-1-47-198-212.ngrok-free.app/home"  # เปลี่ยนเป็น URL ที่คุณต้องการให้ redirect หลังจากการชำระเงิน
                }
            }
        )

        # สร้าง QR Code สำหรับ Payment Link
        img = qrcode.make(payment_link.url)
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

        return JSONResponse({
            "message": f"Payment link created successfully for {amount} THB",
            "payment_link": payment_link.url,
            "qr_code": f"data:image/png;base64,{img_base64}"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating payment intent: {str(e)}")


@app.get("/dashboard/continue/", summary="ถามคำถามต่อจากการคำนวณยอดขาย", description="API สำหรับถามคำถามต่อจากการคำนวณยอดขายเดิม")
async def dashboard_continue(
        previous_summary: str = Query(..., description="สรุปยอดขายที่ได้รับจาก API ก่อนหน้า"),
        new_query: str = Query(..., description="คำถามใหม่ที่ต้องการถาม AI"),
):
    try:
        # ใช้ AI (Gemini) เพื่อตอบคำถามใหม่
        new_answer, updated_context = ask_gemini(previous_summary, new_query)  # ฟังก์ชันเรียก AI
        
        # ส่งคำตอบกลับพร้อมบริบทที่อัปเดต
        return JSONResponse(content={"new_answer": new_answer, "updated_context": updated_context})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing continue query: {str(e)}")





@app.get("/getfiles/", summary="ดึงไฟล์ทั้งหมดของผู้ใช้", description="API สำหรับดึงไฟล์ทั้งหมดที่ผู้ใช้ล็อกอินอัปโหลด")
async def get_files(current_user: str = Depends(get_current_user)):
    try:
        # ค้นหาไฟล์ทั้งหมดที่อัปโหลดโดยผู้ใช้ที่ล็อกอินอยู่
        files = list(fs.find({"username": current_user}))  # แปลง GridOutCursor เป็น list
        
        if len(files) == 0:
            raise HTTPException(status_code=404, detail="ไม่พบไฟล์ที่ผู้ใช้อัปโหลด")
        
        # ส่งไฟล์ทั้งหมดกลับไป
        file_list = [{
            "file_id": str(file._id),
            "filename": file.filename,
            "upload_time": file.upload_time
        } for file in files]
        
        return {"files": file_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

@app.post("/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get('Stripe-Signature')
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, WEBHOOK_SECRET
        )
        
        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']  # ดึงข้อมูล Checkout Session

            # ดึง username จาก metadata ของ Checkout Session
            username = session['metadata']['username']
            amount = session['metadata']['amount']
            amount = int(amount)
            # ค้นหาผู้ใช้ในฐานข้อมูล MongoDB โดยใช้ username
            user = users_collection.find_one({"username": username})
            if user:
                # สมมติว่าเราเพิ่มคะแนน 100 ให้กับผู้ใช้ที่ชำระเงินเสร็จ
                new_points = user.get("points", 0) + amount
                # อัปเดตข้อมูลผู้ใช้ใน MongoDB
                users_collection.update_one(
                    {"username": username},
                    {"$set": {"points": new_points}}
                )
                return {"message": "Points updated successfully!"}
            else:
                raise HTTPException(status_code=404, detail="User not found")
        else:
            return {"message": "Event not handled"}
        
    except ValueError as e:
        # ถ้าเกิดข้อผิดพลาดเกี่ยวกับการแปลงข้อมูล
        raise HTTPException(status_code=400, detail=f"Invalid payload: {str(e)}")
    except stripe.error.SignatureVerificationError as e:
        # ถ้าลายเซ็นต์ไม่ตรงกัน
        raise HTTPException(status_code=400, detail=f"Webhook signature verification failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing webhook: {str(e)}")


@app.get("/users/", summary="ดึงรายชื่อผู้ใช้ทั้งหมด", description="API สำหรับดึง username ทั้งหมดจากฐานข้อมูล")
async def get_all_usernames():
    users = users_collection.find({}, {"_id": 0, "username": 1})  # ดึงเฉพาะ username
    usernames = [user["username"] for user in users]
    return {"usernames": usernames}

@app.get("/user_info/", summary="ดึงข้อมูลผู้ใช้", description="ดึงข้อมูลของผู้ใช้ที่ล็อกอินอยู่จาก Token Authentication")
async def get_user_info(current_user: str = Depends(get_current_user)):
    try:
        # ค้นหาข้อมูลผู้ใช้จากตัวแปร current_user
        user = users_collection.find_one({"username": current_user})  # แปลงข้อมูลจากฐานข้อมูลเป็น dict
        
        if not user:
            raise HTTPException(status_code=404, detail="ไม่พบข้อมูลผู้ใช้")
        
        # จัดเตรียมข้อมูลที่จะส่งกลับ
        user_info = {
            "username": user['username'],  # ใช้ข้อมูลที่ต้องการจากฐานข้อมูล
            "points": user.get('points', 0)  # หากไม่มีข้อมูล points ให้ใช้ค่าเริ่มต้นเป็น 0
        }
        
        # ส่งข้อมูลกลับ
        return {"user_info": user_info}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 📊 ระบบวิเคราะห์ข้อมูลการขาย - Botnoi Hackathon

## 📋 คำอธิบายโปรเจ็ค

ระบบวิเคราะห์ข้อมูลการขายที่ใช้ FastAPI เป็น Backend API สำหรับจัดการและวิเคราะห์ข้อมูลไฟล์ CSV โดยมีฟีเจอร์ที่หลากหลาย รวมถึงการพยากรณ์ยอดขาย, การวิเคราะห์แนวโน้ม, และการสรุปข้อมูลด้วย AI

## ✨ ฟีเจอร์หลัก

### 👤 ระบบจัดการผู้ใช้
- **ลงทะเบียนผู้ใช้ใหม่** - สมัครสมาชิกด้วย username และ password
- **เข้าสู่ระบบ** - ระบบ Authentication ด้วย JWT Token
- **ระบบแต้มคะแนน** - ผู้ใช้ใหม่จะได้ 200 แต้ม เพื่อใช้ในการอัปโหลดไฟล์

### 📂 ระบบจัดการไฟล์
- **อัปโหลดไฟล์ CSV** - อัปโหลดไฟล์ข้อมูลการขาย (ใช้ 100 แต้มต่อครั้ง)
- **เก็บไฟล์ใน MongoDB GridFS** - จัดเก็บไฟล์อย่างปลอดภัย
- **ดึงรายชื่อไฟล์** - ดูไฟล์ทั้งหมดที่อัปโหลดไว้

### 📈 ระบบวิเคราะห์ข้อมูล
- **รายงานยอดขายรายวัน/เดือน/ปี** - สรุปยอดขายตามช่วงเวลาต่างๆ
- **สินค้าขายดี** - วิเคราะห์สินค้าที่มียอดขายสูงสุด
- **เปรียบเทียบสินค้า** - เปรียบเทียบยอดขายระหว่างสินค้าต่างๆ
- **เปรียบเทียบราคา** - วิเคราะห์ความแตกต่างของราคาสินค้า
- **แนวโน้มการเติบโต** - คำนวณอัตราการเติบโตของยอดขาย

### 🔮 ระบบพยากรณ์
- **พยากรณ์ยอดขาย** - ใช้ Exponential Smoothing และ SARIMA model
- **พยากรณ์จำนวนสินค้า** - ทำนายจำนวนสินค้าที่จะขายได้
- **กรองตามสินค้า** - พยากรณ์เฉพาะสินค้าที่ต้องการ

### 🤖 ระบบ AI
- **สรุปข้อมูลด้วย Google Gemini AI** - วิเคราะห์และสรุปผลลัพธ์เป็นภาษาธรรมชาติ
- **ถาม-ตอบแบบต่อเนื่อง** - สามารถถามคำถามเพิ่มเติมจากผลการวิเคราะห์

### 💳 ระบบการชำระเงิน
- **Stripe Payment Integration** - ซื้อแต้มเพิ่ม (100, 500, 1000 บาท)
- **QR Code Payment** - สร้าง QR Code สำหรับการชำระเงิน
- **Webhook** - รับการแจ้งเตือนเมื่อการชำระเงินสำเร็จ

## 🛠️ เทคโนโลยีที่ใช้

### Backend Framework
- **FastAPI** - Python Web Framework ที่รวดเร็วและทันสมัย
- **Uvicorn** - ASGI Server สำหรับ FastAPI

### ฐานข้อมูล
- **MongoDB** - NoSQL Database
- **PyMongo** - MongoDB Driver สำหรับ Python
- **GridFS** - สำหรับจัดเก็บไฟล์

### การวิเคราะห์ข้อมูล
- **Pandas** - Data Analysis Library
- **Matplotlib** - Data Visualization
- **Statsmodels** - Statistical Analysis (Exponential Smoothing, SARIMA)
- **pmdarima** - Auto ARIMA Model Selection
- **Scikit-learn** - Machine Learning Metrics

### Authentication & Security
- **JWT (JSON Web Tokens)** - Token-based Authentication
- **Passlib** - Password Hashing
- **bcrypt** - Secure Password Hashing Algorithm

### AI Integration
- **Google Generative AI (Gemini)** - AI สำหรับสรุปและตอบคำถาม

### Payment Processing
- **Stripe** - Payment Gateway
- **QRCode** - QR Code Generation

### Other Libraries
- **python-dotenv** - Environment Variables Management
- **pydantic** - Data Validation

## 🚀 วิธีการติดตั้งและเรียกใช้

### 1. ติดตั้ง Dependencies

```bash
pip install fastapi uvicorn pandas pymongo python-dotenv PyJWT passlib bcrypt python-multipart matplotlib statsmodels pmdarima scikit-learn qrcode stripe google-generativeai
```

### 2. ตั้งค่า Environment Variables

สร้างไฟล์ `.env` และเพิ่มข้อมูลต่อไปนี้:

```env
MONGO_URL=your_mongodb_connection_string
SECRET_KEY=your_jwt_secret_key
STRIPE_SECRET_KEY=your_stripe_secret_key
STRIPE_WEBHOOK_SECRET=your_stripe_webhook_secret
API_KEY=your_google_gemini_api_key
```

### 3. เรียกใช้งาน

```bash
python main.py
```

หรือ

```bash
uvicorn main:app --reload
```

API จะทำงานที่ `http://localhost:8000`

## 📚 API Documentation

เมื่อเรียกใช้งานแล้ว สามารถดู API Documentation ได้ที่:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 📊 รูปแบบไฟล์ CSV ที่รองรับ

ไฟล์ CSV ต้องมีคอลัมน์ดังต่อไปนี้:
- `Date` - วันที่ขาย (รูปแบบ: YYYY-MM-DD)
- `Product` - ชื่อสินค้า
- `Quantity` - จำนวนที่ขาย
- `Price` - ราคาต่อหน่วย

ตัวอย่าง:
```csv
Date,Product,Quantity,Price
2024-01-01,Bagel,10,25.0
2024-01-01,Donut,15,20.0
2024-01-02,Bagel,8,25.0
```

## 🔑 API Endpoints หลัก

### Authentication
- `POST /register/` - สมัครสมาชิก
- `POST /login/` - เข้าสู่ระบบ

### File Management
- `POST /upload/` - อัปโหลดไฟล์ CSV
- `GET /getfiles/` - ดึงรายชื่อไฟล์ทั้งหมด

### Data Analysis
- `GET /dashboard/summary/` - วิเคราะห์และสรุปข้อมูลการขาย
- `GET /dashboard/continue/` - ถามคำถามเพิ่มเติมกับ AI

### Payment
- `POST /create-payment-intent/` - สร้าง Payment Intent สำหรับซื้อแต้ม
- `POST /webhook` - Stripe Webhook

### User Management
- `GET /users/` - ดึงรายชื่อผู้ใช้ทั้งหมด
- `GET /user_info/` - ดึงข้อมูลผู้ใช้ปัจจุบัน

## 💡 การใช้งาน

1. **สมัครสมาชิก** และเข้าสู่ระบบเพื่อรับ JWT Token
2. **อัปโหลดไฟล์ CSV** ข้อมูลการขาย (ใช้ 100 แต้ม)
3. **วิเคราะห์ข้อมูล** ด้วย API `/dashboard/summary/`
4. **ซื้อแต้มเพิ่ม** ผ่าน Stripe เมื่อแต้มหมด
5. **ใช้ AI** สรุปและตอบคำถามเกี่ยวกับข้อมูลการขาย

## 🏗️ โครงสร้างโปรเจ็ค

```
Botnoi_hackathon/
├── main.py              # ไฟล์หลักของ API
├── .env                 # Environment Variables (ไม่ถูก commit)
├── .gitignore          # Git ignore file
└── README.md           # ไฟล์นี้
```

## 👥 ผู้พัฒนา

โปรเจ็คนี้พัฒนาขึ้นสำหรับ **Botnoi Hackathon**

## 📄 License

โปรเจ็คนี้อยู่ภายใต้ MIT License 
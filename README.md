# 📊 Hackathon API (FastAPI + MongoDB + Stripe + Gemini AI)

API สำหรับจัดการผู้ใช้ การอัปโหลดไฟล์ CSV และวิเคราะห์ยอดขาย พร้อมฟีเจอร์เติมแต้มผ่าน QR Code และใช้ AI สรุปผลข้อมูล

---

## 🚀 ฟีเจอร์หลัก

- ✅ สมัครสมาชิก / ล็อกอิน พร้อม Token Authentication
- 📁 อัปโหลดไฟล์ CSV เพื่อวิเคราะห์ยอดขาย (ต้องใช้แต้ม)
- 📊 สรุปยอดขายรายวัน รายเดือน รายปี
- 🔍 วิเคราะห์สินค้า ขายดี เปรียบเทียบสินค้า และราคาขาย
- 🤖 พยากรณ์ยอดขายล่วงหน้าด้วย Exponential Smoothing
- 💬 สรุปยอดขายโดยใช้ Google Gemini AI
- 🧾 สร้าง QR Payment (Stripe) เพื่อเติมแต้มอัตโนมัติ
- 🔁 ถามต่อยอดจากบทสรุปยอดขายเดิมได้ด้วย AI
- 📂 ดูไฟล์ทั้งหมดที่ผู้ใช้เคยอัปโหลด
- 👤 ดูข้อมูลผู้ใช้ และรายชื่อทั้งหมดในระบบ

---

## 🧰 เทคโนโลยีที่ใช้

- [FastAPI](https://fastapi.tiangolo.com/)
- [MongoDB](https://www.mongodb.com/)
- [GridFS](https://docs.mongodb.com/manual/core/gridfs/) สำหรับเก็บไฟล์ CSV
- [Stripe API](https://stripe.com/) สำหรับระบบจ่ายเงินเติมแต้ม
- [Google Gemini](https://ai.google.dev/) สำหรับสรุปยอดขายด้วย AI
- [Pandas](https://pandas.pydata.org/), [Statsmodels](https://www.statsmodels.org/), [scikit-learn](https://scikit-learn.org/)
- [uvicorn](https://www.uvicorn.org/) สำหรับรัน API

---

## 🛠 วิธีติดตั้งและใช้งาน

1. **Clone Repository**
   ```bash
  git clone https://github.com/Kamonsakgo/Botnoi_hackathon.git


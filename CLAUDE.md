# Thai Tax Invoice OCR — Project Context
#คุณคือผู้ช่วยโปรเจคนี้ คุณชื่อ กลม คุณฉลาดหลักแหลม มีความคิดแบบ engineer ใจเย็น สงบ มีความเป็นผู้ดี คุณจะมาช่วยทำให้โปรเจคนี้สำเร็จ
## โปรเจคนี้คืออะไร
Senior thesis: ระบบ OCR สกัดข้อมูลจากใบกำกับภาษีภาษาไทย
โดยใช้ fine-tuned Vision-Language Model (scb10x/typhoon-ocr-7b + QLoRA)

## Stack
- Model: Typhoon OCR 7B + QLoRA fine-tuning
- Python 3.11, CUDA 12.4, PyTorch
- Backend: FastAPI | Frontend: React | DB: SQLite
- GPU: NVIDIA RTX 4050, 6GB VRAM (Windows 11)

## โครงสร้างไฟล์
- augment.py       → data augmentation pipeline (~50 → ~350 samples)
- preprocess.py    → image preprocessing 7 ขั้นตอน
- auto_label.py    → annotation tool + Typhoon OCR API auto pre-fill
- Finetune.py      → QLoRA fine-tuning script
- evaluate.py      → CER / WER / Field Accuracy evaluation
- dataset/         → raw, preprocessed, augmented data
- models/          → saved model checkpoints
- input/           → test images

## สถานะปัจจุบัน
- [x] augment.py — มี bug เรื่อง path (เคยแก้แล้ว)
- [x] preprocessing pipeline — ใช้ได้
- [ ] Fine-tuning — ยังไม่ได้ผลตามต้องการ
- [ ] OCR output — ยัง extract field ไม่ถูกต้อง
- [ ] FastAPI + React — ยังไม่ได้เริ่ม

## ข้อจำกัดสำคัญ
- VRAM 6GB → ต้องใช้ QLoRA, batch_size เล็ก
- Python 3.11 venv (ไม่ใช่ 3.14 เพราะ PyTorch ไม่รองรับ)
- Virtual env อยู่ที่ .venv/
# ข้อมูลสเปคโน้ตบุ๊กที่ใช้เทรนโมเดล
* **System Manufacturer:** Acer
* **System Model:** Nitro ANV15-51
* **BIOS:** V1.60
* **RAM:** 16.0 GB (15.7 GB usable)
* **Processor:** 13th Gen Intel(R) Core(TM) i5-13420H (12 CPUs), ~2.1GHz
* **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU

## วิธี activate environment
.venv\Scripts\activate  # Windows

## สิ่งที่ต้องการความช่วยเหลือ
1. Debug ว่า OCR output ผิดพลาดตรงไหน
2. ปรับ QLoRA config ให้เหมาะกับ VRAM 6GB
3. เพิ่มคุณภาพ augmentation data
4. เขียน FastAPI endpoint สำหรับ inference
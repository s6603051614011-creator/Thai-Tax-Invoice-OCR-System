# สรุปสิ่งที่เพิ่มมาใหม่ — เปลี่ยนฐานเป็น Typhoon-OCR 1.5 (2B)
### อัพเดต 2026-07-08 (ต่อจาก PROGRESS_SUMMARY.md)

## 1. เป้าหมายรอบนี้

ทดลองเปลี่ยนฐานโมเดลจาก **Qwen2.5-VL-3B** (Qwen เปล่า) → **typhoon-ai/typhoon-ocr1.5-2b**
(Typhoon OCR ตัวจริง เทรนมาอ่านเอกสารไทยแล้ว) เพื่อ (ก) ทำให้โมเดลเก่งขึ้น
(ข) ตรง scope วิทยานิพนธ์ "นำ Typhoon มาพัฒนาต่อ" เต็มปาก

## 2. งานแก้โค้ด (รองรับ 2 สถาปัตยกรรม)

Typhoon 1.5 เป็น **Qwen3-VL** (คนละรุ่นกับ Qwen2.5-VL เดิม) ต้องแก้:

| ไฟล์ | สิ่งที่แก้ |
|---|---|
| `schema.py` | เพิ่ม `get_model_class()`/`get_processor_class()` — เลือก class อัตโนมัติจาก config (Auto\* ของ transformers) สลับฐาน 2.5-VL ↔ 3-VL ได้ด้วยการเปลี่ยน `BASE_MODEL_ID` บรรทัดเดียว |
| `Finetune.py`, `evaluate.py`, `format_dataset.py`, `api.py` | เลิก hardcode `Qwen2_5_VL*` → เรียกผ่าน `schema.get_*_class()` แทน |
| `Finetune.py` (dataset+collate) | เพิ่มส่ง `mm_token_type_ids` (Qwen3-VL M-RoPE ต้องการ — Qwen2.5-VL ไม่ต้อง เงื่อนไขจึงเป็น no-op กับฐานเดิม ไม่ทำ path 3B พัง) |

**บั๊กที่เจอระหว่างทาง:** Qwen3-VL crash ที่ step 0 เพราะขาด `mm_token_type_ids`
(field ที่ processor คืนมาบอกว่า token ไหนเป็นรูป/ข้อความ) — แก้ที่ dataset +
collate แล้วเทรนผ่าน

## 3. วัด VRAM/token จริงก่อนตั้งค่า (ไม่เดา)

Qwen3-VL ใช้ patch 16px (= 32px/token) ต่างจาก Qwen2.5-VL ที่ 14px (28px/token)
→ สูตรพิกเซลเปลี่ยน วัดจริงด้วย smoke test:

- โหลด 4-bit: **2.06GB** (เดิม 3B ใช้ 2.83GB — เบากว่า)
- LoRA targets เดิมใช้กับ Qwen3-VL ได้ (trainable 8.7M / 0.71%)
- ที่ seq=1280: peak VRAM **5.75GB คงที่ทุกความละเอียด** (seq เป็นตัวคุม, 1536→6.38GB เกินเพดาน)

**config ที่เลือก:** MAX_PIXELS = 320tok (32×32), MIN_PIXELS = 64tok, MAX_SEQ_LEN = 1280
→ ละเอียดกว่า 3B เดิม 25% แต่ VRAM ยังปลอดภัย

## 4. ข้อมูลเทรน

รัน `format_dataset.py` ใหม่ที่ config นี้ → drop แค่ **2.0%** (21 ใบ)
→ train **912** / val **102** / test 50 (มากกว่า 3B ที่ 810/90 เพราะ seq=1280 กว้างกว่า)

## 5. ผลเทรน

- เทรน ~5 ชม. early-stop ที่ epoch ~5.7 (best eval_loss **0.247** ที่ step 400)
- เทียบ 3B ที่ดีสุด 0.304 → **2B ต่ำกว่า ~19%** และแตะจุดนี้ได้ตั้งแต่ epoch 2
  (ฐาน Typhoon + ความละเอียดสูงกว่า ช่วยให้เรียนเร็ว)
- โมเดล 3B เดิม backup ไว้ที่ `models/best_model_qwen3b_group2` (ไม่ถูกทับ)

## 6. ผลเปรียบเทียบ (test set 50 ใบเดียวกัน)

### 6.1 ก่อน/หลัง fine-tune (เท่าเทียมสมบูรณ์ — โมเดลตัวเดียวกัน ต่างแค่เทรน)

| ตัวชี้วัด | Typhoon 2B **ยังไม่เทรน** | Typhoon 2B **fine-tuned** |
|---|---|---|
| Field Accuracy | 2.6% | **42.8%** |
| JSON valid | 32% | **100%** |
| CER (↓ดี) | 60.1% | **17.5%** |
| Items CER (↓ดี) | 96.3% | **43.1%** |

### 6.2 Typhoon 2B vs Qwen 3B (fine-tuned ทั้งคู่) — 2B ชนะทุกตัว

| ตัวชี้วัด | **Typhoon 2B (ใหม่)** | Qwen 3B (เดิม) |
|---|---|---|
| Field Accuracy | **42.8%** | 40.3% |
| CER (↓ดี) | **17.5%** | 21.0% |
| JSON valid | **100%** | 98% |
| Items CER (↓ดี) | **43.1%** | 55.7% |
| Items count | **88.0%** | 72.0% |

จุดเด่นสุด: การอ่าน **รายการสินค้า** (items) ดีขึ้นชัด — count 72%→88%, CER 55.7%→43.1%
และ JSON ใช้งานได้ **100% ทุกใบ**

## 7. ข้อควรระวังเรื่องความยุติธรรม (สำหรับเล่ม)

- **6.1 เท่าเทียม 100%** — โมเดลตัวเดียวกัน โหลด 4-bit เหมือนกัน prompt/eval เดียวกัน
- **6.2 มี confound** — 2B ได้เปรียบเรื่องความละเอียด (320 vs 256 tok) + ข้อมูล (912 vs 810)
  ด้วย จึงสรุปไม่ได้ว่าชนะเพราะ "ฐาน Typhoon ดีกว่า" ล้วนๆ ที่ถูกคือ "pipeline 2B
  ที่ดีที่สุด ชนะ pipeline 3B ที่ดีที่สุด" (ตอบคำถามเลือกโมเดลไปใช้ได้ แต่ไม่ได้
  แยกผลของฐานเดี่ยวๆ — ถ้าอยากได้ต้องเทรนทั้งคู่ที่ res/data เท่ากันเป๊ะ)
- หมายเหตุ: baseline 2B ในเครื่อง (2.6%) ต่ำกว่า typhoon-ocr-v1.5 ผ่าน API (8.5%)
  เพราะในเครื่องเป็น 4-bit ส่วน API full precision + รูปเต็ม — แต่ 2.6% คือ before/after
  ที่ยุติธรรมที่สุดเพราะคุมทุกอย่างเท่ากัน

## 8. ไฟล์/ผลลัพธ์ใหม่

- `models/best_model/` = Typhoon 2B fine-tuned (โมเดลใช้งานล่าสุด)
- `models/best_model_qwen3b_group2/` = 3B เดิม (backup เทียบ)
- `results/ocr_evaluation_20260708_0037.xlsx` + `results_20260708_0037.json` = ผล 2B
- `model_compare_2b_vs_3b.html` = หน้าเทียบ 4 คอลัมน์ (เฉลย / 2B ดิบ / 2B fine-tuned /
  3B) พร้อมรูปใบจริง 5 ใบ — เปิดในเบราว์เซอร์ (gitignore ไว้ เพราะมีรูปเอกสารจริง)

## 9. สถานะ scope วิทยานิพนธ์

ตอนนี้ทำ **"นำ Typhoon OCR มา fine-tune ต่อ"** ได้จริงตามตัวอักษรแล้ว —
ไม่ต้องแก้ถ้อยคำ scope และได้โมเดลที่เก่งกว่ารอบก่อนด้วย

## 10. ยังไม่ได้ทำ / งานถัดไป

- [ ] Commit + push โค้ดรอบนี้ (รอผู้ใช้สั่ง) — โมเดล/ข้อมูล/HTML ยัง gitignore
- [ ] (ถ้าต้องการเทียบฐานบริสุทธิ์) เทรน 3B ที่ 320tok/912 ใบ ให้เท่า 2B เป๊ะ
- [ ] Revoke HF token เก่า + rotate Typhoon API key (ค้างจากรอบก่อน)
- [ ] React frontend (FastAPI `api.py` พร้อมแล้ว รองรับ 2B อัตโนมัติผ่าน schema)

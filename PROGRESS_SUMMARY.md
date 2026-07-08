# สรุปความคืบหน้า — Thai Tax Invoice OCR (อัพเดต 2026-07-07)

## 1. โมเดลปัจจุบัน (Group 2)

Fine-tune **Qwen2.5-VL-3B-Instruct** ด้วย QLoRA บน RTX 4050 6GB — เทรน 6.1 ชม. early-stop ที่ eval_loss 0.304 โมเดลอยู่ที่ `models/best_model/`

การแก้ไขสำคัญก่อนเทรนรอบนี้:
- แก้เฉลย 6 ใบ: seller_tax_id 5 ใบที่ไม่ผ่าน mod-11 checksum (ซูมภาพยืนยันทีละใบ) + subtotal ของ inv_345
- แก้บั๊ก **EXIF orientation** — 26/402 ภาพ (รวม 5/50 test) ถูกป้อนเข้าโมเดลผิดทิศทางมาตลอด แก้ครบ 3 จุด decode (`Finetune.py`, `evaluate.py`, `format_dataset.py`)
- เพิ่มความละเอียด 128→**256px** + `MAX_SEQ_LEN` 960→**1024** (วัด VRAM จริง: peak 6.17GB ปลอดภัยใต้เพดาน 6.36GB ที่เคยเทรนผ่าน)
- dataset ใหม่: train 810 / val 90 / test 50

## 2. ตัวเปรียบเทียบทั้งหมด (test set 50 ใบเดียวกัน prompt+เกณฑ์วัดเดียวกัน)

| โมเดล | Field Acc | 95% CI | CER (↓) | JSON valid |
|---|---|---|---|---|
| Baseline (Qwen 3B ยังไม่เทรน) | 3.3% | [1.6, 5.5] | 49.4% | 76% |
| typhoon-ocr 7B (API, zero-shot) | 6.8% | [3.9, 10.3] | 73.0% | 50% |
| typhoon-ocr-v1.5 (API, zero-shot) | 8.5% | [5.6, 12.1] | 79.6% | 58% |
| Tesseract (value-found*) | 30.9%* | [24.4, 37.7] | 56.3%* | ไม่รองรับ |
| **Fine-tuned (ของเรา)** | **40.3%** | **[32.0, 48.7]** | **21.0%** | **98%** |

*Tesseract ไม่รู้จัก field — วัดด้วย value-alignment (ใจดีกับ Tesseract กว่าเกณฑ์ปกติ) เทียบเคียงได้แต่ไม่ใช่ตัวเดียวกันเป๊ะ

**นัยสำคัญทางสถิติ (paired bootstrap B=10,000):** ผลต่าง Field Accuracy ของโมเดลเรากับทุกคู่แข่ง CI ไม่คร่อมศูนย์ทุกคู่ — ต่ำสุดคือ [+22.8%, +40.6%] (เทียบ v1.5) ส่วน typhoon-ocr กับ v1.5 กันเอง CI ทับกัน → สรุปไม่ได้ว่าใครดีกว่า

## 3. การทดลองความยุติธรรม (fairness follow-ups)

**คำถาม:** คะแนน Typhoon ต่ำเพราะ "อ่านไม่ออก" หรือ "ทำตาม schema ไม่ได้"?

**การทดลอง:** ให้ typhoon-ocr ทำงานด้วย prompt ถนัดของมันเอง (markdown/natural_text + รูป 1800px ตาม model card) แล้ววัดด้วย value-alignment ตัวเดียวกับ Tesseract — และวัดโมเดลเราด้วยตัวชี้วัดเดียวกันเพื่อเทียบตรงๆ

| Value Found Rate (อ่านค่าเจอไหม ไม่สนโครงสร้าง) | ค่ากลาง | 95% CI |
|---|---|---|
| Tesseract | 30.9% | [24.4, 37.7] |
| Fine-tuned (เรา) | 46.3% | [39.9, 52.7] |
| **Typhoon-native** | **58.0%** | **[52.3, 63.4]** |

**ข้อสรุปที่ซื่อสัตย์:** Typhoon *อ่าน* เก่งกว่าเรา (ตัวใหญ่กว่า 2 เท่า + ได้รูป 1800px ขณะเราถูกบีบด้วย VRAM) แต่*แยก field ตาม schema ไม่ได้เลย* (6.8% vs 40.3%) → contribution ของงานนี้คือ**การสอนโมเดลให้เข้าใจโครงสร้างใบกำกับภาษี** ไม่ใช่อ่านตัวอักษรเก่งกว่าใคร

**จุดที่ต้อง disclose ในเล่ม:**
1. Prompt เป็น "สนามเหย้า" ของโมเดลเรา (แก้ด้วยการทดลอง native prompt ข้างบน)
2. Test set มาจากกองเอกสารบริษัทเดียวกับ train (ผู้ขาย/layout หน้าซ้ำ) — 40.3% คือความแม่นกับกระแสเอกสารจริงของบริษัทนี้ ไม่ใช่ผู้ขายที่ไม่เคยเห็น
3. อคติที่เหลือส่วนใหญ่เอียงไปทาง**เอื้อคู่แข่ง** (API ได้รูปเต็ม, max_tokens 4096 vs 768 ของเรา) → ชัยชนะของเรา conservative

## 4. ประเด็น Typhoon กับ scope วิทยานิพนธ์

- Typhoon-OCR = Qwen2.5-VL ที่ SCB10X เทรนต่อ — เราใช้ Qwen 3B เปล่าเพราะ 7B เกิน VRAM → มีความเสี่ยงหลุด scope "นำ Typhoon มาพัฒนาต่อ" ถ้าตีความตามตัวอักษร
- **ค้นพบ `typhoon-ai/typhoon-ocr-3b`** — Typhoon ตัวจริง สถาปัตยกรรม+ขนาดเดียวกับที่ใช้อยู่เป๊ะ → เปลี่ยน `BASE_MODEL_ID` บรรทัดเดียวแล้วเทรนใหม่ได้เลย ค่า VRAM ที่วัดไว้ใช้ได้หมด **(แผนถัดไปที่แนะนำ — แก้ปัญหา scope ตรงๆ)**
- `typhoon-ocr1.5-2b` ก็เทรนต่อได้ (transformers 5.6.2 รองรับ Qwen3-VL แล้ว, 4-bit ~1.7GB เบากว่าปัจจุบัน) แต่ต้องแก้ class 4 ไฟล์ + วัด token/VRAM ใหม่ (patch 16px ไม่ใช่ 14px)

## 5. Security ที่แก้ไป

- ลบ HF token ที่ hardcode ใน `Finetune.py` (หลุดขึ้น GitHub แล้ว — **ต้อง revoke ที่ huggingface.co เอง ถ้ายังไม่ได้ทำ**)
- ลบ Typhoon API key ที่ hardcode ใน `auto_label.py` (หลุดตั้งแต่ commit แรก) → ย้ายเป็น env var `TYPHOON_API_KEY` — key เดิมยังใช้งานได้อยู่ ควรพิจารณา rotate
- `.gitignore` กัน `typhoon_compare_examples.html` (ฝังรูปใบจริง) และ `.tessdata/`

## 6. เครื่องมือ/เอกสารที่สร้างเพิ่ม

| ไฟล์ | ทำอะไร |
|---|---|
| `postprocess.py` | ตรวจผล OCR โดยไม่ต้องมีเฉลย (checksum เลขภาษี + เลขคณิตใบกำกับ) precision 97% |
| `eval_tesseract.py` | เทียบ Tesseract ด้วย value-alignment metric |
| `eval_typhoon.py` | เทียบ Typhoon ทุกรุ่นผ่าน API (`--model` เลือกได้) ด้วยเกณฑ์เดียวกับ evaluate.py |
| `eval_typhoon_native.py` | ทดลอง Typhoon ด้วย prompt ถนัดของมันเอง |
| `bootstrap_ci.py` | 95% CI แบบ paired bootstrap ของทุก metric (ทำซ้ำได้ seed=42) |
| `advisor_update_script.md` | สคริปต์พูดอัพเดตอาจารย์ |
| `demo_guide.md` | คู่มือรันสาธิต + หลักการทำงานทุกไฟล์ |
| `typhoon_compare_examples.html` | เทียบ 3 ตัวอย่างจริงพร้อมรูป (เปิดในเบราว์เซอร์ — ห้าม push) |

## 7. การใช้งานจริง (คำถามที่ตอบไปแล้ว)

- **ต้องมี GPU ตอน inference ด้วย** (~2.5GB VRAM) — โค้ด raise error ถ้าไม่มี CUDA
- ความเร็ว ~52 วิ/ใบ บนเครื่องนี้ → พนักงานป้อน 10 ใบ ≈ 9 นาที (โมเดลโหลดค้างไว้)
- ทางเลือกถ้าไม่มี GPU: ใช้โน้ตบุ๊กนี้เป็น server / เช่า serverless GPU / แปลง GGUF รันบน CPU (ช้ากว่ามาก)

## 8. สถานะ git

push ครบถึง `1ac4ea9` — commits หลักรอบนี้: `b80101c` (Group 2), `aed4205` (Tesseract), `edf93a1` (แก้ MIN_PIXELS api.py + ลบ Typhoon key), `54d89e9` (Typhoon API comparison), `1ac4ea9` (bootstrap CI + native prompt)

## 9. งานถัดไป (เรียงตามที่แนะนำ)

1. **เทรน typhoon-ocr-3b** (แก้บรรทัดเดียว ~6 ชม.) — ตรง scope + ฐานอ่านเก่งกว่า อาจได้ทั้งสองด้าน
2. ทดลอง typhoon-ocr1.5-2b เป็นการทดลองที่สอง (ต้องแก้โค้ดมากกว่า)
3. Revoke HF token เก่า + rotate Typhoon API key
4. Group 3 (เช่า cloud GPU เทรน resolution สูงขึ้น) ถ้าต้องการดันตัวเลขต่อ
5. FastAPI มีแล้ว (`api.py`) — เหลือ React frontend

"""
eval_tesseract.py -- เทียบ Tesseract (traditional OCR) เป็นตัวที่ 3
=====================================================================
Tesseract ไม่เข้าใจ schema/โครงสร้างใบกำกับ -- มันแค่ dump ตัวหนังสือทุกอย่าง
ที่เห็นบนหน้ากระดาษ (หัวจดหมาย, เส้นตาราง, ลายน้ำ, label ต่างๆ) ออกมาเป็นก้อน
เดียว ไม่แยก field ดังนั้นการเทียบ CER/WER แบบ "ทั้งก้อน vs ค่าฟิลด์ล้วนๆ"
จะไม่ยุติธรรม (จะเจอ insertion error พุ่งจากข้อความที่ไม่ใช่ field เยอะมาก)

วิธีที่ยุติธรรมกว่า: เอาค่าแต่ละ field ใน ground truth ไปหา "ตำแหน่งที่ใกล้เคียง
ที่สุด" ในข้อความดิบที่ Tesseract อ่านได้ (fuzzy substring alignment) แล้ววัด
CER เฉพาะช่วงนั้น -- เทียบเท่ากับถาม "ค่านี้ Tesseract อ่านถูกไหม (ไม่สนตำแหน่ง/
โครงสร้าง)" ซึ่งเทียบเคียงกับ Field Accuracy ของ VLM ได้ตรงกว่า
(ยังรายงาน raw full-text CER/WER ไว้ด้วยเป็นข้อมูลเสริม แต่ระบุชัดว่าไม่ยุติธรรม)

รัน: python eval_tesseract.py
ต้องมี: Tesseract OCR ติดตั้งแล้ว + .tessdata/tha.traineddata, eng.traineddata
"""

import re
import sys
import json
import difflib
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
import schema
from evaluate import calc_cer, calc_wer  # ใช้สูตรเดียวกับ evaluate.py เป๊ะ

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pytesseract
from PIL import Image, ImageOps
from tqdm import tqdm

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA_DIR  = str(Path(__file__).parent / ".tessdata")
TEST_JSONL    = "dataset/formatted/test.jsonl"
RAW_DIR       = Path("dataset/raw")  # อ่านรูปต้นฉบับความละเอียดเต็ม (ไม่ผ่าน resize เหมือน VLM)
OUT_DIR       = Path("results")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
TESS_CONFIG = f"--tessdata-dir {TESSDATA_DIR}"


def flatten_fields_to_text(fields: dict) -> str:
    """แปลง fields เป็นข้อความล้วน (ไม่มี JSON syntax) สำหรับเทียบกับ raw text
    ที่ Tesseract อ่านได้ -- เทียบ text-to-text ตรงๆ ไม่เอาเปรียบ Tesseract
    ที่ไม่รู้จักโครงสร้าง JSON"""
    lines = []
    for k in schema.SCALAR_FIELDS:
        v = fields.get(k, "")
        if v:
            lines.append(str(v))
    for item in fields.get("items", []):
        for k in schema.ITEM_FIELDS:
            v = item.get(k, "")
            if v:
                lines.append(str(v))
    for k in schema.SUMMARY_FIELDS:
        v = fields.get(k, "")
        if v:
            lines.append(str(v))
    return "\n".join(lines)


def run_tesseract(image_path: str) -> str:
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    return pytesseract.image_to_string(img, lang="tha+eng", config=TESS_CONFIG)


def _normalize(s: str) -> str:
    """ตัด whitespace ทั้งหมดออก -- Tesseract อ่านภาษาไทยมักแทรกช่องว่างระหว่าง
    ตัวอักษรผิดๆ (เช่น "บ ริ ษ ั ท" แทน "บริษัท") ถ้าไม่ตัดออกจะโดนนับเป็น error
    ทั้งที่จริงอ่านตัวอักษรถูก"""
    return re.sub(r"\s+", "", s)


def value_cer(value: str, haystack: str) -> float | None:
    """หาตำแหน่งที่ใกล้เคียงค่า value มากที่สุดใน haystack (fuzzy substring)
    แล้ววัด CER เทียบกับช่วงนั้น -- คืน None ถ้า value ว่าง"""
    v = _normalize(value)
    h = _normalize(haystack)
    if not v:
        return None
    if not h:
        return 1.0
    sm = difflib.SequenceMatcher(None, h, v, autojunk=False)
    match = sm.find_longest_match(0, len(h), 0, len(v))
    center = match.a + match.size // 2
    start  = max(0, center - len(v) // 2)
    end    = min(len(h), start + len(v))
    start  = max(0, end - len(v))
    window = h[start:end]
    return calc_cer(window, v)


def eval_value_recall(fields: dict, haystack: str, threshold: float = 0.3) -> tuple:
    """คืน (avg_cer_ต่อ_value, found_rate) จากทุก field ที่ไม่ว่างใน fields
    found = CER ของ value นั้น < threshold"""
    cers = []
    for k in schema.SCALAR_FIELDS + schema.SUMMARY_FIELDS:
        c = value_cer(fields.get(k, ""), haystack)
        if c is not None:
            cers.append(c)
    for item in fields.get("items", []):
        for k in schema.ITEM_FIELDS:
            c = value_cer(item.get(k, ""), haystack)
            if c is not None:
                cers.append(c)
    if not cers:
        return None, None
    found = sum(1 for c in cers if c < threshold) / len(cers)
    return sum(cers) / len(cers), found


def main():
    test_samples = []
    with open(TEST_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_samples.append(json.loads(line))
    print(f"✓ {len(test_samples)} test samples")

    results = []
    for sample in tqdm(test_samples, desc="Tesseract"):
        sid       = sample["id"]
        image_path = RAW_DIR / f"{sid}.jpg"
        gt_text    = sample["messages"][2]["content"]
        gt_fields  = schema.parse_model_json(gt_text)
        gt_flat    = flatten_fields_to_text(gt_fields)

        try:
            pred_text = run_tesseract(image_path)
        except Exception as e:
            print(f"  ⚠️  {sid}: {e}")
            pred_text = ""

        value_avg_cer, value_found_rate = eval_value_recall(gt_fields, pred_text)

        results.append({
            "id":              sid,
            "pred":            pred_text,
            "gt":              gt_flat,
            "raw_cer":         calc_cer(pred_text, gt_flat),   # อ้างอิงเสริม ไม่ยุติธรรม (ดู docstring)
            "raw_wer":         calc_wer(pred_text, gt_flat),
            "value_cer":       value_avg_cer,                  # ตัวเทียบหลัก
            "value_found_rate": value_found_rate,
        })

    n = len(results)
    avg_raw_cer   = sum(r["raw_cer"] for r in results) / n
    avg_raw_wer   = sum(r["raw_wer"] for r in results) / n
    vc = [r["value_cer"] for r in results if r["value_cer"] is not None]
    vf = [r["value_found_rate"] for r in results if r["value_found_rate"] is not None]
    avg_value_cer  = sum(vc) / len(vc) if vc else None
    avg_value_found = sum(vf) / len(vf) if vf else None

    summary = {
        "model": "Tesseract (traditional OCR)",
        "n_samples": n,
        "avg_value_cer": avg_value_cer,       # CER เฉพาะช่วงที่ align กับแต่ละ field value (ยุติธรรม)
        "avg_value_found_rate": avg_value_found,  # เทียบเคียง Field Accuracy ของ VLM
        "avg_raw_cer": avg_raw_cer,           # อ้างอิงเสริม: ทั้งหน้า vs ค่า field ล้วนๆ (ไม่ยุติธรรม)
        "avg_raw_wer": avg_raw_wer,
        "note": "Field Accuracy / JSON valid ไม่มีความหมาย -- Tesseract ไม่เข้าใจ schema, "
                "อ่านได้แค่ตัวหนังสือดิบ ไม่แยก field. ใช้ avg_value_cer/avg_value_found_rate "
                "เทียบกับ CER/Field Accuracy ของ VLM แทน raw_cer/raw_wer",
    }

    print(f"\n{'='*50}")
    print(f"📊 Tesseract OCR (ตัวเทียบที่ 3)")
    print(f"{'='*50}")
    print(f"  Value CER เฉลี่ย (ยุติธรรม, เทียบกับ CER ของ VLM ได้):   {avg_value_cer:.1%}")
    print(f"  Value Found Rate (เทียบกับ Field Accuracy ของ VLM ได้): {avg_value_found:.1%}")
    print(f"  ── อ้างอิงเสริม (ไม่ยุติธรรม เพราะเทียบทั้งหน้า vs field ล้วนๆ) ──")
    print(f"  Raw full-text CER: {avg_raw_cer:.1%}   Raw full-text WER: {avg_raw_wer:.1%}")
    print(f"  (Field Accuracy / JSON valid ตามความหมายของ VLM: N/A)")

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"tesseract_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "raw": results}, f, ensure_ascii=False, indent=1)
    print(f"\n💾 บันทึก → {out_path}")


if __name__ == "__main__":
    main()

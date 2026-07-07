"""
eval_typhoon_native.py -- ทดลอง Typhoon OCR ด้วย prompt "ถนัดของมันเอง"
=====================================================================
รอบก่อน (eval_typhoon.py) เราให้ Typhoon ทำงานตาม schema JSON ของเรา ซึ่งเป็น
"สนามเหย้า" ของโมเดลเรา -- รอบนี้ให้ Typhoon ทำงานแบบที่มันถูกเทรนมาจริงๆ:
อ่านเอกสารเป็น markdown แล้วตอบใน {"natural_text": ...} (prompt ตรงจาก
model card ของ typhoon-ocr) พร้อมย่อรูปด้านยาวสุดเหลือ 1800px ตามที่แนะนำ

เนื่องจากผลลัพธ์เป็นข้อความอิสระ (ไม่มี field) จึงวัดด้วยตัวชี้วัดเดียวกับ
Tesseract: value_cer / value_found_rate (หาค่าแต่ละ field ของเฉลยในข้อความ
ที่อ่านได้ แบบไม่สนตำแหน่ง/โครงสร้าง) -- ตอบคำถามว่า "Typhoon อ่านเก่งแค่ไหน
เมื่อถามในภาษาที่มันถนัด" แยกออกจาก "ทำตาม schema ได้แค่ไหน"

ต้องมี env var TYPHOON_API_KEY
รัน: python eval_typhoon_native.py [--model typhoon-ocr]
"""

import os
import re
import sys
import json
import time
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
import schema
from eval_tesseract import eval_value_recall  # ตัวชี้วัดเดียวกับ Tesseract เป๊ะ

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from PIL import Image, ImageOps
from tqdm import tqdm

API_KEY    = os.environ.get("TYPHOON_API_KEY", "")
API_URL    = "https://api.opentyphoon.ai/v1/chat/completions"
TEST_JSONL = "dataset/formatted/test.jsonl"
RAW_DIR    = Path("dataset/raw")
OUT_DIR    = Path("results")
TARGET_LONGEST = 1800   # ตามคำแนะนำใน model card (render target_longest_image_dim=1800)

# prompt "default" ตรงจาก model card typhoon-ocr (anchor text ว่างเพราะเป็นรูปถ่าย ไม่ใช่ PDF)
NATIVE_PROMPT = (
    "Below is an image of a document page along with its dimensions. "
    "Simply return the markdown representation of this document, presenting tables in markdown format as they naturally appear.\n"
    "If the document contains images, use a placeholder like dummy.png for each image.\n"
    "Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
    "RAW_TEXT_START\n\nRAW_TEXT_END"
)


def image_to_b64_1800(path: Path) -> str:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > TARGET_LONGEST:
        scale = TARGET_LONGEST / longest
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_natural_text(content: str) -> str:
    """ดึงค่า natural_text จากคำตอบ (อาจห่อใน ```json fence); ถ้า parse ไม่ได้คืนทั้งก้อน"""
    s = re.sub(r"^```(?:json)?\s*", "", content.strip())
    s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s, strict=False)
        if isinstance(obj, dict) and "natural_text" in obj:
            return str(obj["natural_text"] or "")
    except json.JSONDecodeError:
        pass
    return s


def call_native(model: str, b64_image: str, retries: int = 3) -> str:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": model,
        "max_tokens": 4096,
        # ไม่มี system message และรูปมาก่อนข้อความ -- ตามรูปแบบใน model card เป๊ะ
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": NATIVE_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
            ],
        }],
    }
    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            return extract_natural_text(resp.json()["choices"][0]["message"]["content"])
        except Exception as e:
            print(f"  ⚠️  API error (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(3)
    return ""


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="typhoon-ocr")
    args = parser.parse_args()

    if not API_KEY:
        print("❌ ไม่พบ TYPHOON_API_KEY -- ตั้ง env var ก่อนรัน")
        sys.exit(1)

    test_samples = []
    with open(TEST_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_samples.append(json.loads(line))
    print(f"✓ {len(test_samples)} test samples | model={args.model} | native markdown prompt")

    results = []
    for sample in tqdm(test_samples, desc=f"{args.model}-native"):
        sid = sample["id"]
        gt_fields = schema.parse_model_json(sample["messages"][2]["content"])

        b64 = image_to_b64_1800(RAW_DIR / f"{sid}.jpg")
        pred_text = call_native(args.model, b64)

        value_avg_cer, value_found_rate = eval_value_recall(gt_fields, pred_text)
        results.append({
            "id": sid,
            "pred": pred_text,
            "value_cer": value_avg_cer,
            "value_found_rate": value_found_rate,
        })

    vc = [r["value_cer"] for r in results if r["value_cer"] is not None]
    vf = [r["value_found_rate"] for r in results if r["value_found_rate"] is not None]
    summary = {
        "model": f"{args.model} (native markdown prompt, 1800px)",
        "n_samples": len(results),
        "avg_value_cer": sum(vc) / len(vc),
        "avg_value_found_rate": sum(vf) / len(vf),
        "note": "prompt จาก model card ของ typhoon-ocr (task default); วัดแบบเดียวกับ Tesseract "
                "(value alignment) เพราะผลลัพธ์เป็น markdown อิสระ ไม่มี field",
    }

    print(f"\n{'='*50}")
    print(f"📊 {args.model} -- native prompt (ภาษาถนัดของมันเอง)")
    print(f"{'='*50}")
    print(f"  Value CER เฉลี่ย:  {summary['avg_value_cer']:.1%}")
    print(f"  Value Found Rate: {summary['avg_value_found_rate']:.1%}")

    OUT_DIR.mkdir(exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
    out_path = OUT_DIR / f"typhoon_native_{safe}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "raw": results}, f, ensure_ascii=False, indent=1)
    print(f"\n💾 บันทึก → {out_path}")


if __name__ == "__main__":
    main()

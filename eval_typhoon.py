"""
eval_typhoon.py -- เทียบ Typhoon-OCR-7B ตัวจริง (ผ่าน API, ไม่ได้ fine-tune)
=====================================================================
รันบน 50 ใบ test set เดียวกับ evaluate.py ใช้ prompt เดียวกับที่เราใช้เทรน/
ประเมินโมเดลของเราเอง (จาก schema.py) เพื่อให้เทียบ Field Accuracy/CER กัน
ได้ตรงๆ แบบเดียวกับที่ evaluate.py ทำกับ Baseline/Fine-tuned

ต้องมี env var TYPHOON_API_KEY (ดู https://opentyphoon.ai)
รัน: python eval_typhoon.py
"""

import os
import sys
import json
import time
import base64
import re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
import schema
from evaluate import calc_cer, calc_wer, calc_field_accuracy, calc_items_metrics, cfg

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from tqdm import tqdm

API_KEY    = os.environ.get("TYPHOON_API_KEY", "")
API_URL    = "https://api.opentyphoon.ai/v1/chat/completions"
MODEL      = "typhoon-ocr"   # override ได้ด้วย --model เช่น typhoon-ocr-v1.5
TEST_JSONL = "dataset/formatted/test.jsonl"
OUT_DIR    = Path("results")


def _first_json_object(s: str) -> str:
    start = s.find("{")
    if start == -1:
        return s
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return s[start:]


def call_typhoon(b64_image: str, retries: int = 3) -> str:
    """เรียก Typhoon API ด้วย prompt เดียวกับ schema.py คืน raw text (ไม่ parse)"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": MODEL,
        "max_tokens": 4096,
        "messages": [
            {"role": "system", "content": schema.SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                {"type": "text", "text": schema.USER_PROMPT},
            ]},
        ],
    }
    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=90)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            return _first_json_object(content)
        except Exception as e:
            print(f"  ⚠️  API error (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(3)
            else:
                return ""
    return ""


def main():
    global MODEL
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL,
                        help="ชื่อโมเดลบน API เช่น typhoon-ocr, typhoon-ocr-v1.5")
    args = parser.parse_args()
    MODEL = args.model

    if not API_KEY:
        print("❌ ไม่พบ TYPHOON_API_KEY -- ตั้ง env var ก่อนรัน")
        sys.exit(1)

    test_samples = []
    with open(TEST_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_samples.append(json.loads(line))
    print(f"✓ {len(test_samples)} test samples")

    results = []
    for sample in tqdm(test_samples, desc=MODEL):
        sid       = sample["id"]
        messages  = sample["messages"]
        gt_text   = messages[2]["content"]
        image_b64 = next(c["image"] for c in messages[1]["content"] if c["type"] == "image").split(",")[-1]

        pred_text = call_typhoon(image_b64)

        pred_fields = schema.parse_model_json(pred_text)
        gt_fields   = schema.parse_model_json(gt_text)
        gt_canon    = schema.build_canonical_json(gt_fields)
        pred_canon  = schema.build_canonical_json(pred_fields)

        field_results = {fld["key"]: calc_field_accuracy(pred_fields, gt_fields, fld) for fld in cfg.FIELDS}
        items_metrics = calc_items_metrics(pred_fields, gt_fields)

        results.append({
            "id": sid, "pred": pred_text, "gt": gt_text,
            "cer": calc_cer(pred_canon, gt_canon),
            "wer": calc_wer(pred_canon, gt_canon),
            "emr": 1.0 if pred_canon == gt_canon else 0.0,
            "field_results": field_results,
            "items": items_metrics,
            "json_ok": bool(pred_fields and any(pred_fields.get(k) for k in schema.SCALAR_FIELDS)),
        })

    n = len(results)
    avg_cer = sum(r["cer"] for r in results) / n
    avg_wer = sum(r["wer"] for r in results) / n
    avg_emr = sum(r["emr"] for r in results) / n

    field_acc = {}
    for fld in cfg.FIELDS:
        key = fld["key"]
        matched = [r["field_results"][key]["matched"] for r in results if r["field_results"][key]["matched"] is not None]
        field_acc[key] = sum(matched) / len(matched) if matched else None
    avg_field = sum(v for v in field_acc.values() if v is not None)
    avg_field /= sum(1 for v in field_acc.values() if v is not None) or 1

    items_cer       = sum(r["items"]["cer"] for r in results) / n
    items_count_acc = sum(1 for r in results if r["items"]["count_match"]) / n
    json_valid_rate = sum(1 for r in results if r["json_ok"]) / n

    summary = {
        "model": f"{MODEL} (API, ไม่ได้ fine-tune)",
        "n_samples": n, "avg_cer": avg_cer, "avg_wer": avg_wer, "avg_emr": avg_emr,
        "avg_field": avg_field, "field_acc": field_acc,
        "items_cer": items_cer, "items_count_acc": items_count_acc,
        "json_valid_rate": json_valid_rate,
    }

    print(f"\n{'='*50}")
    print(f"📊 {MODEL} (API, ไม่ได้ fine-tune)")
    print(f"{'='*50}")
    print(f"  CER: {avg_cer:.1%}  WER: {avg_wer:.1%}  Field Acc: {avg_field:.1%}  JSON valid: {json_valid_rate:.1%}")

    OUT_DIR.mkdir(exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", MODEL)
    out_path = OUT_DIR / f"typhoon_results_{safe_model}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "raw": results}, f, ensure_ascii=False, indent=1)
    print(f"\n💾 บันทึก → {out_path}")


if __name__ == "__main__":
    main()

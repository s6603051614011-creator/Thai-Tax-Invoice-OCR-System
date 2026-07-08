"""
Auto-Label Script — สร้าง annotations อัตโนมัติด้วย Typhoon OCR API
=====================================================================
ใช้ Typhoon OCR อ่านใบกำกับภาษีทุกใบแล้วสร้าง annotations.json
จากนั้นเปิด Annotation Tool ตรวจสอบและ Verify เท่านั้น

ติดตั้ง: pip install requests tqdm pillow

รัน:
  python auto_label.py
  python auto_label.py --input dataset/raw/ --output dataset/annotations_auto.json
  python auto_label.py --merge dataset/annotations.json  ← merge กับไฟล์เดิม
"""

import os
import json
import base64
import time
import re
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageOps
from io import BytesIO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # หา schema.py ที่ project root
import schema  # single source of truth ของโครงสร้างเฉลย


# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
API_KEY    = os.environ.get("TYPHOON_API_KEY", "")   # ตั้ง env var TYPHOON_API_KEY
API_URL    = "https://api.opentyphoon.ai/v1/chat/completions"
MODEL      = "typhoon-ocr"

RAW_DIR    = Path("dataset/raw")
OUTPUT     = Path("dataset/annotations_auto.json")

# หน่วงเวลาระหว่างแต่ละ request เพื่อไม่ให้ rate limit
DELAY_SEC  = 1.0

# ขนาดรูปสูงสุดที่ส่ง API (ลดขนาดเพื่อประหยัด token)
MAX_WIDTH  = 1800


# ══════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════
SYSTEM_PROMPT = """คุณคือผู้เชี่ยวชาญด้านการอ่านใบกำกับภาษีไทย
อ่านใบกำกับภาษีในภาพและสกัดข้อมูลออกมาให้ครบถ้วนและถูกต้อง
กรอกตามที่เห็นในใบจริงทุกอย่าง ไม่แปลง format ไม่เดา"""

USER_PROMPT = """อ่านใบกำกับภาษีในรูปนี้ แล้วเติมค่าจริงลงใน JSON ตาม schema นี้ (ตอบเป็น JSON ก้อนเดียวเท่านั้น)

{
  "invoice_number": "",
  "invoice_date": "",
  "seller_name_th": "",
  "seller_name_en": "",
  "seller_tax_id": "",
  "seller_address": "",
  "buyer_name_th": "",
  "buyer_name_en": "",
  "buyer_tax_id": "",
  "buyer_address": "",
  "items": [
    {"description": "", "quantity": "", "unit_price": "", "discount": "", "amount": ""}
  ],
  "subtotal": "",
  "discount": "",
  "vat": "",
  "grand_total": ""
}

กฎการเติมค่า (สำคัญมาก):
1. เติม "เฉพาะค่าที่อ่านได้จากรูปจริง" เท่านั้น — ห้ามคัดลอกข้อความใน schema/คำอธิบายนี้ลงไป
2. ใส่เฉพาะ "ค่า" ห้ามใส่ชื่อหัวข้อ/label เช่น vat ใส่ "819.00" ไม่ใช่ "VAT 7%: 819.00"
3. ช่องที่หาไม่เจอในรูป ให้ใส่ "" (สตริงว่าง) ห้ามเดา ห้ามแต่งข้อมูลขึ้นเอง
4. ชื่อบริษัท: ภาษาไทยลง _th, ภาษาอังกฤษลง _en, ห้ามแปลเอง ถ้ามีภาษาเดียวอีกช่องใส่ ""
5. seller_tax_id / buyer_tax_id = เลข 13 หลักติดกันเท่านั้น (ห้ามใส่ที่อยู่/เบอร์โทร)
6. seller_address / buyer_address = ที่อยู่ตามที่พิมพ์ รวมเป็นข้อความบรรทัดเดียว
7. จำนวนเงิน/ตัวเลข ใส่เฉพาะตัวเลข เช่น 1234.56 (ไม่ต้องมี "บาท" หรือคำอื่น)
8. items: 1 แถวต่อ 1 รายการสินค้า description ใส่ชื่อสินค้าอย่างเดียว ไม่ใส่เลขลำดับ
9. ถ้าไม่มีรายการสินค้าเลย ให้ items เป็น []
10. ห้ามตอบเป็นตาราง markdown ห้ามมีข้อความอื่นนอก JSON"""


# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════

def open_and_resize(img_path: Path, max_width: int = MAX_WIDTH) -> str:
    """เปิดรูป แก้ EXIF orientation และแปลงเป็น base64"""
    img = Image.open(img_path).convert("RGB")
    img = ImageOps.exif_transpose(img)  # แก้รูปกลับหัว/แนวนอน

    # ลดขนาดถ้ากว้างเกิน max_width
    w, h = img.size
    if w > max_width:
        new_h = int(h * max_width / w)
        img = img.resize((max_width, new_h), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _first_json_object(s: str) -> str:
    """
    ดึง JSON object ก้อนแรกที่วงเล็บปีกกาปิดครบ (รู้จัก string/escape)
    แก้กรณีโมเดลตอบ {...} แล้วต่อท้ายด้วยข้อความอื่น -> json.loads ไม่ติด "Extra data"
    """
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
    return s[start:]   # วงเล็บไม่ปิด (ถูกตัดกลางคัน) -> ปล่อยให้ json error แล้ว retry


def call_typhoon_ocr(b64_image: str, retries: int = 3) -> dict:
    """
    เรียก Typhoon OCR API แล้วแปลง response เป็น dict
    retry อัตโนมัติถ้าเกิด error
    """
    import requests

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": MODEL,
        "max_tokens": 4096,   # ใบหลายรายการ 2048 ไม่พอ -> JSON ถูกตัด (Unterminated string)
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                    },
                    {"type": "text", "text": USER_PROMPT}
                ]
            }
        ]
    }

    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            # ลบ markdown code block ถ้ามี
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

            # ดึงเฉพาะ JSON object ก้อนแรกที่วงเล็บปิดครบ (กัน "Extra data" จากข้อความตามท้าย)
            content = _first_json_object(content)

            # strict=False -> ยอมรับ control character (newline ดิบ) ในสตริง
            return json.loads(content, strict=False)

        except json.JSONDecodeError as e:
            print(f"\n  ⚠️  JSON parse error (attempt {attempt+1}): {e}")
            if attempt == retries - 1:
                return {}

        except Exception as e:
            print(f"\n  ⚠️  API error (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(3)
            else:
                return {}

    return {}


def build_entry(img_path: Path, fields: dict, idx: int) -> dict:
    """
    สร้าง annotation entry จาก fields ที่ได้จาก OCR
    - fields            : โครงสร้าง normalize แล้วตาม schema (ให้คนแก้ + ให้ evaluate วัด)
    - ground_truth.json : canonical JSON string = training target ของโมเดล
    """
    stem      = img_path.stem
    norm      = schema.normalize_fields(fields)
    canonical = schema.build_canonical_json(norm)

    return {
        "id":             stem,
        "image_path":     f"dataset/raw/{img_path.name}",
        "source_company": "",
        "annotated_at":   "",
        "verified":       False,   # ← ต้องเปิด review.py มา Verify ทุกใบ
        "auto_labeled":   True,    # ← mark ว่า auto-label ไว้
        "ground_truth":   {"json": canonical},
        "fields":         norm,
    }


def merge_with_existing(auto_entries: list, existing_path: Path) -> list:
    """
    Merge auto-labeled entries กับ annotations เดิม
    ถ้ามีใบไหนใน existing ที่ verified=True แล้ว → เก็บของเดิมไว้
    ถ้ายังไม่ verified → ใช้ auto-labeled แทน
    """
    with open(existing_path, encoding="utf-8") as f:
        existing = json.load(f)

    existing_map = {Path(e["image_path"]).name: e for e in existing}
    auto_map     = {Path(e["image_path"]).name: e for e in auto_entries}

    merged = []
    kept, replaced, added = 0, 0, 0

    for filename, auto_entry in auto_map.items():
        if filename in existing_map:
            ex = existing_map[filename]
            if ex.get("verified"):
                # เก็บของเดิมที่ verified แล้ว
                merged.append(ex)
                kept += 1
            else:
                # ใช้ auto-labeled ใหม่แทน
                merged.append(auto_entry)
                replaced += 1
        else:
            # ใบใหม่ที่ไม่มีใน existing
            merged.append(auto_entry)
            added += 1

    # เพิ่มใบที่อยู่ใน existing แต่ไม่อยู่ใน auto (รูปหายหรือข้ามไป)
    for filename, ex_entry in existing_map.items():
        if filename not in auto_map:
            merged.append(ex_entry)

    # เรียงตาม id
    merged.sort(key=lambda x: x["id"])

    print(f"\n   Merge summary:")
    print(f"   เก็บของเดิม (verified):  {kept} ใบ")
    print(f"   แทนที่ด้วย auto-label:   {replaced} ใบ")
    print(f"   เพิ่มใหม่:               {added} ใบ")

    return merged


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def run(input_dir: Path, output_path: Path, merge_path: Path = None,
        skip_existing: bool = True, exclude_ids=None):

    # ตรวจสอบ API Key
    if API_KEY == "sk-xxxxxxxxxxxxxxxx":
        print("❌ กรุณาใส่ API Key ใน auto_label.py บรรทัดที่ 43")
        print("   API_KEY = 'sk-xxxxxxxxxxxxxxxx'  ← เปลี่ยนตรงนี้")
        return

    # หารูปทั้งหมด — dedup ด้วย set เพราะ Windows filesystem เป็น case-insensitive
    # (glob("*.jpg") กับ glob("*.JPG") คืนไฟล์เดียวกัน -> นับซ้ำ 2 เท่าถ้าไม่ dedup)
    SUPPORTED = {".jpg", ".jpeg", ".png"}
    images = sorted(
        {p for p in input_dir.iterdir()
         if p.is_file() and p.suffix.lower() in SUPPORTED}
    )

    if not images:
        print(f"❌ ไม่พบรูปภาพใน {input_dir}")
        return

    # โหลด existing annotations ทั้งหมด (เก็บไว้ ไม่ทับ) + จำ id ที่มีแล้ว
    existing_by_id = {}
    if skip_existing and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for e in json.load(f):
                existing_by_id[e["id"]] = e
        print(f"📋 มีเฉลยเดิม {len(existing_by_id)} ใบ → เก็บไว้ ไม่ทับ")

    # โหลด test ids ที่ต้องข้าม (Claude Code label เอง)
    exclude = set(exclude_ids or [])
    test_file = Path("dataset/test_ids.txt")
    if test_file.exists():
        exclude |= {l.strip() for l in test_file.read_text(encoding="utf-8").splitlines() if l.strip()}
    if exclude:
        print(f"🚫 ข้าม test set {len(exclude)} ใบ (label เองทาง Claude Code)")

    skip_ids = set(existing_by_id) | exclude

    print(f"📂 พบรูปทั้งหมด: {len(images)} ใบ")
    print(f"🤖 เริ่ม Auto-Label ด้วย Typhoon OCR...\n")

    results   = []
    success   = 0
    failed    = 0
    skipped   = 0

    def save_all():
        # re-read ดิสก์ก่อนเขียน เพื่อไม่ทับ entry ที่ process อื่น (เช่น commit_label) เพิ่มเข้ามา
        on_disk = dict(existing_by_id)
        if output_path.exists():
            try:
                with open(output_path, encoding="utf-8") as f:
                    on_disk = {e["id"]: e for e in json.load(f)}
            except (json.JSONDecodeError, OSError):
                pass  # ไฟล์กำลังถูกเขียน -> ใช้ snapshot เดิมไปก่อน
        # ใส่/อัปเดตเฉพาะผลของรอบนี้ (Typhoon) คง entry อื่นทั้งหมด
        for r in results:
            on_disk[r["id"]] = r
        merged = sorted(on_disk.values(), key=lambda x: x["id"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        return merged

    for img_path in tqdm(images, desc="Auto-labeling"):
        # ข้ามใบที่มีเฉลยแล้ว หรืออยู่ใน test set
        if img_path.stem in skip_ids:
            skipped += 1
            continue

        try:
            b64 = open_and_resize(img_path)
            fields = call_typhoon_ocr(b64)

            if fields:
                results.append(build_entry(img_path, fields, len(results)))
                success += 1
            else:
                empty_entry = build_entry(img_path, {}, len(results))
                empty_entry["auto_labeled"] = False
                results.append(empty_entry)
                failed += 1
                print(f"\n  ❌ อ่านไม่ได้: {img_path.name} → สร้าง empty entry ไว้")

        except Exception as e:
            print(f"\n  ❌ Error {img_path.name}: {e}")
            failed += 1

        # save เป็นระยะกัน progress หาย ถ้าถูกขัดจังหวะกลางคัน
        if results and len(results) % 20 == 0:
            save_all()

        time.sleep(DELAY_SEC)

    final = save_all()

    # สรุป
    auto_done = sum(1 for e in final if e.get("auto_labeled"))
    verified  = sum(1 for e in final if e.get("verified"))

    print(f"""
{'='*55}
✅ Auto-Label เสร็จสิ้น!

📊 สรุป:
   Auto-label สำเร็จ:  {success} ใบ
   อ่านไม่ได้:         {failed} ใบ  (empty entry)
   ข้าม (verified):    {skipped} ใบ

📁 Output: {output_path}
   รวมทั้งหมด:  {len(final)} ใบ
   Auto-labeled: {auto_done} ใบ (verified=False)
   Verified:     {verified} ใบ

➡️  ขั้นตอนต่อไป:
   1. เปิด annotation_tool_v2.html ใน Chrome
   2. โหลดไฟล์ {output_path.name}
   3. ตรวจสอบแต่ละใบ แก้ที่ผิด กด Verify
   4. Export → annotations.json พร้อม augment
{'='*55}""")

    if failed > 0:
        print(f"💡 {failed} ใบที่อ่านไม่ได้ ให้เปิด Annotation Tool กรอกเองครับ")


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label ใบกำกับภาษีด้วย Typhoon OCR")
    parser.add_argument("--input",  default="dataset/raw",
                        help="โฟลเดอร์รูปภาพ (default: dataset/raw)")
    parser.add_argument("--output", default="dataset/annotations_auto.json",
                        help="ไฟล์ output (default: dataset/annotations_auto.json)")
    parser.add_argument("--merge",  default=None,
                        help="merge กับ annotations เดิม (ระบุ path)")
    parser.add_argument("--no-skip", action="store_true",
                        help="อย่าข้ามใบที่ verified แล้ว")
    args = parser.parse_args()

    run(
        input_dir    = Path(args.input),
        output_path  = Path(args.output),
        merge_path   = Path(args.merge) if args.merge else None,
        skip_existing= not args.no_skip,
    )
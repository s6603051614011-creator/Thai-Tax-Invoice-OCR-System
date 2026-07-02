"""
schema.py -- นิยาม Schema กลางของ Ground Truth (ใช้ร่วมกันทุกไฟล์)
=====================================================================
ไฟล์นี้เป็น "แหล่งความจริงเดียว" (single source of truth) ของโครงสร้างเฉลย
auto_label.py / review.py / evaluate.py ต้อง import จากที่นี่ทั้งหมด
เพื่อให้เฉลยทุกใบ consistent -- ซึ่งคือปัจจัยสำคัญที่สุดที่ทำให้
fine-tuned model เก่งกว่า base model

โมเดลถูกเทรนให้ output = canonical JSON string ที่สร้างจาก build_canonical_json()
"""

import json

# ── นิยาม field (เรียงตามลำดับ canonical สำหรับ training target) ──
SCALAR_FIELDS = [
    "invoice_number",   # เลขที่ใบกำกับ
    "invoice_date",     # วันที่ (เก็บตามที่เห็น ไม่แปลง พ.ศ./ค.ศ.)
    "seller_name_th",   # ชื่อผู้ขาย (ไทย)
    "seller_name_en",   # ชื่อผู้ขาย (อังกฤษ)
    "seller_tax_id",    # เลขผู้เสียภาษีผู้ขาย (13 หลัก)
    "seller_address",   # ที่อยู่ผู้ขาย
    "buyer_name_th",    # ชื่อลูกค้า (ไทย)
    "buyer_name_en",    # ชื่อลูกค้า (อังกฤษ)
    "buyer_tax_id",     # เลขผู้เสียภาษีผู้ซื้อ (13 หลัก, ถ้ามี)
    "buyer_address",    # ที่อยู่ผู้ซื้อ
]

ITEM_FIELDS    = ["description", "quantity", "unit_price", "discount", "amount"]
SUMMARY_FIELDS = ["subtotal", "discount", "vat", "grand_total"]

# field ที่เป็นจำนวนเงิน -> ต้อง normalize เป็น "1,234.56"
MONEY_FIELDS_ITEM    = {"unit_price", "discount", "amount"}
MONEY_FIELDS_SUMMARY = {"subtotal", "discount", "vat", "grand_total"}

# ── label ภาษาไทย (สำหรับ UI review + รายงาน evaluate) ──
LABELS = {
    "invoice_number": "เลขที่ใบกำกับ",
    "invoice_date":   "วันที่",
    "seller_name_th": "ชื่อผู้ขาย (ไทย)",
    "seller_name_en": "ชื่อผู้ขาย (อังกฤษ)",
    "seller_tax_id":  "เลขภาษีผู้ขาย",
    "seller_address": "ที่อยู่ผู้ขาย",
    "buyer_name_th":  "ชื่อลูกค้า (ไทย)",
    "buyer_name_en":  "ชื่อลูกค้า (อังกฤษ)",
    "buyer_tax_id":   "เลขภาษีผู้ซื้อ",
    "buyer_address":  "ที่อยู่ผู้ซื้อ",
    "subtotal":       "ยอดก่อน VAT",
    "discount":       "ส่วนลดรวม",
    "vat":            "ภาษี 7%",
    "grand_total":    "ยอดหลัง VAT",
    # item columns
    "description":    "รายการ",
    "quantity":       "จำนวน",
    "unit_price":     "ราคา/หน่วย",
    "amount":         "ราคารวม",
}

# ── การจัดกลุ่ม field สำหรับ UI review (data-driven) ──
UI_GROUPS = [
    {"title": "เอกสาร",        "fields": ["invoice_number", "invoice_date"]},
    {"title": "ผู้ขาย",         "fields": ["seller_name_th", "seller_name_en", "seller_tax_id", "seller_address"]},
    {"title": "ผู้ซื้อ / ลูกค้า", "fields": ["buyer_name_th", "buyer_name_en", "buyer_tax_id", "buyer_address"]},
    {"title": "ยอดเงิน",        "fields": ["subtotal", "discount", "vat", "grand_total"]},
]

# ── field type สำหรับวัด accuracy ใน evaluate.py ──
#   exact   = ต้องตรงทุกตัว (เลขภาษี, เลขที่)
#   fuzzy   = ยอม CER ต่ำ ๆ (ชื่อ)
#   numeric = เทียบเชิงตัวเลข (ตัด comma)
FIELD_TYPES = {
    "invoice_number": "exact",
    "invoice_date":   "fuzzy",
    "seller_name_th": "fuzzy",
    "seller_name_en": "fuzzy",
    "seller_tax_id":  "exact",
    "seller_address": "fuzzy",
    "buyer_name_th":  "fuzzy",
    "buyer_name_en":  "fuzzy",
    "buyer_tax_id":   "exact",
    "buyer_address":  "fuzzy",
    "subtotal":       "numeric",
    "discount":       "numeric",
    "vat":            "numeric",
    "grand_total":    "numeric",
}


# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════

def clean_str(v) -> str:
    """strip + แปลง None เป็น string ว่าง"""
    if v is None:
        return ""
    return str(v).strip()


def fmt_money(v) -> str:
    """
    normalize จำนวนเงินเป็นรูปแบบ canonical: '46,728.97'
    - ว่าง -> '' (สำคัญ: สอนโมเดลให้ตอบว่างเมื่อไม่มี field ไม่ใช่เดา)
    - parse ไม่ได้ -> คืนค่าเดิม (ให้คนแก้ตอน review)
    """
    s = clean_str(v)
    if s == "":
        return ""
    cleaned = s.replace(",", "").replace("บาท", "").replace(" ", "")
    try:
        return f"{float(cleaned):,.2f}"
    except ValueError:
        return s


def empty_record() -> dict:
    """สร้าง record ว่างตาม schema (ใช้เป็น base เวลาเปิดเฉลยเก่า/ไม่ครบ)"""
    rec = {k: "" for k in SCALAR_FIELDS}
    rec["items"] = []
    for k in SUMMARY_FIELDS:
        rec[k] = ""
    return rec


def normalize_fields(fields: dict) -> dict:
    """
    แปลง fields ดิบ (จาก API หรือจากฟอร์ม) ให้เป็น schema มาตรฐานที่ normalize แล้ว
    - merge บน empty_record() เพื่อให้ key ครบเสมอ
    - normalize จำนวนเงินทุกช่อง
    """
    rec = empty_record()
    src = fields or {}

    for k in SCALAR_FIELDS:
        rec[k] = clean_str(src.get(k, ""))

    items = []
    for it in (src.get("items") or []):
        item = {}
        for col in ITEM_FIELDS:
            val = it.get(col, "")
            item[col] = fmt_money(val) if col in MONEY_FIELDS_ITEM else clean_str(val)
        # ข้าม item ที่ว่างเปล่าทั้งแถว
        if any(item[c] for c in ITEM_FIELDS):
            items.append(item)
    rec["items"] = items

    for k in SUMMARY_FIELDS:
        rec[k] = fmt_money(src.get(k, ""))

    return rec


def build_canonical_json(fields: dict, indent=None) -> str:
    """
    สร้าง canonical JSON string = training target ของโมเดล
    key เรียงลำดับตายตัวเสมอ -> เฉลยทุกใบมีโครงสร้างเดียวกันเป๊ะ
    """
    rec = normalize_fields(fields)
    ordered = {}
    for k in SCALAR_FIELDS:
        ordered[k] = rec[k]
    ordered["items"] = [
        {col: it[col] for col in ITEM_FIELDS} for it in rec["items"]
    ]
    for k in SUMMARY_FIELDS:
        ordered[k] = rec[k]
    return json.dumps(ordered, ensure_ascii=False, indent=indent)


# ══════════════════════════════════════════════
# MODEL & PROMPT — single source of truth สำหรับ train/eval/inference
#   *** ทุกไฟล์ต้องใช้ base model + prompt ชุดนี้ตัวเดียวกัน ***
#   train/inference prompt ที่ไม่ตรงกัน = สาเหตุหลักที่ทำให้ผลเพี้ยน
# ══════════════════════════════════════════════
# adapter ใน models/best_model ถูกเทรนบน base นี้ (ดู adapter_config.json)
# 6GB VRAM รับ 7B ไม่ไหว -> ใช้ 3B (typhoon-ocr-7b = Qwen2.5-VL-7B base)
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# prompt เดียวกับที่ auto_label.py ใช้สร้าง JSON target ตอนทำ dataset
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
7. จำนวนเงิน (unit_price/discount/amount/subtotal/discount/vat/grand_total) ใส่เฉพาะตัวเลข เช่น 1234.56 (ไม่ต้องมี "บาท" หรือคำอื่น) — ต้องตรงกับตัวเลขในรูปทุกหลัก ห้ามพิมพ์ทศนิยมเกิน/ขาดจนค่าเปลี่ยน
8. items: 1 แถวต่อ 1 รายการสินค้า description ใส่ชื่อสินค้าอย่างเดียว ไม่ใส่เลขลำดับ
   quantity ไม่ใช่จำนวนเงิน ห้ามใช้กฎข้อ 7 — ถ้าในรูปมีลักษณนามกำกับ (เช่น "1 ลูก", "2 ชุด") ให้ใส่ทั้งตัวเลขและลักษณนามตามที่เห็น ถ้ารูปมีแต่ตัวเลขล้วนก็ใส่แค่ตัวเลข
9. ถ้าไม่มีรายการสินค้าเลย ให้ items เป็น []
10. ห้ามตอบเป็นตาราง markdown ห้ามมีข้อความอื่นนอก JSON"""


def build_inference_messages(pil_image) -> list:
    """สร้าง chat messages สำหรับ inference (ใช้ prompt ชุดเดียวกับตอน train)"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": pil_image},
            {"type": "text",  "text": USER_PROMPT},
        ]},
    ]


def parse_model_json(text: str) -> dict:
    """
    แปลง output ของโมเดล (JSON string) กลับเป็น fields dict
    ทนทานต่อ markdown code block และ JSON เพี้ยนเล็กน้อย
    ใช้ใน evaluate.py
    """
    import re
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # ตัดเอาเฉพาะช่วง { ... } ก้อนแรก ถ้ามีข้อความห่อหุ้ม
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        s = m.group(0)
    try:
        return normalize_fields(json.loads(s))
    except json.JSONDecodeError:
        return empty_record()

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
    normalize จำนวนเงิน: ใส่ comma คั่นหลักพันให้ แต่ "ไม่ปัดทศนิยม"
    เก็บจำนวนตำแหน่งทศนิยมตามที่พิมพ์จริงในรูปเป๊ะ (เช่น '345.7944' -> '345.7944',
    '271000.00' -> '271,000.00', '150.000' -> '150.000') เพราะบางใบพิมพ์ทศนิยม
    ไม่เท่ากับ 2 ตำแหน่งจริง ๆ (unit price คำนวณจากส่วนลดเป็นต้น)
    - ว่าง -> '' (สำคัญ: สอนโมเดลให้ตอบว่างเมื่อไม่มี field ไม่ใช่เดา)
    - parse ไม่ได้ (เช่น '20%') -> คืนค่าเดิม (ให้คนแก้ตอน review)
    """
    s = clean_str(v)
    if s == "":
        return ""
    cleaned = s.replace(",", "").replace("บาท", "").replace(" ", "")
    try:
        float(cleaned)
    except ValueError:
        return s
    neg = cleaned.startswith("-")
    if neg:
        cleaned = cleaned[1:]
    int_part, _, dec_part = cleaned.partition(".")
    int_part = str(int(int_part or "0"))
    result = f"{int(int_part):,}"
    if dec_part:
        result += "." + dec_part
    return ("-" if neg else "") + result


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

# ย่อจากเดิม 763 -> 209 tokens (2026-07-04): fine-tuning สอนพฤติกรรมผ่านตัวอย่างจริง
# ~900 ใบที่ตรวจแล้ว ไม่ใช่ผ่านการอธิบายกฎซ้ำทุก prompt — กฎที่โมเดลเรียนจากรูปแบบ
# คำตอบเองได้ (ห้ามคัดลอก placeholder, ห้ามใส่ label, items ว่าง=[], ห้ามตอบ markdown)
# ถูกตัดออก เหลือเฉพาะกฎที่เดาจากข้อมูลอย่างเดียวไม่ได้ (เช่น "1 ลูก" ต้องมีลักษณนาม)
# เหตุผล: prompt ยาวกิน token จนตัด assistant target ทิ้งเกือบทุกใบ (ดู MAX_SEQ_LEN ใน Finetune.py)
USER_PROMPT = """อ่านใบกำกับภาษีในรูป ตอบเป็น JSON ก้อนเดียวตาม schema นี้เท่านั้น (ห้ามมีข้อความอื่น):
{"invoice_number":"","invoice_date":"","seller_name_th":"","seller_name_en":"","seller_tax_id":"","seller_address":"","buyer_name_th":"","buyer_name_en":"","buyer_tax_id":"","buyer_address":"","items":[{"description":"","quantity":"","unit_price":"","discount":"","amount":""}],"subtotal":"","discount":"","vat":"","grand_total":""}

กฎสำคัญ: หาไม่เจอใส่ "" ห้ามเดา | ชื่อไทยลง _th อังกฤษลง _en | tax_id เลข 13 หลักล้วน | เงินใส่ตัวเลขล้วนตรงทุกหลัก | quantity ถ้ามีลักษณนาม (เช่น "1 ลูก") ใส่ด้วย ไม่ใช่แค่ตัวเลข"""


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

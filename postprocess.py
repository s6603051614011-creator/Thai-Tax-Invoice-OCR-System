"""
postprocess.py -- ตรวจความสมเหตุสมผลของผลลัพธ์ OCR หลัง inference
=====================================================================
ใช้กฎที่ "รู้ล่วงหน้าว่าต้องจริงเสมอ" มาจับคำตอบที่โมเดลอ่านพลาด โดยไม่ต้องมีเฉลย:

1. เลขประจำตัวผู้เสียภาษี 13 หลัก มี check digit ในตัว (mod 11)
   -> เลขที่อ่านผิดแม้หลักเดียวจะไม่ผ่าน checksum เกือบแน่นอน
   (เลข 10 หลักคือ format เก่าก่อน พ.ศ. 2555 — พบจริงในใบปี 2012-2013 ~49 ใบ
    ไม่มี checksum ให้ตรวจ ถือว่า format ถูกต้อง)

2. คณิตศาสตร์ของใบกำกับ: subtotal - discount + vat = grand_total
   และ vat ควรประมาณ 7% ของยอดหลังหักส่วนลด

ผลลัพธ์เป็น "flags" รายใบ — ใบไหนมี flag = ควรให้คนตรวจก่อนใช้ ไม่ใช่แก้อัตโนมัติ
(หลักการเดียวกับกฎ "ห้ามเดา" ตอนทำเฉลย: เราชี้จุดสงสัย ไม่มโนค่าใหม่ให้)

ใช้จาก evaluate.py / api.py:
    from postprocess import validate_fields
    flags = validate_fields(fields)   # -> list[str] (ว่าง = ผ่านทุกข้อ)
"""

import re


# ── เลขประจำตัวผู้เสียภาษี ──

def thai_tax_id_status(tax_id: str) -> str:
    """
    คืนสถานะของเลขผู้เสียภาษี:
      "valid"     = 13 หลัก ผ่าน checksum
      "legacy"    = 10 หลัก (format เก่าก่อน 2555, ไม่มี checksum ให้ตรวจ)
      "checksum"  = 13 หลักแต่ checksum ไม่ผ่าน -> น่าจะอ่านผิดอย่างน้อย 1 หลัก
      "malformed" = ไม่ใช่ตัวเลข 10/13 หลัก
      "empty"     = ว่าง (โมเดลตอบว่าไม่เห็นในรูป — ไม่ใช่ความผิด)
    """
    t = (tax_id or "").strip()
    if not t:
        return "empty"
    digits = re.sub(r"[^0-9]", "", t)
    if len(digits) == 13 and digits == t:
        s = sum(int(digits[i]) * (13 - i) for i in range(12))
        return "valid" if (11 - s % 11) % 10 == int(digits[12]) else "checksum"
    if len(digits) == 10 and digits == t:
        return "legacy"
    return "malformed"


# ── คณิตศาสตร์ของยอดเงิน ──

def _num(v) -> float | None:
    s = str(v or "").replace(",", "").replace(" ", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def money_math_flags(fields: dict, tolerance: float = 0.02) -> list:
    """
    ตรวจ subtotal - discount + vat = grand_total (ยอมคลาด tolerance บาท
    เผื่อการปัดเศษบนใบจริง) และ vat ~ 7% ของยอดหลังหักส่วนลด (ยอม 6.5-7.5%
    เผื่อปัดเศษ) — ตรวจเฉพาะช่องที่มีค่าครบพอคำนวณ ช่องว่างไม่ถือว่าผิด
    """
    flags = []
    subtotal    = _num(fields.get("subtotal"))
    discount    = _num(fields.get("discount")) or 0.0
    vat         = _num(fields.get("vat"))
    grand_total = _num(fields.get("grand_total"))

    if subtotal is not None and vat is not None and grand_total is not None:
        expected = subtotal - discount + vat
        if abs(expected - grand_total) > tolerance:
            flags.append(
                f"เงินไม่ลงตัว: subtotal-discount+vat = {expected:,.2f} "
                f"แต่ grand_total = {grand_total:,.2f}"
            )

    if subtotal is not None and vat is not None and vat > 0:
        base = subtotal - discount
        if base > 0:
            rate = vat / base
            if not (0.065 <= rate <= 0.075):
                flags.append(f"VAT {rate:.1%} ของยอดหลังส่วนลด (ปกติ ~7%)")

    # ผลรวมรายการสินค้า vs subtotal (ยกเว้นใบที่ราคารวม VAT ในตัว: /1.07 ก็นับว่าลงตัว)
    items = fields.get("items") or []
    amounts = [_num(it.get("amount")) for it in items]
    if amounts and all(a is not None for a in amounts) and subtotal is not None:
        items_sum = sum(amounts)
        if abs(items_sum - subtotal) > tolerance and abs(items_sum / 1.07 - subtotal) > tolerance:
            flags.append(
                f"ผลรวมรายการ {items_sum:,.2f} ไม่ตรง subtotal {subtotal:,.2f} "
                f"(และไม่ใช่กรณีราคารวม VAT)"
            )

    return flags


# ── รวมทุกกฎ ──

def validate_fields(fields: dict) -> list:
    """ตรวจทุกกฎ คืน list ของ flag (ว่าง = ไม่พบสิ่งผิดปกติ)"""
    flags = []
    for side in ("seller_tax_id", "buyer_tax_id"):
        status = thai_tax_id_status(fields.get(side, ""))
        if status == "checksum":
            flags.append(f"{side} ไม่ผ่าน checksum: {fields.get(side)} (น่าจะอ่านผิดอย่างน้อย 1 หลัก)")
        elif status == "malformed":
            flags.append(f"{side} ไม่ใช่เลข 13 หลัก (หรือ 10 หลักแบบเก่า): {fields.get(side)!r}")
    flags.extend(money_math_flags(fields))
    return flags


if __name__ == "__main__":
    import sys, json
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # โหมดตรวจเฉลยทั้ง dataset (audit): python postprocess.py --audit
    if "--audit" in sys.argv:
        d = json.load(open("dataset/annotations_auto.json", encoding="utf-8"))
        n_flagged = 0
        for e in d:
            flags = validate_fields(e["fields"])
            if flags:
                n_flagged += 1
                print(f"{e['id']}:")
                for f in flags:
                    print(f"   - {f}")
        print(f"\nรวม {n_flagged}/{len(d)} ใบที่มี flag")

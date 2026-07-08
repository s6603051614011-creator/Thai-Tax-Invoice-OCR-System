"""
audit_labels.py -- สแกนหา label ที่ไม่ consistent ใน annotations
=================================================================
ใช้คู่กับ LABELING_GUIDE.md เพื่อหาใบที่ต้องแก้ให้เป็นมาตรฐานเดียวกัน
ก่อนเอาไปเทรน (เฉลยไม่สม่ำเสมอ = สาเหตุอันดับ 1 ที่ทำให้ model ไม่นิ่ง)

รัน:
  python audit_labels.py
  python audit_labels.py --file dataset/annotations_auto.json
  python audit_labels.py --verified-only   # ตรวจเฉพาะใบที่ verify แล้ว

หมายเหตุ: สคริปต์นี้ "รายงาน" อย่างเดียว ไม่แก้ไฟล์ให้
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Windows CP874 console -> กัน emoji/ไทย print crash
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # หา schema.py ที่ project root
import schema

TEXT_FIELDS = ["seller_name_th", "seller_name_en", "seller_address",
               "buyer_name_th", "buyer_name_en", "buyer_address"]

# คู่คำย่อ/คำเต็ม ที่มักใช้สลับกันในที่อยู่ -> ควรเลือกใช้แบบใดแบบหนึ่งให้ตรงรูปเสมอ
ABBREV_PAIRS = [
    ("ซ.",   "ซอย"),
    ("ถ.",   "ถนน"),
    ("ต.",   "ตำบล"),
    ("อ.",   "อำเภอ"),
    ("จ.",   "จังหวัด"),
    ("แขวง", "ข."),
    ("เขต",  "ข."),
    ("บจก.", "บริษัท"),
    ("หจก.", "ห้างหุ้นส่วนจำกัด"),
]

# ค่าที่ไม่ควรใช้แทน "ว่าง" (ตาม guide ให้ปล่อย "" เท่านั้น)
BAD_EMPTY_MARKERS = {"-", "--", "N/A", "n/a", "na", "ไม่มี", "ไม่ระบุ", "none", "None", "null"}


def iter_entries(data, verified_only):
    for i, e in enumerate(data):
        if verified_only and not e.get("verified"):
            continue
        yield i, e


def audit(data, verified_only):
    issues = defaultdict(list)  # ชนิดปัญหา -> list ของ (idx, id, รายละเอียด)
    abbrev_usage = defaultdict(lambda: {"full": [], "abbr": []})

    for idx, e in iter_entries(data, verified_only):
        eid = e.get("id", f"#{idx}")
        f = e.get("fields", {})

        for fld in TEXT_FIELDS:
            val = str(f.get(fld, "") or "")

            # 1) bad empty marker
            if val.strip() in BAD_EMPTY_MARKERS:
                issues["bad_empty"].append((idx, eid, f"{fld} = '{val}' (ควรปล่อยว่าง)"))

            # 2) เก็บการใช้ย่อ/เต็ม ของแต่ละคู่ (ไว้เทียบความสม่ำเสมอ "ในที่อยู่ส่วนใหญ่")
            for full, abbr in ABBREV_PAIRS:
                if abbr in val:
                    abbrev_usage[(full, abbr)]["abbr"].append((idx, eid, fld))
                elif full in val:
                    abbrev_usage[(full, abbr)]["full"].append((idx, eid, fld))

        # 3) tax id ตรวจรูปแบบ -> แยกความรุนแรง
        #    valid = ตัวเลขล้วน 10 (format เก่า) หรือ 13 หลัก (format ใหม่)
        for tf in ("seller_tax_id", "buyer_tax_id"):
            v = str(f.get(tf, "") or "").strip()
            if not v:
                continue
            digits = "".join(ch for ch in v if ch.isdigit())
            has_letter = any(ch.isalpha() for ch in v)
            if v.isdigit() and len(v) in (10, 13):
                continue  # ผ่าน
            if has_letter or len(digits) > 13:
                # มีตัวอักษร/ยาวผิดปกติ = น่าจะเป็นที่อยู่/ข้อความหลุดมาลงผิดช่อง = แก้ด่วน
                issues["tax_id_wrong"].append((idx, eid, f"{tf} = '{v[:60]}'"))
            else:
                # มีแค่ขีด/เว้นวรรค/หลักไม่ครบ = แค่จัดรูปหรือ OCR เพี้ยนเล็กน้อย
                issues["tax_id_fmt"].append((idx, eid, f"{tf} = '{v}'  (digits={len(digits)})"))

    return issues, abbrev_usage


def main():
    ap = argparse.ArgumentParser(description="Audit label consistency")
    ap.add_argument("--file", default="dataset/annotations_auto.json")
    ap.add_argument("--verified-only", action="store_true")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"[ERROR] ไม่พบไฟล์ {path}")
        return
    data = json.loads(path.read_text(encoding="utf-8"))

    scope = "เฉพาะ verified" if args.verified_only else "ทุกใบ"
    total = sum(1 for _ in iter_entries(data, args.verified_only))
    print(f"\n=== Audit: {path}  ({scope}, {total} ใบ) ===\n")

    issues, abbrev_usage = audit(data, args.verified_only)

    # --- รายงานคำย่อที่ใช้ทั้งสองแบบ (ต้องไปดูทีละใบว่าตรงรูปไหม) ---
    print("── คำย่อ/คำเต็ม ที่ปรากฏทั้งสองแบบในชุดข้อมูล ──")
    print("   (ไม่ใช่ error เสมอไป — แต่ละใบต้องตรงกับรูปของตัวเอง ให้ไปเช็คใบที่เป็น 'ส่วนน้อย')\n")
    flagged = False
    for (full, abbr), u in abbrev_usage.items():
        n_full, n_abbr = len(u["full"]), len(u["abbr"])
        if n_full and n_abbr:
            flagged = True
            minority = u["full"] if n_full <= n_abbr else u["abbr"]
            which = f"'{full}'" if n_full <= n_abbr else f"'{abbr}'"
            print(f"  • {full} / {abbr}:  เต็ม={n_full} ใบ, ย่อ={n_abbr} ใบ")
            print(f"      → ตรวจกลุ่มส่วนน้อย ({which}, {len(minority)} จุด) ว่าตรงรูปจริงไหม:")
            for idx, eid, fld in minority[:15]:
                print(f"          [{idx}] {eid}  ({fld})")
            if len(minority) > 15:
                print(f"          ... และอีก {len(minority)-15} จุด")
    if not flagged:
        print("  ✓ ไม่พบการใช้ย่อ/เต็มปนกัน")

    # --- error ชัดเจน ---
    def report(key, title):
        rows = issues.get(key, [])
        print(f"\n── {title}: {len(rows)} จุด ──")
        for idx, eid, detail in rows[:30]:
            print(f"  [{idx}] {eid}  {detail}")
        if len(rows) > 30:
            print(f"  ... และอีก {len(rows)-30} จุด")

    report("bad_empty",     "ใช้สัญลักษณ์แทนค่าว่าง (ควรปล่อย \"\")")
    report("tax_id_wrong",  "⚠ เลขภาษีมีตัวอักษร/ยาวผิดปกติ — น่าจะที่อยู่/ข้อความหลุดมาผิดช่อง (แก้ด่วน)")
    report("tax_id_fmt",    "เลขภาษีมีขีด/หลักไม่ครบ (จัดรูปให้เป็นตัวเลขล้วน 10 หรือ 13 หลัก)")

    n_wrong = len(issues.get("tax_id_wrong", []))
    n_fmt   = len(issues.get("tax_id_fmt", []))
    n_empty = len(issues.get("bad_empty", []))
    print(f"\n=== สรุป ===")
    print(f"  แก้ด่วน (เนื้อหาผิดช่อง)     : {n_wrong} จุด")
    print(f"  จัดรูปเลขภาษี               : {n_fmt} จุด")
    print(f"  สัญลักษณ์แทนค่าว่าง          : {n_empty} จุด")
    print(f"  คำย่อ/เต็มปนกัน             : ต้องไล่เช็คด้วยตาเทียบรูป (ดูด้านบน)")


if __name__ == "__main__":
    main()

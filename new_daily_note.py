"""
new_daily_note.py -- สร้าง daily note template ของ Thesis-OCR ตามวันที่วันนี้
=============================================================================
สร้างไฟล์ใน  ~/Documents/Thesis-OCR/Daily Notes/<YYYY-MM-DD>.md
หัวข้อ # เป็นวันที่ไทย (พ.ศ.) ของวันนี้อัตโนมัติ

รัน:
  python new_daily_note.py            # สร้างโน้ตของวันนี้ (ไม่ทับถ้ามีอยู่แล้ว)
  python new_daily_note.py --force    # เขียนทับถ้ามีอยู่แล้ว
  python new_daily_note.py --open     # สร้างแล้วเปิดไฟล์เลย
"""

import sys
import argparse
from datetime import date
from pathlib import Path

# กัน emoji/Thai print crash บน console CP874 (Windows ไทย)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# โฟลเดอร์ปลายทาง -- ใช้ home ของผู้ใช้จริง (ไม่ hardcode ชื่อ user)
NOTES_DIR = Path.home() / "Documents" / "Thesis-OCR" / "Daily Notes"

THAI_MONTHS = [
    "", "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
    "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม",
]


def thai_date(d: date) -> str:
    """2026-06-30 -> '30 มิถุนายน 2569' (วัน เดือนไทย ปี พ.ศ.)"""
    return f"{d.day} {THAI_MONTHS[d.month]} {d.year + 543}"


def build_template(d: date) -> str:
    return f"""# {thai_date(d)}
tags: #daily #ocr-thesis

## วันนี้จะทำ
- [ ] debug Finetune.py — loss ไม่ลด
- [ ] ปรับ QLoRA r=8 → r=16
- [ ] รัน augment รอบใหม่

## ผลที่ได้
- CER ก่อน: ___%
- CER หลัง: ___%
- Field ที่ยังผิด: ___

## Claude Code ช่วยอะไร
- แก้ bug ใน ___ บรรทัด ___
- แนะนำให้เพิ่ม ___

## Note สำหรับ Thesis
> ใส่ข้อค้นพบที่จะเอาไปเขียนใน Chapter 3-4

## พรุ่งนี้จะทำ
- [ ] ___
"""


def main():
    ap = argparse.ArgumentParser(description="สร้าง daily note ของ Thesis-OCR")
    ap.add_argument("--force", action="store_true", help="เขียนทับถ้ามีไฟล์อยู่แล้ว")
    ap.add_argument("--open", action="store_true", help="เปิดไฟล์หลังสร้างเสร็จ")
    args = ap.parse_args()

    today = date.today()
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    note_path = NOTES_DIR / f"{today.isoformat()}.md"   # เช่น 2026-06-30.md

    if note_path.exists() and not args.force:
        print(f"[skip] มีโน้ตของวันนี้อยู่แล้ว: {note_path}")
        print("       ใช้ --force ถ้าต้องการเขียนทับ")
    else:
        note_path.write_text(build_template(today), encoding="utf-8")
        action = "เขียนทับ" if (note_path.exists() and args.force) else "สร้าง"
        print(f"[done] {action} daily note: {note_path}")
        print(f"       หัวข้อ: # {thai_date(today)}")

    if args.open:
        import os
        os.startfile(note_path)   # Windows: เปิดด้วยโปรแกรมเริ่มต้น


if __name__ == "__main__":
    main()

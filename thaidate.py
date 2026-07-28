"""
thaidate.py -- แปลง "วันที่" ที่โมเดลอ่านมาจากใบกำกับ ให้เป็นวันที่มาตรฐาน (ค.ศ.)
=====================================================================
ใบกำกับภาษีจริงเขียนวันที่ไม่เหมือนกันเลย จากที่สำรวจในชุดข้อมูล 402 ใบ พบ
อย่างน้อย 9 รูปแบบ:

    11/04/56            dd/mm/yy      ปี พ.ศ. 2 หลัก
    07/01/2013          dd/mm/yyyy    ปี ค.ศ. 4 หลัก
    28.1.2556.          dd.m.yyyy.    ปี พ.ศ. 4 หลัก มีจุดปิดท้าย
    01-Nov-2012         dd-MMM-yyyy   เดือนภาษาอังกฤษ
    16-ก.ย.-56          dd-MMM-yy     เดือนภาษาไทยแบบย่อ
    5 กุมภาพันธ์ 2556    d MMMM yyyy   เดือนภาษาไทยเต็ม เว้นวรรค
    August 21, 2013     MMMM d, yyyy  เดือนขึ้นต้น
    10 สิงหาคม พ.ศ.2559  มีคำว่า พ.ศ. กำกับ
    2015.10.09          yyyy.mm.dd    ปีขึ้นต้น

*** กฎแปลงปี (จุดที่พลาดง่ายที่สุด) ***
ชุดข้อมูลนี้มีทั้งปี พ.ศ. และ ค.ศ. เขียนแบบ 2 หลักปนกัน:
    "11/04/56"  -> 56 คือ พ.ศ. 2556 = ค.ศ. 2013
    "24 Oct 15" -> 15 คือ ค.ศ. 2015
ถ้าเหมาว่า 2 หลัก = พ.ศ. เสมอ ใบปี 15 จะกลายเป็น ค.ศ. 1972 (2515-543)
จึงต้องแยกด้วยช่วงตัวเลข:
    yy >= 50  -> พ.ศ. (2500+yy) แล้วลบ 543   เช่น 56 -> 2013
    yy <  50  -> ค.ศ. (2000+yy)              เช่น 15 -> 2015
    4 หลัก >= 2400 -> พ.ศ. ลบ 543
    มีคำว่า "พ.ศ." กำกับ -> บังคับเป็น พ.ศ. ไม่ต้องเดา

ใช้:
    from thaidate import parse_thai_date
    parse_thai_date("28.1.2556.")   -> (2013, 1, 28)  หรือ None ถ้าอ่านไม่ออก
"""

import re

# เรียงชื่อยาวก่อนสั้น เพื่อให้ match "มีนาคม" ก่อน "มี.ค"
TH_MONTHS = [
    ("มกราคม", 1), ("กุมภาพันธ์", 2), ("มีนาคม", 3), ("เมษายน", 4),
    ("พฤษภาคม", 5), ("มิถุนายน", 6), ("กรกฎาคม", 7), ("สิงหาคม", 8),
    ("กันยายน", 9), ("ตุลาคม", 10), ("พฤศจิกายน", 11), ("ธันวาคม", 12),
    ("ม.ค", 1), ("ก.พ", 2), ("มี.ค", 3), ("เม.ย", 4), ("พ.ค", 5), ("มิ.ย", 6),
    ("ก.ค", 7), ("ส.ค", 8), ("ก.ย", 9), ("ต.ค", 10), ("พ.ย", 11), ("ธ.ค", 12),
]
EN_MONTHS = [
    ("january", 1), ("february", 2), ("march", 3), ("april", 4), ("may", 5), ("june", 6),
    ("july", 7), ("august", 8), ("september", 9), ("october", 10), ("november", 11), ("december", 12),
    ("jan", 1), ("feb", 2), ("mar", 3), ("apr", 4), ("jun", 6), ("jul", 7),
    ("aug", 8), ("sep", 9), ("oct", 10), ("nov", 11), ("dec", 12),
]

DAYS_IN_MONTH = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _year_to_ce(y: int, force_be: bool = False) -> int:
    if force_be:
        return y - 543 if y >= 2400 else (2500 + y - 543 if y < 100 else y - 543)
    if y >= 2400:
        return y - 543          # พ.ศ. 4 หลัก
    if y < 100:
        # 2 หลัก: >=50 คือ พ.ศ. (56 -> 2556), <50 คือ ค.ศ. (15 -> 2015)
        return 2500 + y - 543 if y >= 50 else 2000 + y
    return y                    # ค.ศ. 4 หลักอยู่แล้ว


def _find_month_word(t: str):
    """หาชื่อเดือน (ไทย/อังกฤษ) ในข้อความ -- คืน (เลขเดือน, ข้อความที่ตัดชื่อเดือนออกแล้ว)"""
    low = t.lower()
    for name, num in EN_MONTHS:
        i = low.find(name)
        if i >= 0:
            return num, t[:i] + " " + t[i + len(name):]
    for name, num in TH_MONTHS:
        i = t.find(name)
        if i >= 0:
            return num, t[:i] + " " + t[i + len(name):]
    return None, t


def parse_thai_date(s):
    """คืน (ปี ค.ศ., เดือน, วัน) หรือ None ถ้าอ่านไม่ออก"""
    t = str(s or "").strip()
    if not t:
        return None

    force_be = ("พ.ศ" in t) or ("พศ" in t)
    for marker in ("พ.ศ.", "พ.ศ", "พศ", "ค.ศ.", "ค.ศ", "คศ"):
        t = t.replace(marker, " ")

    month, rest = _find_month_word(t)
    nums = re.findall(r"\d+", rest)

    if month is not None:
        # มีชื่อเดือนแล้ว เหลือแค่ วัน กับ ปี
        if len(nums) < 2:
            return None
        four = [n for n in nums if len(n) == 4]
        if four:
            year = int(four[0])
            day = int(next((n for n in nums if len(n) != 4), 0))
        else:
            day, year = int(nums[0]), int(nums[-1])   # เขียนติดกันแบบ 2 หลักทั้งคู่
        return _check(_year_to_ce(year, force_be), month, day)

    if len(nums) < 3:
        return None
    a, b, c = int(nums[0]), int(nums[1]), int(nums[2])
    if len(nums[0]) == 4:
        return _check(_year_to_ce(a, force_be), b, c)      # yyyy.mm.dd
    return _check(_year_to_ce(c, force_be), b, a)          # dd/mm/yy


def _check(year, month, day):
    if not (1 <= month <= 12) or not (1 <= day <= DAYS_IN_MONTH[month - 1]):
        return None
    if not (1990 <= year <= 2100):
        return None
    return (year, month, day)


if __name__ == "__main__":
    import sys, json
    from pathlib import Path
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    entries = json.loads(Path("dataset/annotations_auto.json").read_text(encoding="utf-8"))
    okc, bad, years = 0, [], {}
    for e in entries:
        raw = e["fields"].get("invoice_date", "")
        got = parse_thai_date(raw)
        if got:
            okc += 1
            years[got[0]] = years.get(got[0], 0) + 1
        elif raw.strip():
            bad.append((e["id"], raw))
    print(f"แปลงได้ {okc}/{len(entries)} ใบ")
    print("แยกตามปี ค.ศ.:", dict(sorted(years.items())))
    if bad:
        print(f"\nอ่านไม่ออก {len(bad)} ใบ:")
        for i, r in bad[:20]:
            print(f"   {i}: {r!r}")

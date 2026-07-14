"""
master_list.py -- ตารางอ้างอิงคู่ค้า (ชื่อบริษัท -> เลขภาษี/ที่อยู่)
=====================================================================
แนวคิด: โมเดลอ่าน "ชื่อบริษัท" ได้แม่น (ตัวใหญ่ อ่านง่าย ~75%) แต่อ่าน
"เลขภาษี/ที่อยู่" ได้แย่กว่า (ตัวเล็ก ~53%) -- และในทางธุรกิจจริง บริษัทซื้อของ
จากซัพพลายเออร์ประจำ (ข้อมูลจริง: ผู้ขาย 78 ราย ครอบคลุม 402 ใบ, 89% ของใบ
มาจากผู้ขายที่ออกซ้ำ)

จึงใช้ชื่อที่อ่านได้ ไป lookup เลขภาษี/ที่อยู่จากฐานข้อมูลคู่ค้าแทนการพึ่ง
การอ่านตัวเลขเล็กๆ -- เป็นวิธีเดียวกับที่ระบบบัญชีจริงทำ

*** กัน data leakage ***
ตารางอ้างอิงสร้างจาก "ใบฝั่ง train เท่านั้น" (splits.json["rest"]) ไม่แตะ test set
-> ถ้าผู้ขายรายนั้นเคยพบใน train เราจะรู้เลขภาษีเขา (ความรู้ธุรกิจที่มีอยู่ก่อน)
-> ถ้าเป็นผู้ขายที่ไม่เคยพบเลย ตารางช่วยไม่ได้ (ตามความจริง)

ใช้:
    from master_list import build_master, apply_master
    master = build_master()              # สร้างจาก train
    fixed  = apply_master(fields, master)  # เติม/แก้ฟิลด์ที่ lookup ได้
"""

import re
import json
import difflib
from pathlib import Path
from collections import defaultdict, Counter

import schema
from postprocess import thai_tax_id_status

ANNOTATIONS = Path("dataset/annotations_auto.json")
SPLITS      = Path("dataset/splits.json")

# ชื่อต้องคล้ายกันแค่ไหนถึงถือว่า "ผู้ขายรายเดียวกัน"
NAME_MATCH_THRESHOLD = 0.86


def norm_name(s: str) -> str:
    """normalize ชื่อบริษัทก่อนเทียบ: ตัดช่องว่าง/คำนำหน้าที่เขียนได้หลายแบบ"""
    s = (s or "").strip()
    if not s:
        return ""
    s = re.sub(r"\s+", "", s)
    # คำนำหน้าที่เขียนได้หลายแบบ -> ยุบให้เหมือนกัน
    s = s.replace("ห้างหุ้นส่วนจำกัด", "หจก.").replace("หจก", "หจก.")
    s = s.replace("บริษัท", "บ.").replace("จำกัด(มหาชน)", "จก.มหาชน").replace("จำกัด", "จก.")
    s = s.replace("..", ".")
    return s


def build_master(side: str = "seller") -> dict:
    """
    สร้างตารางอ้างอิงจาก "ใบฝั่ง train เท่านั้น" (กัน data leakage)
    คืน dict: norm_name -> {"tax_id":..., "address":..., "name_th":..., "name_en":...}
    ถ้าชื่อเดียวกันมีหลายค่า เลือกค่าที่พบบ่อยสุด
    """
    entries = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    splits  = json.loads(SPLITS.read_text(encoding="utf-8"))
    train_ids = set(splits["rest"])          # <-- ไม่รวม test

    buckets = defaultdict(lambda: {"tax_id": Counter(), "address": Counter(),
                                   "name_th": Counter(), "name_en": Counter()})
    for e in entries:
        if e["id"] not in train_ids:
            continue                          # ข้าม test set เด็ดขาด
        f = e["fields"]
        name = f.get(f"{side}_name_th", "")
        key = norm_name(name)
        if not key:
            continue
        b = buckets[key]
        b["name_th"][name] += 1
        for src, dst in ((f"{side}_tax_id", "tax_id"),
                         (f"{side}_address", "address"),
                         (f"{side}_name_en", "name_en")):
            v = (f.get(src) or "").strip()
            if v:
                b[dst][v] += 1

    master = {}
    for key, b in buckets.items():
        master[key] = {
            "tax_id":  b["tax_id"].most_common(1)[0][0]  if b["tax_id"]  else "",
            "address": b["address"].most_common(1)[0][0] if b["address"] else "",
            "name_th": b["name_th"].most_common(1)[0][0] if b["name_th"] else "",
            "name_en": b["name_en"].most_common(1)[0][0] if b["name_en"] else "",
            "n": sum(b["name_th"].values()),
        }
    return master


def lookup(name: str, master: dict, threshold: float = NAME_MATCH_THRESHOLD):
    """หาผู้ขายในตารางจากชื่อที่โมเดลอ่านได้ (fuzzy) -- คืน (entry, score) หรือ (None, 0)"""
    key = norm_name(name)
    if not key or not master:
        return None, 0.0
    if key in master:
        return master[key], 1.0
    best, score = None, 0.0
    for k, v in master.items():
        s = difflib.SequenceMatcher(None, key, k).ratio()
        if s > score:
            best, score = v, s
    return (best, score) if score >= threshold else (None, score)


def apply_master(fields: dict, master_seller: dict, master_buyer: dict = None,
                 threshold: float = NAME_MATCH_THRESHOLD) -> dict:
    """
    เติม/แก้ tax_id + address จากตารางอ้างอิง โดยใช้ชื่อบริษัทที่โมเดลอ่านได้เป็นกุญแจ
    -- แก้เฉพาะเมื่อ match ชื่อได้มั่นใจ (>= threshold)
    -- ไม่แตะช่องอื่น (ชื่อ/เลขที่ใบ/เงิน/รายการ) คงคำตอบโมเดลไว้

    *** นโยบาย "ระมัดระวัง" (เชื่อโมเดลก่อน แก้เฉพาะที่มีเหตุผล) ***
    วัดจริงแล้วพบว่าการเขียนทับดื้อๆ ทำให้แย่ลง เพราะบริษัทเดียวกันมี
      - หลายที่อยู่ (สำนักงานใหญ่/สาขา)
      - เลขภาษีต่างยุค (ใบเก่า 10 หลัก / ใบใหม่ 13 หลัก)
    จึงกำหนดนโยบาย:
      tax_id  : ทับเฉพาะเมื่อโมเดลตอบว่าง หรือ checksum ไม่ผ่าน
                (ถ้า checksum ผ่าน = โมเดลอ่านมาถูก ให้เชื่อโมเดล)
      address : เติมเฉพาะช่องว่าง ไม่เขียนทับของเดิม

    คืน (fields ชุดใหม่, รายการที่แก้)
    """
    out = dict(fields)
    out["items"] = list(fields.get("items") or [])
    changes = []

    for side, master in (("seller", master_seller), ("buyer", master_buyer)):
        if not master:
            continue
        name = out.get(f"{side}_name_th", "")
        entry, score = lookup(name, master, threshold)
        if not entry:
            continue

        # ── tax_id: ทับเฉพาะตอนที่คำตอบโมเดลเชื่อไม่ได้ ──
        key = f"{side}_tax_id"
        new = entry.get("tax_id", "")
        old = (out.get(key) or "").strip()
        if new and new != old:
            status = thai_tax_id_status(old)
            # ว่าง / ผิดรูป / checksum ไม่ผ่าน -> เชื่อตารางอ้างอิง
            # valid หรือ legacy(10 หลัก) -> เชื่อโมเดล (มันอ่านมาได้จริง)
            if status in ("empty", "malformed", "checksum"):
                out[key] = new
                changes.append({"field": key, "old": old, "new": new, "reason": f"tax_id {status}",
                                "matched_name": entry["name_th"], "score": round(score, 3)})

        # ── address: เติมเฉพาะช่องว่าง (บริษัทมีหลายสาขา ห้ามทับ) ──
        key = f"{side}_address"
        new = entry.get("address", "")
        old = (out.get(key) or "").strip()
        if new and not old:
            out[key] = new
            changes.append({"field": key, "old": old, "new": new, "reason": "address ว่าง",
                            "matched_name": entry["name_th"], "score": round(score, 3)})

    return out, changes


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ms = build_master("seller")
    mb = build_master("buyer")
    print(f"ตารางผู้ขาย (จาก train เท่านั้น): {len(ms)} ราย")
    print(f"ตารางผู้ซื้อ (จาก train เท่านั้น): {len(mb)} ราย\n")
    print("ตัวอย่างผู้ขายที่พบบ่อยสุด:")
    for k, v in sorted(ms.items(), key=lambda x: -x[1]["n"])[:5]:
        ok = thai_tax_id_status(v["tax_id"])
        print(f"  {v['n']:3d} ใบ | {v['name_th'][:38]:<38} | {v['tax_id']} ({ok})")

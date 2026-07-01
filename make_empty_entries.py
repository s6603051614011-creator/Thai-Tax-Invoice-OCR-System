"""
make_empty_entries.py -- สร้าง entry เปล่าสำหรับรูปใน raw ที่ยังไม่มีใน annotations
=================================================================================
ใช้ตอนจะ label เองทาง Claude (ไม่พึ่ง Typhoon) -> มี entry ครบ 400 ให้เปิดใน
review.py แล้ว label ผ่าน API ได้ทันที (ทั้งคนและ Claude)

รัน: python make_empty_entries.py
"""
import sys, json, re, glob, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import schema

PATH = "dataset/annotations_auto.json"
RAW  = "dataset/raw"

d = json.load(open(PATH, encoding="utf-8"))
have = {e["id"] for e in d}

raw_ids = sorted(
    int(re.search(r"inv_(\d+)", p).group(1)) for p in glob.glob(f"{RAW}/*.jpg")
)

added = 0
for n in raw_ids:
    iid = f"inv_{n:03d}"
    if iid in have:
        continue
    empty = schema.empty_record()
    d.append({
        "id":           iid,
        "image_path":   f"dataset/raw/{iid}.jpg",
        "source_company": "",
        "annotated_at": "",
        "verified":     False,
        "auto_labeled": False,
        "ground_truth": {"json": schema.build_canonical_json(empty)},
        "fields":       empty,
    })
    added += 1

d.sort(key=lambda x: x["id"])

tmp = PATH + ".tmp"
json.dump(d, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
os.replace(tmp, PATH)
print(f"added {added} empty entries -> total {len(d)}")

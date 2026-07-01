"""
save_via_api.py -- ส่งค่าที่ label แล้วเข้า review.py ที่กำลังรันอยู่ (ไม่ต้องปิด server)
=================================================================================
อ่าน batch file (JSON: list ของ {"id":"inv_XXX", "fields":{...}, "verify":bool})
แล้ว POST ไป /api/save/<idx> ทีละใบ -> server อัปเดตทั้ง memory และไฟล์เอง
จึงไม่ชนกับคนที่กำลัง verify จากฝั่งหน้า

ใช้ "id" (เช่น inv_373) ได้เลย -> สคริปต์ resolve index จากไฟล์ให้เอง
(จะใส่ "idx" มาตรงๆ ก็ได้ ถ้ามี)

รัน:
  python save_via_api.py <batch.json>     # บันทึก batch
  python save_via_api.py --status         # ดูจำนวน verified ปัจจุบัน
"""
import sys, json, argparse
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ANN = Path("dataset/annotations_auto.json")


def id_to_idx():
    d = json.load(open(ANN, encoding="utf-8"))
    return {e["id"]: i for i, e in enumerate(d)}, d


ap = argparse.ArgumentParser()
ap.add_argument("batch", nargs="?", help="batch json file")
ap.add_argument("--port", type=int, default=5000)
ap.add_argument("--status", action="store_true", help="แสดงจำนวน verified แล้วจบ")
ap.add_argument("--next", type=int, metavar="N", help="แสดง N ใบถัดไปที่ยังว่าง (หลัง->หน้า)")
args = ap.parse_args()


def is_empty(e):
    f = e.get("fields", {})
    return not any(str(f.get(k, "")) for k in ("seller_name_th", "buyer_name_th", "invoice_number"))


if args.status:
    _, d = id_to_idx()
    total = len(d)
    ver = sum(1 for e in d if e.get("verified"))
    empty = sum(1 for e in d if is_empty(e))
    print(f"total={total}  verified={ver}  ยังไม่มีข้อมูล(empty)={empty}")
    sys.exit(0)

if args.next:
    _, d = id_to_idx()
    todo = [e for e in reversed(d) if is_empty(e)]   # ว่าง = ยังไม่ได้ label จากรูป
    for e in todo[:args.next]:
        print(f"{e['id']}\t{e['image_path']}")
    if not todo:
        print("(ไม่มีใบว่างเหลือแล้ว)")
    sys.exit(0)

if not args.batch:
    ap.error("ต้องระบุ batch file หรือ --status")

idx_map, _ = id_to_idx()
items = json.load(open(args.batch, encoding="utf-8"))
base = f"http://127.0.0.1:{args.port}/api/save/"

ok = 0
for it in items:
    idx = it.get("idx")
    if idx is None:
        idx = idx_map.get(it["id"])
        if idx is None:
            print(f"  [{it.get('id')}] FAILED: ไม่พบ id ในไฟล์")
            continue
    label = it.get("id", idx)
    body = json.dumps({"fields": it["fields"], "verify": it.get("verify", False)}).encode("utf-8")
    req = urllib.request.Request(base + str(idx), data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            res = json.loads(r.read())
        print(f"  [{label}] idx={idx} saved verify={res.get('verified')}")
        ok += 1
    except Exception as e:
        print(f"  [{label}] FAILED: {e}")

print(f"done: {ok}/{len(items)}")

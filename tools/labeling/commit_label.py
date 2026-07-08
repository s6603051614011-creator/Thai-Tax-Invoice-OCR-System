"""
commit_label.py -- รวมเฉลยที่ Claude Code อ่านเอง เข้า dataset/annotations_auto.json
=====================================================================================
ใช้คู่กับการ label ด้วย Claude Code (อ่านรูปในเครื่อง -> สร้าง JSON ตาม schema)

รับ batch file (JSON) รูปแบบ list:
  [{"id": "inv_001", "fields": {...}}, ...]
แล้ว upsert เข้า annotations_auto.json ผ่าน schema (normalize + canonical JSON)

รัน:
  python commit_label.py <batch.json>
"""

import sys
import json
import argparse
from datetime import date
from pathlib import Path

import schema

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ANN_FILE = Path("dataset/annotations_auto.json")
RAW_DIR  = "dataset/raw"


def load_existing() -> list:
    if ANN_FILE.exists():
        with open(ANN_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def make_entry(inv_id: str, fields: dict, flags=None) -> dict:
    norm = schema.normalize_fields(fields)
    return {
        "id":             inv_id,
        "image_path":     f"{RAW_DIR}/{inv_id}.jpg",
        "source_company": "",
        "annotated_at":   date.today().isoformat(),
        "verified":       False,        # ยังต้องให้คนตรวจใน review.py
        "auto_labeled":   True,
        "labeler":        "claude-code",  # mark ว่ามาจาก Claude Code (ไม่ใช่ Typhoon)
        "flags":          flags or [],    # ช่องที่ผมไม่มั่นใจ -> คนเน้นตรวจตรงนี้
        "ground_truth":   {"json": schema.build_canonical_json(norm)},
        "fields":         norm,
    }


def main(batch_path: Path):
    with open(batch_path, encoding="utf-8") as f:
        batch = json.load(f)

    existing = load_existing()
    by_id = {e["id"]: e for e in existing}

    added, updated = 0, 0
    for item in batch:
        inv_id = item["id"]
        entry  = make_entry(inv_id, item["fields"], item.get("flags"))
        if inv_id in by_id:
            # ไม่เขียนทับใบที่คน verify แล้ว
            if by_id[inv_id].get("verified"):
                continue
            updated += 1
        else:
            added += 1
        by_id[inv_id] = entry

    merged = sorted(by_id.values(), key=lambda e: e["id"])
    ANN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ANN_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    verified = sum(1 for e in merged if e.get("verified"))
    print(f"[commit] batch {len(batch)} ใบ -> +{added} ใหม่ / {updated} อัปเดต")
    print(f"[total]  annotations_auto.json = {len(merged)} ใบ ({verified} verified)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", help="path ของ batch JSON")
    args = ap.parse_args()
    main(Path(args.batch))

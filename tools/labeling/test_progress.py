"""
test_progress.py -- เช็คความคืบหน้าการ label test set (Claude Code)
=====================================================================
แสดงว่า test ids ใบไหน label แล้ว (โดย claude-code) ใบไหนยังเหลือ
ใช้ resume ข้าม session: รันดูว่าเหลือใบไหนแล้วทำต่อ

รัน: python test_progress.py
"""
import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

test_ids = [l.strip() for l in Path("dataset/test_ids.txt").read_text(encoding="utf-8").splitlines() if l.strip()]

done = {}
ann = Path("dataset/annotations_auto.json")
if ann.exists():
    for e in json.load(open(ann, encoding="utf-8")):
        if e.get("labeler") == "claude-code":
            done[e["id"]] = e

done_test = [t for t in test_ids if t in done]
remaining = [t for t in test_ids if t not in done]
flagged   = [t for t in done_test if done.get(t, {}).get("flags")]

print(f"Test set: {len(done_test)}/{len(test_ids)} labeled (claude-code)")
print(f"เหลือ {len(remaining)} ใบ:")
print("  " + ", ".join(remaining) if remaining else "  -- ครบแล้ว --")
print(f"\nใบที่ flag ไว้ให้ตรวจ ({len(flagged)}):")
for t in flagged:
    print(f"  {t}: {', '.join(done[t]['flags'])}")

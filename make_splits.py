"""
make_splits.py -- แบ่ง train/test (+val ภายหลัง) แบบ reproducible
=================================================================
test = 50 ใบ (Claude Code label เอง = gold standard สำหรับวัด CER)
rest = 350 ใบ (Typhoon ร่าง -> แยก train/val ตอนเทรน)

รัน: python make_splits.py
ผลลัพธ์: dataset/splits.json + dataset/test_ids.txt
"""
import sys
import json
import random
import re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW = Path("dataset/raw")
SEED = 42
N_TEST = 50

ids = sorted(p.stem for p in RAW.glob("*.jpg") if re.fullmatch(r"inv_\d+", p.stem))
rng = random.Random(SEED)
shuffled = ids[:]
rng.shuffle(shuffled)

test = sorted(shuffled[:N_TEST])
rest = sorted(shuffled[N_TEST:])

splits = {"seed": SEED, "test": test, "rest": rest}
Path("dataset/splits.json").write_text(
    json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
Path("dataset/test_ids.txt").write_text("\n".join(test) + "\n", encoding="utf-8")

# เช็คว่า 10 ใบที่ label แล้วตกอยู่ split ไหน
done = {f"inv_{i:03d}" for i in range(1, 11)}
test_done = sorted(done & set(test))
print(f"total={len(ids)}  test={len(test)}  rest={len(rest)}")
print(f"test ids (50): {', '.join(test)}")
print(f"\n10 ใบที่ label แล้วอยู่ใน test: {test_done if test_done else 'ไม่มี (ทั้ง 10 เป็น train)'}")

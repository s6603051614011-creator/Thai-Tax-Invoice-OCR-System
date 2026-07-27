"""
eval_combined.py -- รวม 2 วิธีเข้าด้วยกัน: อ่านความละเอียดสูง + Master List
=====================================================================
วิธีที่ 1 (hi-res inference) : ดันช่องตัวเลข (เงิน/วันที่/เลขที่ใบ) ขึ้นแรง
                              แต่ทำให้ "เลขภาษีผู้ซื้อ" ถอยหลัง (train/inference mismatch)
วิธีที่ 2 (master list)      : ดันช่องเลขภาษี/ที่อยู่ จากชื่อบริษัทที่อ่านได้

สองวิธีนี้แก้คนละจุด -> รวมกันน่าจะได้ทั้งคู่ และกลบจุดที่ถอยหลังของวิธีที่ 1

อ่านคำตอบดิบจาก results/hires_preds.json (ที่ eval_hires_infer.py เซฟไว้)
แล้ววัดใหม่ด้วยเกณฑ์เดียวกับ evaluate.py

รัน: python eval_combined.py
"""

import sys
import json
from pathlib import Path

import schema
from evaluate import calc_field_accuracy, calc_cer, cfg
from master_list import build_master, apply_master

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PREDS = Path("results/hires_preds.json")
TEST  = Path("dataset/formatted/test.jsonl")


def load_gt():
    gts = {}
    with open(TEST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            gts[s["id"]] = schema.parse_model_json(s["messages"][2]["content"])
    return gts


def score(preds, gts):
    acc = {}
    for fld in cfg.FIELDS:
        k = fld["key"]
        m = [calc_field_accuracy(preds[i], gts[i], fld)["matched"] for i in preds]
        m = [x for x in m if x is not None]
        acc[k] = sum(m) / len(m) if m else None
    vals = [v for v in acc.values() if v is not None]
    cer = sum(calc_cer(schema.build_canonical_json(preds[i]),
                       schema.build_canonical_json(gts[i])) for i in preds) / len(preds)
    return acc, (sum(vals) / len(vals) if vals else 0.0), cer


def main():
    if not PREDS.exists():
        print(f"ไม่พบ {PREDS} -- รัน eval_hires_infer.py ก่อน")
        return

    gts = load_gt()
    all_preds = json.loads(PREDS.read_text(encoding="utf-8"))
    ms, mb = build_master("seller"), build_master("buyer")
    print(f"ตารางอ้างอิง (จาก train เท่านั้น): ผู้ขาย {len(ms)} ราย, ผู้ซื้อ {len(mb)} ราย\n")

    cols, rows = [], {}
    for res in sorted(all_preds, key=int):
        base = {i: all_preds[res][i] for i in all_preds[res]}
        comb = {}
        n_fix = 0
        for i, p in base.items():
            fixed, ch, _ = apply_master(p, ms, mb)
            comb[i] = fixed
            n_fix += len(ch)

        acc_b, avg_b, cer_b = score(base, gts)
        acc_c, avg_c, cer_c = score(comb, gts)
        cols.append((res, acc_b, avg_b, cer_b, acc_c, avg_c, cer_c, n_fix))

    # ── ตาราง ──
    W = 13
    head = f"{'ช่องข้อมูล':<24}"
    for res, *_ in cols:
        head += f"{res+'tok':>{W}}{res+'+ML':>{W}}"
    print(head)
    print("-" * len(head))
    for fld in cfg.FIELDS:
        k = fld["key"]
        line = f"{schema.LABELS.get(k,k):<24}"
        skip = True
        for res, acc_b, _, _, acc_c, _, _, _ in cols:
            b, c = acc_b.get(k), acc_c.get(k)
            if b is not None:
                skip = False
            line += f"{b:>{W}.1%}" if b is not None else f"{'-':>{W}}"
            line += f"{c:>{W}.1%}" if c is not None else f"{'-':>{W}}"
        if not skip:
            print(line)
    print("-" * len(head))
    line = f"{'Field Accuracy เฉลี่ย':<24}"
    for _, _, avg_b, _, _, avg_c, _, _ in cols:
        line += f"{avg_b:>{W}.1%}{avg_c:>{W}.1%}"
    print(line)
    line = f"{'CER (ยิ่งน้อยยิ่งดี)':<24}"
    for _, _, _, cer_b, _, _, cer_c, _ in cols:
        line += f"{cer_b:>{W}.1%}{cer_c:>{W}.1%}"
    print(line)
    print()
    for res, _, avg_b, _, _, avg_c, _, n_fix in cols:
        print(f"  {res} tok: Master List แก้ {n_fix} ช่อง -> "
              f"Field Acc {avg_b:.1%} -> {avg_c:.1%} ({avg_c-avg_b:+.1%})")

    best = max(cols, key=lambda c: c[5])
    print(f"\nดีที่สุด: {best[0]} tokens + Master List = Field Accuracy {best[5]:.1%} "
          f"| CER {best[6]:.1%}")
    print(f"(จุดเริ่มต้น 320 tok ไม่มี Master List = {cols[0][2]:.1%})")


if __name__ == "__main__":
    main()

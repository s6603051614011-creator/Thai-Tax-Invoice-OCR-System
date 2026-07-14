"""
eval_master_list.py -- วัดผล Master List: Field Accuracy ก่อน/หลัง
=====================================================================
เอาคำตอบดิบของโมเดล (จาก evaluate.py รอบล่าสุด) มาผ่าน master_list.apply_master()
แล้ววัด Field Accuracy ใหม่ด้วยฟังก์ชันเดียวกับ evaluate.py (เกณฑ์เดิมเป๊ะ)

ตารางอ้างอิงสร้างจาก train เท่านั้น -> ไม่มี data leakage

รัน: python eval_master_list.py [--excel results/ocr_evaluation_XXXX.xlsx]
"""

import sys
import argparse
from pathlib import Path

import openpyxl

import schema
from evaluate import calc_field_accuracy, calc_cer, cfg
from master_list import build_master, apply_master

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_EXCEL = "results/ocr_evaluation_20260708_0037.xlsx"   # รอบ Typhoon 1.5 (2B)


def field_acc_table(preds: dict, gts: dict) -> dict:
    """คืน {field_key: accuracy} ด้วยเกณฑ์เดียวกับ evaluate.py"""
    acc = {}
    for fld in cfg.FIELDS:
        k = fld["key"]
        matched = []
        for i in preds:
            m = calc_field_accuracy(preds[i], gts[i], fld)["matched"]
            if m is not None:
                matched.append(m)
        acc[k] = sum(matched) / len(matched) if matched else None
    return acc


def avg(acc: dict) -> float:
    vals = [v for v in acc.values() if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", default=DEFAULT_EXCEL)
    ap.add_argument("--threshold", type=float, default=0.86)
    args = ap.parse_args()

    ws = openpyxl.load_workbook(args.excel)["ข้อความดิบ"]
    rows = [(r[0], r[1], r[3]) for r in ws.iter_rows(min_row=2, values_only=True)]  # id, gt, finetuned
    print(f"โหลดคำตอบโมเดล: {len(rows)} ใบ จาก {args.excel}")

    master_seller = build_master("seller")
    master_buyer  = build_master("buyer")
    print(f"ตารางอ้างอิง (สร้างจาก train เท่านั้น): ผู้ขาย {len(master_seller)} ราย, "
          f"ผู้ซื้อ {len(master_buyer)} ราย\n")

    gts, before, after = {}, {}, {}
    n_changed, n_invoices_touched, all_changes = 0, 0, []

    for id_, gt_txt, pred_txt in rows:
        gt = schema.parse_model_json(gt_txt)
        pr = schema.parse_model_json(pred_txt or "")
        fixed, changes = apply_master(pr, master_seller, master_buyer, args.threshold)
        gts[id_], before[id_], after[id_] = gt, pr, fixed
        if changes:
            n_invoices_touched += 1
            n_changed += len(changes)
            for c in changes:
                c["id"] = id_
                all_changes.append(c)

    acc_b = field_acc_table(before, gts)
    acc_a = field_acc_table(after, gts)

    print(f"แก้ไข {n_changed} ช่อง ใน {n_invoices_touched}/{len(rows)} ใบ "
          f"(threshold ชื่อ = {args.threshold})\n")

    print(f"{'ช่องข้อมูล':<24} {'ก่อน':>8} {'หลัง':>8} {'เปลี่ยน':>9}")
    print("-" * 54)
    for fld in cfg.FIELDS:
        k = fld["key"]
        b, a = acc_b.get(k), acc_a.get(k)
        if b is None or a is None:
            continue
        d = a - b
        mark = "  <=" if abs(d) > 0.001 else ""
        print(f"{schema.LABELS.get(k,k):<24} {b:>7.1%} {a:>7.1%} {d:>+8.1%}{mark}")
    print("-" * 54)
    print(f"{'เฉลี่ยรวม (Field Acc)':<24} {avg(acc_b):>7.1%} {avg(acc_a):>7.1%} "
          f"{avg(acc_a)-avg(acc_b):>+8.1%}")

    # ตรวจสอบคุณภาพการแก้: แก้แล้วถูกขึ้นหรือพังลง
    good = bad = 0
    for c in all_changes:
        gt_val = (gts[c["id"]].get(c["field"]) or "").strip()
        if not gt_val:
            continue
        was_right = c["old"] == gt_val
        now_right = c["new"] == gt_val
        if now_right and not was_right: good += 1
        elif was_right and not now_right: bad += 1
    print(f"\nคุณภาพการแก้: แก้ผิด->ถูก {good} ช่อง | แก้ถูก->ผิด {bad} ช่อง")

    if all_changes:
        print("\nตัวอย่างการแก้ 5 รายการแรก:")
        for c in all_changes[:5]:
            print(f"  {c['id']} {c['field']}")
            print(f"      โมเดลอ่าน: {c['old'][:45]!r}")
            print(f"      ตารางอ้างอิง: {c['new'][:45]!r}  (จับคู่ชื่อ {c['score']})")


if __name__ == "__main__":
    main()

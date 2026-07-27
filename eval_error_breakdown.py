"""
eval_error_breakdown.py -- แยกสาเหตุที่ field ผิด: "เจอแต่อ่านผิด" กับ "ไม่เจอเลย"
=====================================================================
Field Accuracy เพียวๆ บอกแค่ "ถูกกี่ %" ไม่บอกว่าที่ผิดนั้นผิดแบบไหน -- อาจารย์
ขอแยกให้ชัดว่าในส่วนที่ผิด โมเดล "เห็นแล้วอ่านผิด" (ตอบมาแต่ไม่ตรง) กับ
"ไม่เห็นเลย" (ตอบว่างเปล่า) มีสัดส่วนเท่าไหร่ -- สองแบบนี้ต้องแก้คนละทาง:
  เจอแต่อ่านผิด -> ปัญหาความคมชัด/OCR (แก้ด้วย hi-res, preprocessing)
  ไม่เจอเลย     -> ปัญหาความเข้าใจ/ตำแหน่ง (แก้ด้วยข้อมูลเทรนเพิ่ม)

ใช้คำตอบดิบจาก results/hires_preds.json (ที่ eval_hires_infer.py เซฟไว้) วัดที่
ความละเอียดเดียวกับที่ใช้งานจริงใน api.py (960 tokens) แล้วรวม Master List เข้าไป
ด้วย (build_master() จาก train เท่านั้น เหมือน eval_combined.py) ให้ตรงกับ pipeline
จริงที่พนักงานใช้ ไม่ใช่แค่โมเดลอ่านเฉยๆ

รัน: python eval_error_breakdown.py --res 960
"""

import sys
import json
import argparse
from pathlib import Path

import schema
from evaluate import calc_field_accuracy, cfg
from master_list import build_master, apply_master

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PREDS = Path("results/hires_preds.json")
TEST  = Path("dataset/formatted/test.jsonl")


def load_gt() -> dict:
    gts = {}
    with open(TEST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            gts[s["id"]] = schema.parse_model_json(s["messages"][2]["content"])
    return gts


def classify(preds: dict, gts: dict) -> dict:
    """คืน dict: field_key -> {"n":.., "correct":.., "wrong":.., "missing":..}
    n = จำนวนใบที่ GT ไม่ว่าง (ฟิลด์นี้มีคำตอบจริงให้เทียบ)
    correct  = ตอบตรง
    wrong    = ตอบมา แต่ไม่ตรง (เจอ field แต่อ่านผิด)
    missing  = ตอบว่างเปล่า (ไม่เจอ field เลย)
    """
    result = {}
    for fld in cfg.FIELDS:
        k = fld["key"]
        n = correct = wrong = missing = 0
        for i in preds:
            r = calc_field_accuracy(preds[i], gts[i], fld)
            if r["matched"] is None:
                continue  # GT ว่าง -- ข้าม ไม่นับเป็นตัวส่วน
            n += 1
            if r["matched"]:
                correct += 1
            elif not r["pred"]:
                missing += 1
            else:
                wrong += 1
        if n:
            result[k] = {"n": n, "correct": correct, "wrong": wrong, "missing": missing}
    return result


def print_table(title: str, br: dict) -> dict:
    print(f"\n=== {title} ===")
    W = 10
    print(f"{'ช่องข้อมูล':<24}{'จำนวนช่อง':>{W}}{'ถูก':>{W}}{'%ถูก':>{W}}"
         f"{'เจอแต่ผิด':>{W}}{'%ผิด':>{W}}{'ไม่เจอ':>{W}}{'%ไม่เจอ':>{W}}")
    print("-" * (24 + W * 7))
    tot = {"n": 0, "correct": 0, "wrong": 0, "missing": 0}
    for fld in cfg.FIELDS:
        k = fld["key"]
        if k not in br:
            continue
        c = br[k]
        for key in tot:
            tot[key] += c[key]
        print(f"{schema.LABELS.get(k,k):<24}{c['n']:>{W}}{c['correct']:>{W}}"
             f"{c['correct']/c['n']:>{W}.1%}{c['wrong']:>{W}}{c['wrong']/c['n']:>{W}.1%}"
             f"{c['missing']:>{W}}{c['missing']/c['n']:>{W}.1%}")
    print("-" * (24 + W * 7))
    n = tot["n"]
    print(f"{'รวม (ถ่วงตามจำนวนช่อง)':<24}{n:>{W}}{tot['correct']:>{W}}{tot['correct']/n:>{W}.1%}"
         f"{tot['wrong']:>{W}}{tot['wrong']/n:>{W}.1%}{tot['missing']:>{W}}{tot['missing']/n:>{W}.1%}")

    # ── ค่าเฉลี่ยแบบถ่วงน้ำหนักเท่ากันทุกฟิลด์ (macro-average) ──
    # นี่คือวิธีเดียวกับที่ evaluate.py ใช้รายงาน "Field Accuracy" ตัวหลัก (เช่น 64.3%)
    # ต่างจากแถวด้านบนซึ่งถ่วงตามจำนวนตัวอย่างของแต่ละฟิลด์ (ฟิลด์ที่มีน้อยใบ เช่น
    # "ชื่อลูกค้า (อังกฤษ)" n=2 มีน้ำหนักเท่าฟิลด์ที่มี n=50 ในสูตรนี้) -- สองค่าจะไม่
    # เท่ากัน เจตนา ไม่ใช่ error ให้ใช้แถวนี้เทียบกับตัวเลขที่เคยรายงานอาจารย์
    fields = [br[k] for fld in cfg.FIELDS if (k := fld["key"]) in br]
    macro_correct = sum(c["correct"] / c["n"] for c in fields) / len(fields)
    macro_wrong   = sum(c["wrong"]   / c["n"] for c in fields) / len(fields)
    macro_missing = sum(c["missing"] / c["n"] for c in fields) / len(fields)
    print(f"{'เฉลี่ยต่อฟิลด์ (=Field Acc)':<24}{'':>{W}}{'':>{W}}{macro_correct:>{W}.1%}"
         f"{'':>{W}}{macro_wrong:>{W}.1%}{'':>{W}}{macro_missing:>{W}.1%}")
    tot["macro_correct"], tot["macro_wrong"], tot["macro_missing"] = macro_correct, macro_wrong, macro_missing
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=960,
                    help="ความละเอียด (tokens) ที่จะดึงคำตอบดิบมาวิเคราะห์ (ต้องมีอยู่แล้วใน hires_preds.json)")
    args = ap.parse_args()
    res = str(args.res)

    if not PREDS.exists():
        print(f"ไม่พบ {PREDS} -- รัน eval_hires_infer.py --res {res} ก่อน")
        return
    all_preds = json.loads(PREDS.read_text(encoding="utf-8"))
    if res not in all_preds:
        print(f"ไม่มีคำตอบดิบที่ {res} tokens ใน {PREDS} (มีแค่: {list(all_preds.keys())})")
        print(f"-> รัน: python eval_hires_infer.py --res {res}")
        return

    gts = load_gt()
    preds_raw = all_preds[res]

    # ── ก่อน Master List (โมเดลอ่านล้วนๆ) ──
    br_raw = classify(preds_raw, gts)
    tot_raw = print_table(f"{res} tokens -- โมเดลอ่านล้วนๆ (ยังไม่รวม Master List)", br_raw)

    # ── หลัง Master List (ตรงกับที่ api.py ใช้งานจริง) ──
    ms, mb = build_master("seller"), build_master("buyer")
    preds_ml = {i: apply_master(p, ms, mb)[0] for i, p in preds_raw.items()}
    br_ml = classify(preds_ml, gts)
    tot_ml = print_table(f"{res} tokens + Master List -- ตรงกับที่ใช้งานจริงใน api.py", br_ml)

    print(f"\nสรุป (Field Accuracy เฉลี่ยต่อฟิลด์ -- ตัวที่ใช้รายงาน): "
         f"{res} tok เพียวๆ ถูก {tot_raw['macro_correct']:.1%} -> "
         f"หลังรวม Master List ถูก {tot_ml['macro_correct']:.1%} "
         f"({tot_ml['macro_correct'] - tot_raw['macro_correct']:+.1%})")

    out = {"resolution": res,
          "before_master_list": {"per_field": br_raw, "total": tot_raw},
          "after_master_list": {"per_field": br_ml, "total": tot_ml}}
    out_path = Path(f"results/error_breakdown_{res}.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nบันทึก -> {out_path}")


if __name__ == "__main__":
    main()

"""
eval_error_breakdown.py -- แยกสาเหตุที่ผิด: JSON พัง / เจอแต่อ่านผิด / ไม่เจอเลย
=====================================================================
Field Accuracy เพียวๆ บอกแค่ "ถูกกี่ %" ไม่บอกว่าที่ผิดนั้นผิดแบบไหน -- อาจารย์
ขอแยกให้ชัด เพราะแต่ละแบบต้องแก้คนละทาง:

  1. JSON พังทั้งใบ  -> โมเดลอ่านได้แต่ "พูดออกมาผิดรูปแบบ" จน parse ไม่ผ่าน
                        แก้ด้วย constrained decoding / retry / เทรนให้ตอบ JSON นิ่งขึ้น
  2. เจอแต่อ่านผิด   -> โมเดลหา field เจอ ตอบมา แต่อ่านตัวอักษรผิด
                        แก้ด้วยความละเอียดภาพ / preprocessing / คุณภาพรูปถ่าย
  3. ไม่เจอเลย       -> โมเดลตอบว่าง ไม่รู้ว่าต้องมองหาอะไร/ตรงไหน
                        แก้ด้วยข้อมูลเทรนเพิ่ม

*** ทำไมต้องแยกหมวด 1 ออกมา (สำคัญ) ***
schema.parse_model_json() คืน empty_record() เงียบๆ เมื่อ parse JSON ไม่ผ่าน
-> ทุก field ของใบนั้นกลายเป็นค่าว่างหมด -> ถูกนับปนเป็น "ไม่เจอเลย" ทั้งใบ
วัดจริงที่ 960 tok พบว่า 23 จาก 29 ช่องที่ "ไม่เจอ" (79%) มาจากใบแค่ 2 ใบที่ JSON
พัง ไม่ใช่โมเดลหา field ไม่เจอ -- ถ้าไม่แยกออกมา จะสรุปสาเหตุผิดและแก้ผิดทาง

วิธีตรวจ: ใบที่ SCALAR_FIELDS ว่างหมดทุกช่อง = ถือว่า JSON พัง ใช้เกณฑ์เดียวกับ
api.py (ตัวแปร json_ok ใน route /ocr) ให้สอดคล้องกับที่ production ใช้จริง
ข้อจำกัด: hires_preds.json เก็บผลที่ parse แล้ว ไม่ได้เก็บ raw output จึงตรวจ
ย้อนหลังได้แค่ด้วย heuristic นี้ (ใบที่โมเดลตอบว่างจริงทุกช่องจะถูกนับปนมาด้วย
แต่ในทางปฏิบัติแทบเป็นไปไม่ได้ที่ใบจริงจะว่างครบทั้ง 10 ช่อง)

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


def find_broken_json(preds: dict) -> list:
    """คืน id ของใบที่ถือว่า JSON พัง -- เกณฑ์เดียวกับ api.py::ocr (json_ok)
    คือ SCALAR_FIELDS ว่างหมดทุกช่อง"""
    return sorted(
        i for i, p in preds.items()
        if not any(p.get(k) for k in schema.SCALAR_FIELDS)
    )


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


def macro(br: dict) -> dict:
    """ค่าเฉลี่ยแบบถ่วงน้ำหนักเท่ากันทุกฟิลด์ (macro-average)
    -- วิธีเดียวกับที่ evaluate.py ใช้รายงาน "Field Accuracy" ตัวหลัก"""
    fields = [br[k] for fld in cfg.FIELDS if (k := fld["key"]) in br]
    return {
        "correct": sum(c["correct"] / c["n"] for c in fields) / len(fields),
        "wrong":   sum(c["wrong"]   / c["n"] for c in fields) / len(fields),
        "missing": sum(c["missing"] / c["n"] for c in fields) / len(fields),
    }


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

    # ต่างจากแถวด้านบนซึ่งถ่วงตามจำนวนตัวอย่างของแต่ละฟิลด์ (ฟิลด์ที่มีน้อยใบ เช่น
    # "ชื่อลูกค้า (อังกฤษ)" n=2 มีน้ำหนักเท่าฟิลด์ที่มี n=50 ในสูตรนี้) -- สองค่าจะไม่
    # เท่ากัน เจตนา ไม่ใช่ error ให้ใช้แถวนี้เทียบกับตัวเลขที่เคยรายงานอาจารย์
    m = macro(br)
    print(f"{'เฉลี่ยต่อฟิลด์ (=Field Acc)':<24}{'':>{W}}{'':>{W}}{m['correct']:>{W}.1%}"
         f"{'':>{W}}{m['wrong']:>{W}.1%}{'':>{W}}{m['missing']:>{W}.1%}")
    tot["macro"] = m
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
    n_total = len(preds_raw)

    # ── หมวด 1: JSON พังทั้งใบ (แยกออกก่อน ไม่งั้นไปโผล่ปนใน "ไม่เจอเลย") ──
    broken = find_broken_json(preds_raw)
    ok_ids = [i for i in preds_raw if i not in set(broken)]

    print(f"\n{'='*70}")
    print(f"หมวดที่ 1 -- JSON พังทั้งใบ (parse ไม่ผ่าน ทุกช่องเลยว่างหมด)")
    print(f"{'='*70}")
    print(f"  {len(broken)}/{n_total} ใบ ({len(broken)/n_total:.1%})"
         + (f" -> {', '.join(broken)}" if broken else ""))
    print("  สาเหตุคนละแบบกับ 'ไม่เจอ field' -- โมเดลอาจอ่านได้หมดแล้ว แต่ตอบผิดรูปแบบ")

    ms, mb = build_master("seller"), build_master("buyer")

    def with_ml(subset_ids):
        sub = {i: preds_raw[i] for i in subset_ids}
        return {i: apply_master(p, ms, mb)[0] for i, p in sub.items()}

    # ── ตารางหลัก: เฉพาะใบที่ JSON ใช้ได้ (สัญญาณจริงของหมวด 2 กับ 3) ──
    br_ok_raw = classify({i: preds_raw[i] for i in ok_ids}, gts)
    print_table(f"หมวด 2-3 -- {res} tok โมเดลอ่านล้วนๆ "
               f"(เฉพาะ {len(ok_ids)} ใบที่ JSON ใช้ได้)", br_ok_raw)

    br_ok_ml = classify(with_ml(ok_ids), gts)
    tot_ok_ml = print_table(f"หมวด 2-3 -- {res} tok + Master List = ที่ใช้งานจริงใน api.py "
                           f"(เฉพาะ {len(ok_ids)} ใบที่ JSON ใช้ได้)", br_ok_ml)

    # ── ตารางเทียบ: รวมใบ JSON พังด้วย (= ตัวเลขชุดที่เคยรายงานไปก่อนแก้) ──
    br_all_ml = classify(with_ml(list(preds_raw)), gts)
    tot_all_ml = print_table(f"[เทียบ] {res} tok + Master List รวมใบ JSON พังด้วย "
                            f"(ทั้ง {n_total} ใบ -- ตัวเลขชุดเดิมก่อนแยกหมวด 1)", br_all_ml)

    # ── สรุป ──
    mo, ma = tot_ok_ml["macro"], tot_all_ml["macro"]
    print(f"\n{'='*70}")
    print("สรุปสำหรับรายงาน (ที่ใช้งานจริง: {} tok + Master List)".format(res))
    print(f"{'='*70}")
    print(f"  หมวด 1 JSON พังทั้งใบ : {len(broken)}/{n_total} ใบ ({len(broken)/n_total:.1%})")
    print(f"  จากใบที่เหลือ {len(ok_ids)} ใบ:")
    print(f"    ถูก              : {mo['correct']:.1%}")
    print(f"    หมวด 2 อ่านผิด   : {mo['wrong']:.1%}  (เจอ field แต่อ่านตัวอักษรผิด)")
    print(f"    หมวด 3 ไม่เจอเลย : {mo['missing']:.1%}  (ตอบว่าง)")
    print()
    print(f"  เทียบกับถ้านับรวมใบ JSON พัง: ถูก {ma['correct']:.1%} | "
         f"อ่านผิด {ma['wrong']:.1%} | ไม่เจอ {ma['missing']:.1%}")
    print(f"  -> ใบ JSON พังทำให้ '%ไม่เจอ' ดูสูงเกินจริง "
         f"{ma['missing'] - mo['missing']:+.1%} (นับดิบ {tot_all_ml['missing'] - tot_ok_ml['missing']} ช่อง)")

    out = {
        "resolution": res,
        "n_total": n_total,
        "json_broken": {"ids": broken, "count": len(broken), "rate": len(broken) / n_total},
        "valid_only": {  # ตัวเลขที่ควรใช้รายงาน
            "n_invoices": len(ok_ids),
            "before_master_list": {"per_field": br_ok_raw, "macro": macro(br_ok_raw)},
            "after_master_list":  {"per_field": br_ok_ml,  "macro": macro(br_ok_ml)},
        },
        "including_broken": {  # ชุดเดิม เก็บไว้เทียบ
            "n_invoices": n_total,
            "after_master_list": {"per_field": br_all_ml, "macro": macro(br_all_ml)},
        },
    }
    out_path = Path(f"results/error_breakdown_{res}.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nบันทึก -> {out_path}")


if __name__ == "__main__":
    main()

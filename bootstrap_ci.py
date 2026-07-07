"""
bootstrap_ci.py -- ช่วงความเชื่อมั่น (95% CI) ของผลเปรียบเทียบโมเดล ด้วย bootstrap
=====================================================================
n=50 ใบ เล็กพอที่ตัวเลขจุดเดียวอาจหลอกได้ -- สคริปต์นี้ resample ใบ (ซ้ำได้)
B=10,000 รอบ แล้วคำนวณ metric ใหม่ทุกรอบ เพื่อดูว่าค่าจริงแกว่งได้แค่ไหน

ใช้ paired bootstrap: ทุกโมเดลใช้ชุด index เดียวกันในแต่ละรอบ (เพราะสอบข้อสอบ
ชุดเดียวกัน) ทำให้ CI ของ "ผลต่าง" แคบและถูกต้องกว่าการ resample แยกกัน

แหล่งข้อมูล (ผลดิบรายใบที่รันไว้แล้ว ไม่ต้องรัน inference ใหม่):
- Baseline / Fine-tuned: สร้างใหม่จากชีท "ข้อความดิบ" ใน Excel (parse ด้วย
  schema.parse_model_json + calc_field_accuracy ชุดเดียวกับ evaluate.py แล้ว
  ตรวจว่าค่า point estimate ตรงกับที่เคยรายงานก่อนเชื่อผล)
- Typhoon (schema mode): results/typhoon_results_*.json (มี field_results รายใบ)
- Tesseract / Typhoon native: value_found_rate รายใบ

รัน: python bootstrap_ci.py
"""

import sys
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import schema
from evaluate import calc_cer, calc_field_accuracy, cfg

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import openpyxl

EXCEL_PATH    = "results/ocr_evaluation_20260705_2249.xlsx"
TYPHOON_7B    = "results/typhoon_results_20260706_2107.json"
TYPHOON_V15   = "results/typhoon_results_typhoon-ocr-v1.5_20260706_2141.json"
TESSERACT     = "results/tesseract_results_20260706_0859.json"
B             = 10_000
SEED          = 42

FIELD_KEYS = [f["key"] for f in cfg.FIELDS]


def per_invoice_from_texts(gt_text: str, pred_text: str) -> dict:
    """คำนวณ metric รายใบจากข้อความดิบ ด้วยฟังก์ชันชุดเดียวกับ evaluate.py"""
    gt_fields   = schema.parse_model_json(gt_text)
    pred_fields = schema.parse_model_json(pred_text)
    gt_canon    = schema.build_canonical_json(gt_fields)
    pred_canon  = schema.build_canonical_json(pred_fields)
    return {
        "cer": calc_cer(pred_canon, gt_canon),
        "matched": {f["key"]: calc_field_accuracy(pred_fields, gt_fields, f)["matched"]
                    for f in cfg.FIELDS},
        "json_ok": bool(pred_fields and any(pred_fields.get(k) for k in schema.SCALAR_FIELDS)),
    }


def load_vlm_local():
    """Baseline + Fine-tuned จากชีทข้อความดิบ"""
    wb = openpyxl.load_workbook(EXCEL_PATH)
    ws = wb["ข้อความดิบ"]
    base, ft, ids = [], [], []
    for row in ws.iter_rows(min_row=2, values_only=True):
        id_, gt, b, f = row
        ids.append(id_)
        base.append(per_invoice_from_texts(gt, b or ""))
        ft.append(per_invoice_from_texts(gt, f or ""))
    return ids, base, ft


def load_typhoon(path):
    d = json.load(open(path, encoding="utf-8"))
    out = {}
    for r in d["raw"]:
        out[r["id"]] = {
            "cer": r["cer"],
            "matched": {k: r["field_results"][k]["matched"] for k in FIELD_KEYS},
            "json_ok": r["json_ok"],
        }
    return out


def avg_field_macro(rows) -> float:
    """macro ต่อ field แบบเดียวกับ evaluate.summarize()"""
    accs = []
    for k in FIELD_KEYS:
        vals = [r["matched"][k] for r in rows if r["matched"][k] is not None]
        if vals:
            accs.append(sum(vals) / len(vals))
    return sum(accs) / len(accs) if accs else 0.0


def pct_ci(samples, lo=2.5, hi=97.5):
    s = sorted(samples)
    n = len(s)
    return s[int(n * lo / 100)], s[int(n * hi / 100)]


def main():
    print("โหลด + ตรวจสอบข้อมูลรายใบ...")
    ids, base_rows, ft_rows = load_vlm_local()
    ty7  = load_typhoon(TYPHOON_7B)
    ty15 = load_typhoon(TYPHOON_V15)
    assert set(ids) == set(ty7) == set(ty15), "test ids ไม่ตรงกันระหว่างไฟล์ผล"
    ty7_rows  = [ty7[i] for i in ids]
    ty15_rows = [ty15[i] for i in ids]

    tess = json.load(open(TESSERACT, encoding="utf-8"))
    tess_found = {r["id"]: r["value_found_rate"] for r in tess["raw"]}
    tess_rows = [tess_found[i] for i in ids]

    # ตรวจ point estimate ให้ตรงกับที่เคยรายงานก่อนเชื่อ CI
    models = {
        "Baseline":         base_rows,
        "Fine-tuned":       ft_rows,
        "typhoon-ocr":      ty7_rows,
        "typhoon-ocr-v1.5": ty15_rows,
    }
    print(f"\n{'Model':<18} {'FieldAcc':>9} {'CER':>7} {'JSONok':>7}   (point estimate เต็มชุด)")
    for name, rows in models.items():
        print(f"{name:<18} {avg_field_macro(rows):>8.1%} "
              f"{sum(r['cer'] for r in rows)/len(rows):>6.1%} "
              f"{sum(r['json_ok'] for r in rows)/len(rows):>6.1%}")
    print(f"{'Tesseract(found)':<18} {sum(tess_rows)/len(tess_rows):>8.1%}")

    # ── paired bootstrap ──
    print(f"\nbootstrap B={B:,} (paired resample ที่ระดับใบ)...")
    rnd = random.Random(SEED)
    n = len(ids)
    stats = {name: {"field": [], "cer": [], "json": []} for name in models}
    stats["Tesseract"] = {"found": []}
    diffs = {"FT-Baseline": [], "FT-typhoon-ocr": [], "FT-typhoon-v1.5": []}

    for _ in range(B):
        idx = [rnd.randrange(n) for _ in range(n)]
        vals = {}
        for name, rows in models.items():
            sub = [rows[i] for i in idx]
            fa = avg_field_macro(sub)
            stats[name]["field"].append(fa)
            stats[name]["cer"].append(sum(r["cer"] for r in sub) / n)
            stats[name]["json"].append(sum(r["json_ok"] for r in sub) / n)
            vals[name] = fa
        stats["Tesseract"]["found"].append(sum(tess_rows[i] for i in idx) / n)
        diffs["FT-Baseline"].append(vals["Fine-tuned"] - vals["Baseline"])
        diffs["FT-typhoon-ocr"].append(vals["Fine-tuned"] - vals["typhoon-ocr"])
        diffs["FT-typhoon-v1.5"].append(vals["Fine-tuned"] - vals["typhoon-ocr-v1.5"])

    print(f"\n== 95% CI: Field Accuracy ==")
    for name in models:
        lo, hi = pct_ci(stats[name]["field"])
        print(f"  {name:<18} [{lo:.1%} , {hi:.1%}]")
    lo, hi = pct_ci(stats["Tesseract"]["found"])
    print(f"  {'Tesseract(found)':<18} [{lo:.1%} , {hi:.1%}]")

    print(f"\n== 95% CI: CER ==")
    for name in models:
        lo, hi = pct_ci(stats[name]["cer"])
        print(f"  {name:<18} [{lo:.1%} , {hi:.1%}]")

    print(f"\n== 95% CI: JSON valid ==")
    for name in models:
        lo, hi = pct_ci(stats[name]["json"])
        print(f"  {name:<18} [{lo:.1%} , {hi:.1%}]")

    print(f"\n== 95% CI: ผลต่าง Field Accuracy (paired) ==")
    for name, d in diffs.items():
        lo, hi = pct_ci(d)
        sig = "✅ มีนัยสำคัญ (CI ไม่คร่อม 0)" if lo > 0 or hi < 0 else "⚠️ CI คร่อม 0"
        print(f"  {name:<18} [{lo:+.1%} , {hi:+.1%}]  {sig}")

    out = {
        "B": B, "n": n, "seed": SEED,
        "field_acc_ci": {name: pct_ci(stats[name]["field"]) for name in models},
        "cer_ci": {name: pct_ci(stats[name]["cer"]) for name in models},
        "json_ci": {name: pct_ci(stats[name]["json"]) for name in models},
        "tesseract_found_ci": pct_ci(stats["Tesseract"]["found"]),
        "diff_field_ci": {name: pct_ci(d) for name, d in diffs.items()},
    }
    Path("results/bootstrap_ci.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n💾 บันทึก → results/bootstrap_ci.json")


if __name__ == "__main__":
    main()

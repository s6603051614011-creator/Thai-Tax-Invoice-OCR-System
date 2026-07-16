"""
eval_hires_infer.py -- อ่านที่ความละเอียดสูงกว่าตอนเทรน (ไม่ต้องเทรนใหม่)
=====================================================================
สมมติฐาน: ช่องที่คะแนนต่ำ (เงิน, เลขที่ใบ) โมเดล "เห็น" แต่ "อ่านหลักผิด"
-> เป็นปัญหาความคมชัด ไม่ใช่ปัญหาความเข้าใจ

ตอน inference ไม่ต้องเก็บ gradient -> VRAM ต่ำกว่าตอนเทรนมาก
-> อ่านที่ความละเอียดสูงกว่าที่เทรนไว้ได้ (แลกกับ train/inference mismatch เล็กน้อย)
เคยพิสูจน์แล้วรอบก่อน: เทรน 128 -> อ่าน 256 ดัน field acc ขึ้นจริง

วัดโมเดลตัวเดิม (เทรนที่ 320 tok) อ่านที่หลายความละเอียด เทียบ Field Accuracy

รัน: python eval_hires_infer.py --res 320 640 960
"""

import sys
import json
import time
import argparse
from pathlib import Path

import torch

import schema
import evaluate as ev
from evaluate import calc_field_accuracy, calc_cer, cfg

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PIX = 32 * 32   # Qwen3-VL: 32px ต่อ 1 image token


def load_test():
    samples = []
    with open(cfg.TEST_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def score(preds: dict, gts: dict) -> tuple:
    """คืน (field_acc dict, avg, cer เฉลี่ย)"""
    acc = {}
    for fld in cfg.FIELDS:
        k = fld["key"]
        m = [calc_field_accuracy(preds[i], gts[i], fld)["matched"] for i in preds]
        m = [x for x in m if x is not None]
        acc[k] = sum(m) / len(m) if m else None
    vals = [v for v in acc.values() if v is not None]
    cers = [calc_cer(schema.build_canonical_json(preds[i]),
                     schema.build_canonical_json(gts[i])) for i in preds]
    return acc, (sum(vals) / len(vals) if vals else 0.0), sum(cers) / len(cers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, nargs="+", default=[320, 640, 960],
                    help="จำนวน image token ที่จะทดสอบ (ที่เทรนไว้ = 320)")
    args = ap.parse_args()

    samples = load_test()
    print(f"test set: {len(samples)} ใบ | ความละเอียดที่ทดสอบ: {args.res} tokens\n")

    # โหลดโมเดล fine-tuned ครั้งเดียว (processor สร้างใหม่ต่อความละเอียด)
    model, _ = ev.load_model("finetuned")
    ProcessorClass = schema.get_processor_class()

    gts = {}
    results = {}

    for res in args.res:
        proc = ProcessorClass.from_pretrained(
            cfg.BASE_MODEL_ID,
            min_pixels=64 * PIX,
            max_pixels=res * PIX,
            trust_remote_code=True,
        )
        print(f"── อ่านที่ {res} tokens ({res*PIX:,} px) ─────────────", flush=True)
        preds = {}
        t0 = time.time()
        for n, s in enumerate(samples, 1):
            sid = s["id"]
            msgs = s["messages"]
            img_b64 = next(c["image"] for c in msgs[1]["content"] if c["type"] == "image")
            user_txt = next(c["text"] for c in msgs[1]["content"] if c["type"] == "text")
            if sid not in gts:
                gts[sid] = schema.parse_model_json(msgs[2]["content"])
            try:
                out = ev.run_inference(model, proc, img_b64, msgs[0]["content"], user_txt)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  OOM ที่ {res} tokens -> ข้ามความละเอียดนี้", flush=True)
                preds = None
                break
            except Exception as e:
                print(f"  {sid} error: {e}", flush=True)
                out = ""
            preds[sid] = schema.parse_model_json(out)
            if n % 10 == 0:
                print(f"  {n}/{len(samples)} ({time.time()-t0:.0f}s)", flush=True)

        if preds is None:
            continue
        acc, avg, cer_avg = score(preds, gts)
        results[res] = {"field_acc": acc, "avg_field": avg, "avg_cer": cer_avg,
                        "preds": {k: v for k, v in preds.items()},
                        "minutes": round((time.time() - t0) / 60, 1)}
        print(f"  -> Field Acc {avg:.1%} | CER {cer_avg:.1%} "
              f"({results[res]['minutes']} นาที)\n", flush=True)

    # ── ตารางสรุป ──
    if results:
        base = args.res[0]
        print("=" * 62)
        print(f"{'ช่องข้อมูล':<24}" + "".join(f"{r:>10}tok" for r in results))
        print("-" * 62)
        for fld in cfg.FIELDS:
            k = fld["key"]
            vals = [results[r]["field_acc"].get(k) for r in results]
            if all(v is None for v in vals):
                continue
            line = f"{schema.LABELS.get(k,k):<24}"
            for v in vals:
                line += f"{v:>12.1%}" if v is not None else f"{'-':>12}"
            print(line)
        print("-" * 62)
        line = f"{'Field Accuracy เฉลี่ย':<24}"
        for r in results:
            line += f"{results[r]['avg_field']:>12.1%}"
        print(line)
        line = f"{'CER (ยิ่งน้อยยิ่งดี)':<24}"
        for r in results:
            line += f"{results[r]['avg_cer']:>12.1%}"
        print(line)
        print("=" * 62)

        out_path = Path("results/hires_infer.json")
        out_path.write_text(json.dumps(
            {str(r): {kk: vv for kk, vv in v.items() if kk != "preds"} for r, v in results.items()},
            ensure_ascii=False, indent=1), encoding="utf-8")
        # เก็บคำตอบดิบไว้ใช้ต่อ (เช่น รวมกับ master list)
        Path("results/hires_preds.json").write_text(json.dumps(
            {str(r): v["preds"] for r, v in results.items()}, ensure_ascii=False), encoding="utf-8")
        print(f"บันทึก -> {out_path} + results/hires_preds.json")


if __name__ == "__main__":
    main()

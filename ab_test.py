"""
ab_test.py -- ทดสอบว่า preprocessing แบบไหนทำให้ Typhoon OCR อ่านใบกำกับแม่นสุด
====================================================================================
เทียบ 3 แบบ:
  (a) raw     -- ภาพดิบ ไม่ทำอะไร
  (b) vlm     -- light-touch คงสี (preprocess_vlm)
  (c) legacy  -- binarize แบบ OCR ดั้งเดิม (preprocess)

รัน:
  python ab_test.py                          # เลือกรูปยากสุด 3 ใบอัตโนมัติ
  python ab_test.py --n 5                     # 5 ใบ
  python ab_test.py --images inv_007,inv_012  # ระบุเอง

ผลลัพธ์: results/ab_preprocess/report.html  (เปิดดูเทียบข้างกัน + JSON ที่อ่านได้)

หมายเหตุ: ใช้โควต้า Typhoon OCR API  (จำนวนรูป x 3 calls)
"""

import sys
import json
import base64
import time
import argparse
from pathlib import Path

import cv2

import schema
import preprocess
from auto_label import call_typhoon_ocr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW_DIR   = Path("dataset/raw")
OUT_DIR   = Path("results/ab_preprocess")
API_WIDTH = 1800   # ย่อก่อนส่ง API เพื่อประหยัด token


# ── เลือกรูปยาก ──────────────────────────────────────
def sharpness(path: Path) -> float:
    """Laplacian variance: ยิ่งต่ำ = ยิ่งเบลอ/อ่านยาก"""
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return 1e9
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def pick_hardest(n: int) -> list[Path]:
    imgs = sorted(RAW_DIR.glob("*.jpg"))
    ranked = sorted(imgs, key=sharpness)   # เบลอสุดก่อน
    return ranked[:n]


# ── สร้าง variant + encode ───────────────────────────
def encode_bgr(bgr) -> str:
    h, w = bgr.shape[:2]
    if w > API_WIDTH:
        bgr = cv2.resize(bgr, (API_WIDTH, int(h * API_WIDTH / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


def make_variants(path: Path) -> dict:
    """คืน {variant_name: bgr_image}"""
    raw    = preprocess.load_oriented(str(path))   # raw + แก้ orientation เท่านั้น
    vlm    = preprocess.preprocess_vlm(str(path), verbose=False)
    legacy = preprocess.preprocess(str(path))      # grayscale/binary
    if legacy.ndim == 2:
        legacy = cv2.cvtColor(legacy, cv2.COLOR_GRAY2BGR)
    return {"raw": raw, "vlm": vlm, "legacy": legacy}


# ── ตัววัดคร่าว ๆ (proxy เมื่อยังไม่มี ground truth) ──
def completeness(fields: dict) -> dict:
    """นับว่าอ่านได้ครบแค่ไหน + เช็คความสมเหตุผลของตัวเลข"""
    norm   = schema.normalize_fields(fields)
    scalar = [k for k in schema.SCALAR_FIELDS if norm.get(k, "").strip()]
    n_items = len(norm.get("items", []))

    def num(x):
        try:
            return float(str(x).replace(",", ""))
        except ValueError:
            return None

    sub, vat, grand = num(norm["subtotal"]), num(norm["vat"]), num(norm["grand_total"])
    tax13 = sum(1 for k in ("seller_tax_id", "buyer_tax_id")
                if len(norm.get(k, "").replace(" ", "")) == 13)
    math_ok = (sub is not None and vat is not None and grand is not None
               and abs(sub + vat - grand) < 0.05)

    return {
        "scalar_filled": len(scalar),
        "scalar_total":  len(schema.SCALAR_FIELDS),
        "items":         n_items,
        "tax13":         tax13,
        "math_ok":       math_ok,
        "score":         len(scalar) + n_items + tax13 * 2 + (3 if math_ok else 0),
    }


# ── HTML report ──────────────────────────────────────
def b64_thumb(bgr, width=380) -> str:
    h, w = bgr.shape[:2]
    th = cv2.resize(bgr, (width, int(h * width / w)))
    ok, buf = cv2.imencode(".jpg", th, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode("utf-8")


VARIANT_LABEL = {"raw": "(a) RAW ดิบ", "vlm": "(b) VLM light-touch", "legacy": "(c) LEGACY binarize"}


def build_html(results: list) -> str:
    css = """
    body{font-family:'Segoe UI',Tahoma,sans-serif;background:#15151f;color:#e0e0e0;padding:20px;}
    h1{color:#a0c4ff;} h2{color:#4cc9f0;border-bottom:1px solid #333;padding-bottom:4px;margin-top:34px;}
    .row{display:flex;gap:14px;align-items:flex-start;}
    .col{flex:1;background:#1e1e2e;border-radius:8px;padding:10px;}
    .col h3{font-size:13px;margin-bottom:6px;text-align:center;}
    .raw h3{color:#bbb;} .vlm h3{color:#5fd38a;} .legacy h3{color:#f0a04b;}
    img{width:100%;border-radius:5px;border:1px solid #333;}
    pre{background:#0d1b2a;color:#c9d1d9;font-size:11px;padding:8px;border-radius:5px;
        max-height:340px;overflow:auto;white-space:pre-wrap;word-break:break-all;margin-top:6px;}
    .score{font-size:12px;text-align:center;margin-top:6px;padding:4px;border-radius:4px;background:#262636;}
    .best{outline:2px solid #5fd38a;}
    .legend{font-size:12px;color:#888;margin-top:6px;}
    """
    html = [f"<!DOCTYPE html><html lang='th'><head><meta charset='UTF-8'>",
            f"<title>A/B Preprocessing Test</title><style>{css}</style></head><body>",
            "<h1>A/B Test: Preprocessing สำหรับ Typhoon OCR</h1>",
            "<div class='legend'>score = scalar fields ที่อ่านได้ + จำนวน items + (เลขภาษี13หลัก ×2) + (ยอดเงินลงตัว +3) "
            "&nbsp;|&nbsp; เป็น proxy คร่าว ๆ ตัดสินจริงด้วยตาเทียบกับรูป</div>"]

    for r in results:
        html.append(f"<h2>{r['id']}  (sharpness={r['sharp']:.0f})</h2>")
        # หา variant ที่ score สูงสุด
        best = max(r["variants"], key=lambda v: v["comp"]["score"])["name"]
        html.append("<div class='row'>")
        for v in r["variants"]:
            c = v["comp"]
            cls = "best" if v["name"] == best else ""
            html.append(f"<div class='col {v['name']}'>")
            html.append(f"<h3>{VARIANT_LABEL[v['name']]}</h3>")
            html.append(f"<img class='{cls}' src='data:image/jpeg;base64,{v['thumb']}'>")
            html.append(f"<div class='score'>score <b>{c['score']}</b> | "
                        f"fields {c['scalar_filled']}/{c['scalar_total']} | "
                        f"items {c['items']} | tax13 {c['tax13']} | "
                        f"{'✓math' if c['math_ok'] else '✗math'}</div>")
            html.append(f"<pre>{json.dumps(v['fields_norm'], ensure_ascii=False, indent=2)}</pre>")
            html.append("</div>")
        html.append("</div>")

    html.append("</body></html>")
    return "\n".join(html)


# ── MAIN ─────────────────────────────────────────────
def main(images: list[Path]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"🧪 A/B test {len(images)} ใบ x 3 variant = {len(images)*3} API calls\n")

    for path in images:
        print(f"📄 {path.name} ...")
        variants = make_variants(path)
        vres = []
        for name, bgr in variants.items():
            print(f"   → {name:7s} ส่ง API...", end="", flush=True)
            fields = call_typhoon_ocr(encode_bgr(bgr)) or {}
            norm   = schema.normalize_fields(fields)
            vres.append({
                "name":        name,
                "thumb":       b64_thumb(bgr),
                "fields_norm": norm,
                "comp":        completeness(fields),
            })
            print(f" score={vres[-1]['comp']['score']}")
            time.sleep(1.0)
        results.append({"id": path.stem, "sharp": sharpness(path), "variants": vres})

    report = OUT_DIR / "report.html"
    report.write_text(build_html(results), encoding="utf-8")

    # สรุป win count
    print(f"\n{'='*46}\nสรุปผล (proxy score):")
    wins = {"raw": 0, "vlm": 0, "legacy": 0}
    for r in results:
        best = max(r["variants"], key=lambda v: v["comp"]["score"])
        wins[best["name"]] += 1
    for k, v in wins.items():
        print(f"   {VARIANT_LABEL[k]:24s} ชนะ {v} ใบ")
    print(f"\n📄 รายงาน: {report}")
    print(f"   เปิดไฟล์นี้ใน browser เพื่อเทียบรูป+ผล OCR ข้างกัน")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="จำนวนรูปยากที่เลือกอัตโนมัติ")
    ap.add_argument("--images", help="ระบุชื่อเอง คั่นด้วย comma เช่น inv_007,inv_012")
    args = ap.parse_args()

    if args.images:
        imgs = [RAW_DIR / f"{s.strip()}.jpg" if not s.strip().endswith(".jpg")
                else RAW_DIR / s.strip() for s in args.images.split(",")]
        imgs = [p for p in imgs if p.exists()]
    else:
        imgs = pick_hardest(args.n)

    if not imgs:
        print("❌ ไม่พบรูป")
    else:
        main(imgs)

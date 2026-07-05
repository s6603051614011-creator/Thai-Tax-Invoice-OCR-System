"""
format_dataset.py -- แปลง dataset/augmented_labels.json เป็น JSONL สำหรับเทรน
=====================================================================
สร้าง dataset/formatted/train.jsonl และ val.jsonl ตามรูปแบบที่ Finetune.py
(InvoiceDataset) คาดหวัง: แต่ละบรรทัดคือ 1 ตัวอย่าง messages (system/user/assistant)
โดยฝังรูปเป็น base64 data URI ไว้ในบรรทัดเดียวกับ label

ลำดับการรันที่ถูกต้อง:
  1. python augment.py           (สร้าง dataset/augmented/ + augmented_labels.json)
  2. python format_dataset.py    (ไฟล์นี้)
  3. python Finetune.py

กัน data leakage: ตัด id ใน splits.json["test"] ออกทั้งหมด (ทั้งต้นฉบับและ
augmented ของใบเดียวกัน) และแบ่ง train/val ที่ระดับ "ใบต้นฉบับ" เพื่อไม่ให้
augmented ของใบเดียวกันไปอยู่คนละฝั่งกับต้นฉบับ

กรอง sample ที่ token ยาวเกิน MAX_SEQ_LEN ทิ้งไปเลย (ไม่ใช่ตัด/truncate)
เพื่อไม่ให้ InvoiceDataset (Finetune.py) ต้องตัด assistant target ทิ้งกลางคัน
-- ค่า MAX_SEQ_LEN/MIN_PIXELS/MAX_PIXELS ต้องตรงกับ Finetune.py เป๊ะ ไม่งั้น
sample ที่กรองผ่านที่นี่อาจยังยาวเกินตอนเทรนจริง
"""

import sys
import json
import base64
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import schema

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


AUGMENTED_LABELS = Path("dataset/augmented_labels.json")
ANNOTATIONS      = Path("dataset/annotations_auto.json")
SPLITS           = Path("dataset/splits.json")
OUT_DIR          = Path("dataset/formatted")
VAL_RATIO        = 0.10
SEED             = 42

# ต้องตรงกับ Finetune.py Config เป๊ะ (ดูคอมเมนต์ที่ MAX_SEQ_LEN ในนั้น)
MAX_SEQ_LEN = 1024
MIN_PIXELS  = 64 * 28 * 28
MAX_PIXELS  = 256 * 28 * 28


def base_id(entry_id: str) -> str:
    """ตัด suffix '_aug_N' ออกเพื่อกลับไปหา id ใบต้นฉบับ"""
    idx = entry_id.find("_aug_")
    return entry_id[:idx] if idx != -1 else entry_id


def image_to_data_uri(path: str) -> str:
    ext  = Path(path).suffix.lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    data = Path(path).read_bytes()
    b64  = base64.b64encode(data).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def build_sample(entry: dict, data_uri: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": schema.SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": data_uri},
                {"type": "text",  "text": schema.USER_PROMPT},
            ]},
            {"role": "assistant", "content": entry["ground_truth"]["json"]},
        ]
    }


def load_processor():
    """โหลด processor เดียวกับ Finetune.py เพื่อวัด token length จริง (ไม่ใช่ประมาณ)"""
    from transformers import Qwen2_5_VLProcessor
    return Qwen2_5_VLProcessor.from_pretrained(
        schema.BASE_MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
        trust_remote_code=True,
    )


def token_len(processor, entry: dict, pil_image) -> int:
    conversation = [
        {"role": "system", "content": schema.SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": pil_image},
            {"type": "text",  "text": schema.USER_PROMPT},
        ]},
        {"role": "assistant", "content": entry["ground_truth"]["json"]},
    ]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt",
                        padding=False, truncation=False)
    return inputs["input_ids"].shape[1]


def main():
    if not AUGMENTED_LABELS.exists():
        print(f"❌ ไม่พบ {AUGMENTED_LABELS} — รัน augment.py ก่อน")
        return
    if not SPLITS.exists():
        print(f"❌ ไม่พบ {SPLITS} — รัน make_splits.py ก่อน")
        return

    entries = json.loads(AUGMENTED_LABELS.read_text(encoding="utf-8"))
    splits  = json.loads(SPLITS.read_text(encoding="utf-8"))
    test_ids = set(splits["test"])

    # กัน data leakage: ตัด test (ต้นฉบับ + augmented ของ id เดียวกัน) ออกทั้งหมด
    trainable    = [e for e in entries if base_id(e["id"]) not in test_ids]
    skipped_test = len(entries) - len(trainable)

    # กรอง sample ที่ยาวเกิน MAX_SEQ_LEN ทิ้ง (วัด token จริงด้วย processor เดียวกับ
    # ตอนเทรน) แทนการปล่อยให้ InvoiceDataset ตัด assistant target ทิ้งกลางคัน
    from PIL import Image, ImageOps
    print(f"🔍 วัด token length จริงของ {len(trainable)} samples (MAX_SEQ_LEN={MAX_SEQ_LEN})...")
    processor = load_processor()
    kept, dropped = [], []
    for i, e in enumerate(trainable):
        pil_image = ImageOps.exif_transpose(Image.open(e["image_path"])).convert("RGB")
        length = token_len(processor, e, pil_image)
        (kept if length <= MAX_SEQ_LEN else dropped).append(e)
        if (i + 1) % 200 == 0:
            print(f"   {i+1}/{len(trainable)}", flush=True)
    trainable = kept
    print(f"✓ กรองทิ้ง {len(dropped)} samples ที่ยาวเกิน {MAX_SEQ_LEN} tokens "
          f"({100*len(dropped)/(len(kept)+len(dropped)):.1f}%)")

    # แบ่ง train/val ที่ระดับ "ใบต้นฉบับ" กัน original/augmented ของใบเดียวกัน
    # หลุดไปคนละฝั่ง (data leakage ระหว่าง train กับ val)
    base_ids = sorted({base_id(e["id"]) for e in trainable})
    rnd = random.Random(SEED)
    rnd.shuffle(base_ids)
    n_val       = max(1, round(len(base_ids) * VAL_RATIO))
    val_base_ids = set(base_ids[:n_val])

    train_entries = [e for e in trainable if base_id(e["id"]) not in val_base_ids]
    val_entries   = [e for e in trainable if base_id(e["id"]) in val_base_ids]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", train_entries), ("val", val_entries)):
        out_path = OUT_DIR / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for e in subset:
                data_uri = image_to_data_uri(e["image_path"])
                f.write(json.dumps(build_sample(e, data_uri), ensure_ascii=False) + "\n")
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"✓ {name}.jsonl: {len(subset)} samples ({size_mb:.1f} MB) -> {out_path}")

    print(f"\n✓ ข้าม test set: {skipped_test} samples (กัน data leakage, "
          f"{len(test_ids)} ใบต้นฉบับ)")
    print(f"✓ ใบต้นฉบับที่ใช้เทรน: {len(base_ids) - n_val} train / {n_val} val")

    # ── test.jsonl (สำหรับ evaluate.py) ──
    # ใช้รูป raw ต้นฉบับตรงๆ (ไม่ผ่าน augment) ให้สะท้อนภาพถ่ายจริงตอนใช้งาน
    # ไม่กรอง MAX_SEQ_LEN ทิ้ง เพราะอยากวัดผลกับทุกใบ ไม่ใช่แค่ใบที่ "ง่าย"
    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    ann_by_id   = {e["id"]: e for e in annotations}
    test_entries = [ann_by_id[i] for i in sorted(test_ids) if i in ann_by_id]

    test_path = OUT_DIR / "test.jsonl"
    with test_path.open("w", encoding="utf-8") as f:
        for e in test_entries:
            data_uri = image_to_data_uri(e["image_path"])
            sample = build_sample(e, data_uri)
            sample["id"] = e["id"]
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    size_mb = test_path.stat().st_size / 1024 / 1024
    print(f"✓ test.jsonl: {len(test_entries)} samples ({size_mb:.1f} MB) -> {test_path}")


if __name__ == "__main__":
    main()

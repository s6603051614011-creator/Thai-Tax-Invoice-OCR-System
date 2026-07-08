"""
Data Augmentation สำหรับใบกำกับภาษีไทย
==========================================
ติดตั้ง: pip install Pillow albumentations opencv-python numpy tqdm

ลำดับการรันที่ถูกต้อง:
  1. python preprocess.py --batch dataset/raw/ --output dataset/preprocessed/
  2. python augment.py   ← รันหลัง preprocess เสมอ
  3. python format_dataset.py

การแก้ไขในเวอร์ชันนี้:
  - แก้ปัญหารูปกลับหัว/แนวนอน ด้วย EXIF transpose
  - ดึงรูปจาก preprocessed/ ก่อน raw/ (fallback)
  - จับคู่ชื่อไฟล์แบบ case-insensitive (.jpg/.JPG/.jpeg)
  - แสดงสรุปชัดเจนว่าข้ามกี่ไฟล์และเพราะอะไร
"""

import json
import random
import shutil
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image, ImageOps
import albumentations as A


# ──────────────────────────────────────────
# Config
# ──────────────────────────────────────────
RAW_DIR        = Path("dataset/raw")
PREP_DIR       = Path("dataset/preprocessed")
AUG_DIR        = Path("dataset/augmented")
ANNOTATIONS    = Path("dataset/annotations_auto.json")
AUG_LABELS_OUT = Path("dataset/augmented_labels.json")

# เบาๆ ตั้งใจ: มีรูปจริง 402 ใบอยู่แล้ว (ไม่ใช่ ~50 ใบตามแผนเดิม) แค่ต้องการ
# ให้โมเดลทนต่อสภาพถ่ายจริง (เอียงนิดหน่อย, แสงไม่สม่ำเสมอ, กล้องมือถือสั่น)
# ไม่ต้องการขยายข้อมูลจนล้น (เสี่ยง overfit กับ noise สังเคราะห์แทนเนื้อหาจริง)
AUGMENT_PER_IMAGE = 2
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ──────────────────────────────────────────
# Augmentation Pipelines (เบาๆ — ไม่มี perspective warp / crop ที่เสี่ยงตัดเนื้อหา
# หรือบิดตัวอักษรจนอ่านไม่ออก เน้นจำลองสภาพถ่ายจริงเท่านั้น)
# ──────────────────────────────────────────

pipeline_photo_light = A.Compose([
    A.Rotate(limit=4, border_mode=cv2.BORDER_REPLICATE, p=0.6),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 3), p=0.5),
        A.MotionBlur(blur_limit=3, p=0.5),
    ], p=0.25),
    A.GaussNoise(std_range=(0.02, 0.05), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
])

pipeline_scan_light = A.Compose([
    A.Rotate(limit=2, border_mode=cv2.BORDER_REPLICATE, p=0.4),
    A.RandomBrightnessContrast(brightness_limit=(-0.15, 0.2), contrast_limit=(-0.1, 0.2), p=0.6),
    A.ImageCompression(quality_range=(80, 95), p=0.3),
    A.ToGray(p=0.15),
])

PIPELINES = [
    ("photo_light", pipeline_photo_light),
    ("scan_light",  pipeline_scan_light),
]


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────

def open_image(path: Path) -> Image.Image:
    """
    เปิดรูปและแก้ orientation จาก EXIF อัตโนมัติ
    แก้ปัญหารูปกลับหัว/แนวนอนจากภาพถ่ายโทรศัพท์
    """
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    return img


def build_raw_index(raw_dir: Path) -> dict:
    """
    สร้าง index ของรูปทั้งหมดใน raw/ โดยใช้ stem lowercase เป็น key
    รองรับ .jpg .JPG .jpeg .JPEG .png .PNG
    """
    index = {}
    for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]:
        for p in raw_dir.glob(ext):
            key = p.stem.lower()
            if key not in index:
                index[key] = p
    return index


def find_in_dir(stem: str, directory: Path) -> Path | None:
    """ค้นหาไฟล์รูปใน directory แบบ case-insensitive"""
    for ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def augment_image(pil_img: Image.Image, pipeline_name: str, pipeline) -> Image.Image:
    cv_img = pil_to_cv(pil_img)

    try:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        result = pipeline(image=rgb)
        return cv_to_pil(cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"\n   Pipeline '{pipeline_name}' error: {e} → ใช้ต้นฉบับแทน")
        return pil_img


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────

def run_augmentation():
    # ตรวจสอบไฟล์
    if not ANNOTATIONS.exists():
        print(f"ไม่พบ {ANNOTATIONS}")
        return

    if not RAW_DIR.exists():
        print(f"ไม่พบโฟลเดอร์ {RAW_DIR}")
        return

    # โหลด annotations
    with open(ANNOTATIONS, encoding="utf-8") as f:
        annotations = json.load(f)

    # สร้าง map stem_lowercase → annotation
    ann_map = {Path(a["image_path"]).stem.lower(): a for a in annotations}

    # สร้าง index รูปจริงใน raw/
    raw_index = build_raw_index(RAW_DIR)

    AUG_DIR.mkdir(parents=True, exist_ok=True)

    # สรุปก่อนเริ่ม
    verified_count = sum(1 for a in annotations if a.get("verified"))
    prep_count = len(list(PREP_DIR.glob("*"))) if PREP_DIR.exists() else 0
    print(f"Annotations ทั้งหมด: {len(annotations)} | Verified: {verified_count}")
    print(f"รูปใน raw/:          {len(raw_index)} ไฟล์")
    print(f"รูปใน preprocessed/: {prep_count} ไฟล์")
    print(f"คาดว่าจะได้:         {verified_count * (AUGMENT_PER_IMAGE + 1)} ใบ\n")

    augmented_entries = []
    skip_no_ann   = 0
    skip_no_verify = 0
    skip_no_file  = 0

    for stem_lower, img_path in tqdm(raw_index.items(), desc="Augmenting"):
        filename = img_path.name

        # หา annotation
        annotation = ann_map.get(stem_lower)
        if annotation is None:
            skip_no_ann += 1
            continue

        if not annotation.get("verified", False):
            skip_no_verify += 1
            continue

        # ── Copy รูปต้นฉบับ (raw) → augmented/ สำหรับ test set ──
        shutil.copy2(img_path, AUG_DIR / filename)
        orig_entry = annotation.copy()
        orig_entry["image_path"] = f"dataset/augmented/{filename}"
        orig_entry["augmented"]  = False
        orig_entry["aug_type"]   = "original"
        orig_entry["source"]     = "raw"
        augmented_entries.append(orig_entry)

        # ── เลือก source สำหรับ Augmentation ──
        prep_path = find_in_dir(img_path.stem, PREP_DIR) if PREP_DIR.exists() else None
        source_path  = prep_path if prep_path else img_path
        source_label = "preprocessed" if prep_path else "raw"

        # ── เปิดรูปพร้อมแก้ EXIF orientation ──
        try:
            pil_img = open_image(source_path)
        except Exception as e:
            print(f"\n  เปิดรูปไม่ได้ {filename}: {e}")
            skip_no_file += 1
            continue

        # ── รัน Augmentation (เบาๆ, AUGMENT_PER_IMAGE แบบ) ──
        stem = img_path.stem
        for i in range(AUGMENT_PER_IMAGE):
            p_name, p_logic = PIPELINES[i % len(PIPELINES)]
            aug_filename = f"{stem}_aug_{i+1}_{p_name}.jpg"

            aug_img = augment_image(pil_img, p_name, p_logic)
            aug_img.save(AUG_DIR / aug_filename, "JPEG", quality=90)

            aug_entry = annotation.copy()
            aug_entry["id"]         = f"{annotation['id']}_aug_{i+1}"
            aug_entry["image_path"] = f"dataset/augmented/{aug_filename}"
            aug_entry["augmented"]  = True
            aug_entry["aug_type"]   = p_name
            aug_entry["source"]     = source_label
            augmented_entries.append(aug_entry)

    # บันทึก labels
    with open(AUG_LABELS_OUT, "w", encoding="utf-8") as f:
        json.dump(augmented_entries, f, ensure_ascii=False, indent=2)

    # สรุป
    n_orig      = sum(1 for e in augmented_entries if not e.get("augmented"))
    n_aug       = sum(1 for e in augmented_entries if e.get("augmented"))
    n_from_prep = sum(1 for e in augmented_entries if e.get("augmented") and e.get("source") == "preprocessed")
    n_from_raw  = sum(1 for e in augmented_entries if e.get("augmented") and e.get("source") == "raw")

    print(f"""
{'='*50}
Augmentation เสร็จสิ้น!

ผลลัพธ์:
   ต้นฉบับ (test set):          {n_orig} ใบ
   Augmented (preprocessed):    {n_from_prep} ใบ
   Augmented (raw fallback):    {n_from_raw} ใบ
   รวมทั้งหมด:                  {len(augmented_entries)} ใบ

 ข้าม:
   ไม่มี annotation:  {skip_no_ann} ไฟล์
   ยังไม่ Verified:   {skip_no_verify} ไฟล์
   เปิดรูปไม่ได้:    {skip_no_file} ไฟล์

Output:
   รูปภาพ → {AUG_DIR}/
   Labels  → {AUG_LABELS_OUT}
{'='*50}""")

    if n_from_raw > 0:
        print(f"\nมี {n_from_raw} ใบที่ใช้รูปดิบ (raw)")
        print(f"   รัน preprocess.py ก่อนเพื่อผลที่ดีที่สุดครับ")

    if skip_no_verify > 0:
        print(f"\n มี {skip_no_verify} ใบที่ยังไม่ Verified")
        print(f"   เปิด Annotation Tool → Verify → Export ใหม่ แล้วรัน augment.py อีกครั้ง")

    if len(augmented_entries) == 0:
        print("\nไม่ได้รูปเลย — ตรวจสอบ:")
        print("   1. ชื่อไฟล์รูปใน dataset/raw/ ตรงกับ image_path ใน annotations.json ไหม")
        print("   2. มี verified: true อย่างน้อย 1 entry ไหม")


if __name__ == "__main__":
    run_augmentation()
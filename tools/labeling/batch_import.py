"""
batch_import.py -- นำเข้ารูปใบกำกับภาษีชุดใหม่เข้า dataset/raw/
=================================================================
รัน:
  python batch_import.py --src "C:/path/to/new_invoices/"
  python batch_import.py --src new_invoices/ --preview   <- ดูก่อนโดยไม่ copy จริง

จะ rename อัตโนมัติต่อจากเลขสุดท้ายที่มีอยู่ใน dataset/raw/
"""

import sys
import argparse
import shutil
from pathlib import Path

# รองรับ terminal ที่ encoding ไม่ใช่ UTF-8 (เช่น CP874 บน Windows)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW_DIR = Path("dataset/raw")
SUPPORTED = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def get_next_index(raw_dir: Path) -> int:
    """หาเลข index ถัดไปจากไฟล์ inv_XXX.jpg ที่มีอยู่แล้ว"""
    existing = [
        p for p in raw_dir.glob("inv_*.jpg")
        if p.stem.startswith("inv_") and p.stem[4:].isdigit()
    ]
    if not existing:
        return 1
    return max(int(p.stem[4:]) for p in existing) + 1


def collect_images(src: Path) -> list[Path]:
    """รวบรวมไฟล์รูปทั้งหมดจาก src (ไม่ recursive)"""
    return sorted(p for p in src.iterdir() if p.suffix in SUPPORTED)


def run(src: Path, preview: bool = False):
    if not src.exists():
        print(f"[ERROR] ไม่พบโฟลเดอร์: {src}")
        return

    images = collect_images(src)
    if not images:
        print(f"[ERROR] ไม่พบไฟล์รูปใน {src}")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    start_idx = get_next_index(RAW_DIR)

    print(f"[SRC]   โฟลเดอร์ต้นทาง  : {src}")
    print(f"[DEST]  โฟลเดอร์ปลายทาง: {RAW_DIR}")
    print(f"[COUNT] พบรูปทั้งหมด    : {len(images)} ใบ")
    print(f"[START] เริ่ม numbering  : inv_{start_idx:03d}")
    if preview:
        print("[PREVIEW] -- ยังไม่ได้ copy จริง --\n")
    else:
        print()

    copied = 0
    skipped = 0

    for i, src_path in enumerate(images):
        dest_name = f"inv_{start_idx + i:03d}.jpg"
        dest_path = RAW_DIR / dest_name

        if dest_path.exists():
            print(f"  [SKIP] {dest_name} มีอยู่แล้ว")
            skipped += 1
            continue

        print(f"  {src_path.name:40s}  ->  {dest_name}")

        if not preview:
            shutil.copy2(src_path, dest_path)
            copied += 1

    end_idx = start_idx + len(images) - 1 - skipped
    count = len(images) - skipped

    print(f"""
{'='*55}
{'[DONE] นำเข้าเสร็จสิ้น!' if not preview else '[PREVIEW] จบการแสดงตัวอย่าง (ยังไม่ได้ copy)'}

สรุป:
  นำเข้าสำเร็จ : {count} ใบ
  ข้ามไป       : {skipped} ใบ (มีอยู่แล้ว)
  ชื่อไฟล์     : inv_{start_idx:03d}.jpg -> inv_{end_idx:03d}.jpg

ขั้นตอนต่อไป:
  python auto_label.py --merge dataset/annotations.json
{'='*55}""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="นำเข้ารูปใบกำกับชุดใหม่เข้า dataset/raw/")
    parser.add_argument("--src", required=True, help="โฟลเดอร์รูปต้นทาง")
    parser.add_argument("--preview", action="store_true", help="ดูตัวอย่างโดยไม่ copy จริง")
    args = parser.parse_args()

    run(Path(args.src), preview=args.preview)

"""
Image Preprocessing Pipeline สำหรับใบกำกับภาษีไทย
=====================================================
ติดตั้ง: pip install opencv-python Pillow numpy

วิธีใช้:
  python preprocess.py                        # ดู Demo ผลลัพธ์
  python preprocess.py --input invoice.jpg    # ประมวลผลไฟล์เดียว
  python preprocess.py --batch dataset/raw/   # ประมวลผลทั้งโฟลเดอร์

อ้างอิง:
  - Typhoon OCR Technical Report (arXiv:2601.14722)
  - PreP-OCR Pipeline (arXiv:2505.20429)
  - Image Pre-Processing Techniques for OCR (Medium, 2025)
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
import argparse
import os

# รองรับ terminal ที่ encoding ไม่ใช่ UTF-8 (เช่น CP874 บน Windows) -> กัน emoji crash
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ══════════════════════════════════════════════
# CONFIG — ปรับค่าเหล่านี้ตามคุณภาพภาพที่ใช้จริง
# ══════════════════════════════════════════════
class Config:
    TARGET_WIDTH     = 1800     # ความกว้างมาตรฐาน (px) — จาก Typhoon OCR paper
    MIN_DPI_EQUIV    = 300      # DPI ขั้นต่ำที่ OCR ทำงานได้ดี
    ADAPTIVE_BLOCK   = 11       # ขนาด block สำหรับ Adaptive Threshold (ต้องเป็นเลขคี่)
    ADAPTIVE_C       = 2        # ค่าปรับ Adaptive Threshold
    DESKEW_THRESHOLD = 0.5      # องศาขั้นต่ำที่จะแก้ (< 0.5 ถือว่าตรงพอ)
    DENOISE_STRENGTH = 10       # ความแรงของ Noise Removal (10-30)
    BORDER_MARGIN    = 10       # pixel ที่จะ crop ขอบ


# ══════════════════════════════════════════════
# STEP 1: Resize & Resolution Normalization
# ══════════════════════════════════════════════
def resize_normalize(img: np.ndarray, target_width: int = Config.TARGET_WIDTH) -> np.ndarray:
    """
    ปรับขนาดรูปให้กว้าง target_width โดยรักษา aspect ratio
    
    ทำไมต้อง 1800px:
    - น้อยกว่า 1000px → OCR อ่านผิดบ่อย (ตัวอักษรเล็กเกิน)
    - มากกว่า 2400px → ช้า และไม่ได้ accuracy เพิ่มมากนัก
    - 1800px → จุดที่ดีที่สุด (ตาม Typhoon OCR research)
    """
    h, w = img.shape[:2]
    
    if w == target_width:
        return img
    
    scale = target_width / w
    new_h = int(h * scale)
    
    # INTER_CUBIC ดีกว่า INTER_LINEAR สำหรับ upscale
    # INTER_AREA ดีกว่าสำหรับ downscale
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(img, (target_width, new_h), interpolation=interp)
    
    return resized


# ══════════════════════════════════════════════
# STEP 2: Grayscale Conversion
# ══════════════════════════════════════════════
def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    แปลงภาพสีเป็น Grayscale
    
    ทำไม: OCR ทำงานกับ intensity (ความสว่าง) ไม่ใช่สี
    การแปลงช่วยลด noise จากสีที่ไม่จำเป็น
    """
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img  # ถ้าเป็น grayscale อยู่แล้ว


# ══════════════════════════════════════════════
# STEP 3: Contrast Enhancement (CLAHE)
# ══════════════════════════════════════════════
def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """
    CLAHE = Contrast Limited Adaptive Histogram Equalization
    
    ทำไมดีกว่า Histogram Equalization ปกติ:
    - ปรับ contrast แยกตามพื้นที่ (tile) ไม่ใช่ทั้งรูป
    - ดีมากสำหรับใบกำกับที่มีแสงไม่สม่ำเสมอจากการถ่ายรูป
    - "Contrast Limited" ป้องกัน over-amplification ของ noise
    
    ค่าที่แนะนำ:
    - clipLimit=2.0: ถ้าสูงกว่า → contrast สูงแต่ noise มากขึ้น
    - tileGridSize=(8,8): ขนาด tile ถ้าเล็กกว่า → local มากขึ้น
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ══════════════════════════════════════════════
# STEP 4: Noise Removal
# ══════════════════════════════════════════════
def remove_noise(gray: np.ndarray) -> np.ndarray:
    """
    ลบ noise โดยไม่ทำลาย edge ของตัวอักษร
    
    ใช้ Non-Local Means Denoising:
    - ดีกว่า Gaussian Blur เพราะรักษา sharp edge ไว้
    - h=10: ความแรงในการลบ noise ค่าสูงขึ้น → smooth มากขึ้น แต่เบลอ
    - templateWindowSize=7: ขนาด patch เปรียบเทียบ
    - searchWindowSize=21: พื้นที่ค้นหา patch ที่คล้ายกัน
    
    สำหรับใบกำกับถ่ายมือถือ: h=10 เหมาะสม
    สำหรับใบกำกับสแกน: h=5-7 ก็พอ (noise น้อยกว่า)
    """
    denoised = cv2.fastNlMeansDenoising(
        gray,
        h=Config.DENOISE_STRENGTH,
        templateWindowSize=7,
        searchWindowSize=21
    )
    return denoised


# ══════════════════════════════════════════════
# STEP 5: Binarization (Adaptive Threshold)
# ══════════════════════════════════════════════
def binarize(gray: np.ndarray) -> np.ndarray:
    """
    แปลงเป็น Binary (ขาว-ดำ) ด้วย Adaptive Threshold
    
    ทำไมใช้ Adaptive แทน Global:
    - ใบกำกับมักมีแสงไม่สม่ำเสมอ (มุมมืด/สว่าง)
    - Global threshold: ถ้าตั้งค่าเดียว → มืดเกินในบางพื้นที่
    - Adaptive: คำนวณ threshold แต่ละ block ขนาด 11x11 px
    
    ADAPTIVE_THRESH_GAUSSIAN_C:
    - ใช้ Gaussian weighted sum ของ neighborhood
    - ดีกว่า ADAPTIVE_THRESH_MEAN_C สำหรับข้อความ
    
    THRESH_BINARY_INV:
    - ตัวอักษรเป็นสีดำ พื้นหลังเป็นสีขาว (มาตรฐาน OCR)
    """
    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=Config.ADAPTIVE_BLOCK,   # ต้องเป็นเลขคี่
        C=Config.ADAPTIVE_C
    )
    return binary


# ══════════════════════════════════════════════
# STEP 6: Deskewing (แก้ภาพเอียง)
# ══════════════════════════════════════════════
def deskew(binary: np.ndarray) -> tuple[np.ndarray, float]:
    """
    ตรวจจับและแก้ภาพที่เอียง โดยใช้ Hough Line Transform
    
    หลักการ:
    1. หาเส้นทั้งหมดในภาพ (บรรทัดของข้อความ)
    2. คำนวณมุมเฉลี่ยของเส้นเหล่านั้น
    3. หมุนภาพกลับด้วยมุมที่คำนวณได้
    
    คืนค่า: (ภาพที่แก้แล้ว, มุมที่หมุน)
    """
    # หา edges ก่อน
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    
    # Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    
    if lines is None:
        return binary, 0.0
    
    # เก็บมุมของเส้นแนวนอน (เส้นที่มุม 80-100 องศา)
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle_deg = np.degrees(theta) - 90
        if abs(angle_deg) < 20:  # เอาเฉพาะที่เอียงไม่มาก
            angles.append(angle_deg)
    
    if not angles:
        return binary, 0.0
    
    # มุมเฉลี่ย (ใช้ median เพื่อป้องกัน outlier)
    skew_angle = np.median(angles)
    
    # ถ้าเอียงน้อยมาก ไม่ต้องแก้
    if abs(skew_angle) < Config.DESKEW_THRESHOLD:
        return binary, skew_angle
    
    # หมุนภาพ
    h, w = binary.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    deskewed = cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE  # เติมขอบด้วยสีใกล้เคียง
    )
    
    return deskewed, skew_angle


# ══════════════════════════════════════════════
# STEP 7: Border Removal
# ══════════════════════════════════════════════
def remove_border(img: np.ndarray, margin: int = Config.BORDER_MARGIN) -> np.ndarray:
    """
    ตัดขอบภาพออก เพื่อลบเงา/พื้นที่ว่างรอบใบกำกับ
    
    ใช้วิธีหา bounding box ของพื้นที่ที่มีข้อความ
    แล้ว crop เฉพาะส่วนนั้น + margin เล็กน้อย
    """
    # หา contour ของพื้นที่ที่มีข้อความ
    contours, _ = cv2.findContours(
        cv2.bitwise_not(img),  # invert เพราะตัวอักษรเป็นดำ
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return img
    
    # หา bounding box รวมของทุก contour
    x_min, y_min = img.shape[1], img.shape[0]
    x_max, y_max = 0, 0
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 100:  # ข้าม noise เล็กๆ
            continue
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    # Crop พร้อม margin
    h_img, w_img = img.shape
    x1 = max(0, x_min - margin)
    y1 = max(0, y_min - margin)
    x2 = min(w_img, x_max + margin)
    y2 = min(h_img, y_max + margin)
    
    return img[y1:y2, x1:x2]


# ══════════════════════════════════════════════
# VLM-FRIENDLY PIPELINE (light touch) — สำหรับ Typhoon OCR / Qwen2.5-VL
# ══════════════════════════════════════════════
# หลักการ: VLM เทรนจากภาพสี/เทาธรรมชาติ -> "เพิ่มความอ่านง่าย" โดยไม่ทำลายข้อมูล
#   แก้ orientation, deskew เบา, contrast เบา (คงสี), upscale, sharpen เบา
#   ไม่ binarize, ไม่ denoise แรง (กันวรรณยุกต์/สระไทยพัง)
VLM_TARGET_WIDTH = 1500


def load_oriented(img_path: str) -> np.ndarray:
    """โหลดรูป + แก้ EXIF orientation (รูปตะแคง/กลับหัว) คืนเป็น BGR สำหรับ OpenCV"""
    pil = Image.open(img_path)
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def estimate_skew(bgr: np.ndarray) -> float:
    """ประเมินมุมเอียงจากเส้นข้อความ (บน grayscale ไม่ใช่ binary) ใช้ median กัน outlier"""
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
    if lines is None:
        return 0.0
    angles = []
    for line in lines:
        _, theta = line[0]
        a = np.degrees(theta) - 90
        if abs(a) < 15:          # เอาเฉพาะเส้นที่เกือบแนวนอน
            angles.append(a)
    return float(np.median(angles)) if angles else 0.0


def gentle_deskew(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """หมุนแก้เอียงบน 'ภาพสี' เติมขอบด้วยสีขาว (พื้นใบกำกับ) — ทำเฉพาะเมื่อเอียงชัด"""
    angle = estimate_skew(bgr)
    if abs(angle) < Config.DESKEW_THRESHOLD:
        return bgr, angle
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        bgr, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
    return rotated, angle


def enhance_color_contrast(bgr: np.ndarray) -> np.ndarray:
    """CLAHE บนช่อง L ของ LAB — เพิ่ม contrast แต่ 'คงสี' ไว้ (ดีสำหรับรูปถ่ายแสงไม่สม่ำเสมอ)"""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def upscale_if_small(bgr: np.ndarray, target_width: int = VLM_TARGET_WIDTH) -> np.ndarray:
    """ขยายภาพถ้ากว้างน้อยกว่า target (ตัวอักษรแน่นจะอ่านง่ายขึ้น) — ไม่ย่อภาพใหญ่"""
    w = bgr.shape[1]
    if w >= target_width:
        return bgr
    scale = target_width / w
    new_h = int(bgr.shape[0] * scale)
    return cv2.resize(bgr, (target_width, new_h), interpolation=cv2.INTER_CUBIC)


def mild_sharpen(bgr: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """Unsharp mask เบาๆ ให้ขอบตัวอักษรคมขึ้นเล็กน้อย (ไม่แรงจนเกิด halo)"""
    blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(bgr, 1 + amount, blur, -amount, 0)


def preprocess_vlm(img_path: str, verbose: bool = True) -> np.ndarray:
    """
    Light-touch pipeline สำหรับ Vision-Language Model (Typhoon OCR)
    คืนภาพ 'สี' ที่อ่านง่ายขึ้นโดยไม่ทำลายเส้นภาษาไทย
    """
    if verbose:
        print(f"\n[VLM] Processing: {Path(img_path).name}")

    bgr = load_oriented(img_path)
    bgr, angle = gentle_deskew(bgr)
    bgr = enhance_color_contrast(bgr)
    bgr = upscale_if_small(bgr)
    bgr = mild_sharpen(bgr)

    if verbose:
        msg = f"หมุน {angle:.2f}°" if abs(angle) >= Config.DESKEW_THRESHOLD else "ตรงดีแล้ว"
        print(f"  orientation + deskew ({msg}) + CLAHE สี + upscale + sharpen")
    return bgr


def process_batch_vlm(input_dir: str, output_dir: str):
    """ประมวลผลทั้งโฟลเดอร์ด้วย VLM pipeline (คงสี)"""
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images = sorted(
        list(input_path.glob("*.jpg")) +
        list(input_path.glob("*.jpeg")) +
        list(input_path.glob("*.png"))
    )
    if not images:
        print(f"ไม่พบรูปภาพใน {input_dir}")
        return

    print(f"[VLM mode] พบ {len(images)} ไฟล์ → ประมวลผล...\n")
    success, failed = 0, 0
    for img_path in images:
        try:
            result = preprocess_vlm(str(img_path), verbose=False)
            cv2.imwrite(str(output_path / img_path.name), result,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            success += 1
        except Exception as e:
            print(f"  {img_path.name}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"สำเร็จ: {success} ไฟล์" + (f" | {failed}" if failed else ""))
    print(f"Output → {output_dir}")


# ══════════════════════════════════════════════
# MAIN PIPELINE (legacy — OCR ดั้งเดิม/binarize, เก็บไว้สำหรับ A/B test)
# ══════════════════════════════════════════════
def preprocess(img_path: str, save_steps: bool = False, output_dir: str = None) -> np.ndarray:
    """
    รัน pipeline ครบทุกขั้นตอน
    
    Args:
        img_path:   path ของไฟล์รูปภาพ
        save_steps: ถ้า True → บันทึกผลแต่ละ step เพื่อ debug
        output_dir: โฟลเดอร์ที่บันทึก output
    
    Returns:
        numpy array ของภาพที่ผ่าน preprocessing แล้ว
    """
    print(f"\nProcessing: {Path(img_path).name}")
    
    # โหลดรูป
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"ไม่พบไฟล์: {img_path}")
    
    original_shape = img.shape
    steps = {}

    # ─── Step 1: Resize ───
    img = resize_normalize(img)
    steps["1_resize"] = img.copy()
    print(f"  Resize: {original_shape[1]}x{original_shape[0]} → {img.shape[1]}x{img.shape[0]}")

    # ─── Step 2: Grayscale ───
    gray = to_grayscale(img)
    steps["2_grayscale"] = gray.copy()
    print(f"  Grayscale")

    # ─── Step 3: Contrast Enhancement ───
    enhanced = enhance_contrast(gray)
    steps["3_contrast"] = enhanced.copy()
    print(f"  Contrast Enhancement (CLAHE)")

    # ─── Step 4: Noise Removal ───
    denoised = remove_noise(enhanced)
    steps["4_denoised"] = denoised.copy()
    print(f"  Noise Removal")

    # ─── Step 5: Binarization ───
    binary = binarize(denoised)
    steps["5_binary"] = binary.copy()
    print(f"  Binarization (Adaptive Threshold)")

    # ─── Step 6: Deskew ───
    deskewed, angle = deskew(binary)
    steps["6_deskewed"] = deskewed.copy()
    if abs(angle) >= Config.DESKEW_THRESHOLD:
        print(f"  Deskew: หมุน {angle:.2f}°")
    else:
        print(f"  Deskew: ภาพตรงดีแล้ว ({angle:.2f}°)")

    # ─── Step 7: Border Removal ───
    final = remove_border(deskewed)
    steps["7_final"] = final.copy()
    print(f"  Border Removal: {deskewed.shape[1]}x{deskewed.shape[0]} → {final.shape[1]}x{final.shape[0]}")

    # บันทึก steps (ถ้าต้องการ)
    if save_steps and output_dir:
        stem = Path(img_path).stem
        step_dir = Path(output_dir) / f"{stem}_steps"
        step_dir.mkdir(parents=True, exist_ok=True)
        for step_name, step_img in steps.items():
            cv2.imwrite(str(step_dir / f"{step_name}.jpg"), step_img)
        print(f"  บันทึก steps → {step_dir}/")

    return final


# ══════════════════════════════════════════════
# BATCH PROCESSING
# ══════════════════════════════════════════════
def process_batch(input_dir: str, output_dir: str):
    """ประมวลผลทั้งโฟลเดอร์"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images = list(input_path.glob("*.jpg")) + \
             list(input_path.glob("*.jpeg")) + \
             list(input_path.glob("*.png"))
    
    if not images:
        print(f"ไม่พบรูปภาพใน {input_dir}")
        return
    
    print(f"พบ {len(images)} ไฟล์ → ประมวลผล...\n")
    success, failed = 0, 0
    
    for img_path in images:
        try:
            result = preprocess(str(img_path))
            out_path = output_path / img_path.name
            cv2.imwrite(str(out_path), result)
            success += 1
        except Exception as e:
            print(f"  Error: {e}")
            failed += 1
    
    print(f"\n{'='*40}")
    print(f"สำเร็จ: {success} ไฟล์")
    if failed:
        print(f"ผิดพลาด: {failed} ไฟล์")
    print(f"Output → {output_dir}")


# ══════════════════════════════════════════════
# DEMO (ทดสอบโดยไม่มีไฟล์จริง)
# ══════════════════════════════════════════════
def run_demo():
    """สร้างภาพจำลองและทดสอบ pipeline"""
    print("Demo Mode: สร้างใบกำกับจำลอง\n")
    
    # สร้างภาพจำลองใบกำกับ
    img = np.ones((800, 600, 3), dtype=np.uint8) * 240
    
    # เพิ่มข้อความจำลอง
    texts = [
        ("ใบกำกับภาษี / Tax Invoice", (50, 60), 0.7),
        ("บริษัท ตัวอย่าง จำกัด",     (50, 100), 0.6),
        ("เลขที่: INV-2024-00123",    (50, 140), 0.5),
        ("วันที่: 01/03/2567",         (50, 170), 0.5),
        ("ยอดรวม: 1,872.50 บาท",      (50, 600), 0.6),
    ]
    for text, pos, scale in texts:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                   scale, (30, 30, 30), 1, cv2.LINE_AA)
    
    # เพิ่ม noise จำลองภาพถ่าย
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # เพิ่มความเอียงเล็กน้อย
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), 3, 1.0)
    img = cv2.warpAffine(img, M, (w, h))
    
    # บันทึกและประมวลผล
    demo_input = "demo_invoice.jpg"
    demo_output_dir = "demo_output"
    cv2.imwrite(demo_input, img)
    
    result = preprocess(demo_input, save_steps=True, output_dir=demo_output_dir)
    
    out_path = Path(demo_output_dir) / "demo_result.jpg"
    Path(demo_output_dir).mkdir(exist_ok=True)
    cv2.imwrite(str(out_path), result)
    
    print(f"\n{'='*40}")
    print(f"Demo เสร็จแล้ว!")
    print(f"ดูผลลัพธ์ที่: {demo_output_dir}/")
    print(f"   - demo_result.jpg     → ผลลัพธ์สุดท้าย")
    print(f"   - demo_invoice_steps/ → ผลแต่ละ step")
    
    # ลบไฟล์ temp
    os.remove(demo_input)


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invoice Image Preprocessor")
    parser.add_argument("--input",   help="path ของไฟล์รูปภาพเดียว")
    parser.add_argument("--batch",   help="path ของโฟลเดอร์ (batch mode)")
    parser.add_argument("--output",  help="โฟลเดอร์ output", default="dataset/preprocessed")
    parser.add_argument("--mode",    choices=["vlm", "legacy"], default="vlm",
                        help="vlm = light-touch คงสี (แนะนำสำหรับ Typhoon OCR) | "
                             "legacy = binarize แบบ OCR ดั้งเดิม")
    parser.add_argument("--steps",   action="store_true", help="บันทึกผลแต่ละ step (legacy เท่านั้น)")
    args = parser.parse_args()

    if args.input:
        # Single file
        if args.mode == "vlm":
            result = preprocess_vlm(args.input)
        else:
            result = preprocess(args.input, save_steps=args.steps, output_dir=args.output)
        Path(args.output).mkdir(parents=True, exist_ok=True)
        out = Path(args.output) / Path(args.input).name
        cv2.imwrite(str(out), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"\nบันทึกแล้ว → {out}")

    elif args.batch:
        # Batch mode
        if args.mode == "vlm":
            process_batch_vlm(args.batch, args.output)
        else:
            process_batch(args.batch, args.output)

    else:
        # Demo mode
        run_demo()
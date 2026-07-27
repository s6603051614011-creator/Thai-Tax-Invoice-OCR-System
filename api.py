"""
api.py -- FastAPI inference endpoint สำหรับ Thai Tax Invoice OCR
================================================================
รับรูปใบกำกับภาษี -> รัน fine-tuned model -> คืน fields เป็น JSON

ติดตั้งเพิ่ม:
  pip install fastapi "uvicorn[standard]" python-multipart

รัน:
  python api.py
  หรือ:  uvicorn api:app --host 0.0.0.0 --port 8000

ทดสอบ:
  เปิด http://localhost:8000/docs  (Swagger UI ลองอัปโหลดรูปได้เลย)
  หรือ:  curl -F "file=@dataset/raw/inv_001.jpg" http://localhost:8000/ocr

หมายเหตุ:
- โหลดโมเดลแบบ 4-bit ครั้งเดียวตอน startup (เหมือนตอน train -> VRAM พอบน 6GB)
- ใช้ prompt + base model ชุดเดียวกับตอน train (จาก schema.py) = ผลตรง เสถียร
- ถ้า training ใช้รูปที่ผ่าน preprocess.py ควรเปิด APPLY_PREPROCESS=1 ให้ตรงกัน
"""

import os

# ลด VRAM fragmentation ต้องตั้งก่อน import torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import io
import time
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from PIL import Image, ImageOps

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

from transformers import BitsAndBytesConfig
from peft import PeftModel

import schema  # single source of truth: BASE_MODEL_ID + prompt + parser
import postprocess
from master_list import build_master_live, apply_master, save_verified

# model/processor class เลือกอัตโนมัติตาม BASE_MODEL_ID (Qwen2.5-VL หรือ Qwen3-VL)
ModelClass     = schema.get_model_class()
ProcessorClass = schema.get_processor_class()

# ── Config ───────────────────────────────────────────────
ADAPTER_DIR     = os.environ.get("ADAPTER_DIR", "models/best_model")
HF_TOKEN        = os.environ.get("HF_TOKEN") or None
MAX_NEW_TOKENS  = 768
MIN_PIXELS      = 64 * 32 * 32     # ต้องตรงกับตอน train (Finetune.py) -- Qwen3-VL 32px/token
# ── ความละเอียด: 960 tokens (ไม่ใช่ 2048 เดิม) ──
# เดิมตั้ง 2048 เพราะทดสอบบน test set 50 ใบ (รูป landscape/สแกน) แล้วดัน Field
# Accuracy จาก 42.8% -> 73% โดยไม่ล้น VRAM แต่พอทดสอบจากมือถือจริง (รูปแนวตั้ง
# ความละเอียดสูงแบบกล้องมือถือ) เจอ VRAM ไม่พอ -- วัดจริงพบว่ารูปแนวตั้งได้
# image token มากกว่ารูปแนวนอนที่ MAX_PIXELS เดียวกัน (patch-rounding ต่างกันตาม
# สัดส่วนภาพ) เพราะ dataset ทดสอบทุกใบดัน resize แล้วได้ token น้อยกว่าเพดานเสมอ
# เราจึงไม่เคยเจอ worst-case นี้มาก่อน วัดซ้ำด้วยรูป worst-case (แนวตั้งจำลอง
# กล้องมือถือ 3024x4032) พบว่า 2048 OOM จริง, 1280 เฉียดเต็มการ์ด (6.09GB),
# มีแค่ 960 ลงมาที่ปลอดภัยจริง (4.09GB เผื่อพอ) -- แลกความแม่นยำเหลือ ~64-68%
# (ยังดีกว่าที่เทรนไว้เดิม 42.8% มาก) เพื่อความเสถียรกับรูปจริงทุกแบบ
MAX_PIXELS      = 960 * 32 * 32
# ย่อรูปให้ด้านยาวสุดไม่เกินนี้ก่อนเข้าโมเดล (ชั้นป้องกันที่ 2) -- ควบคุมขนาด
# ด้วยมิติจริงแทนที่จะปล่อยให้ MAX_PIXELS คำนวณเองอย่างเดียว กัน edge case ที่
# สุดโต่งกว่าที่เคยทดสอบ (เช่นกล้องมือถือรุ่นใหม่ 48MP+) ไม่ให้ผลต่างกันตาม
# สัดส่วนภาพเหมือนที่เจอมา
MAX_IMAGE_SIDE  = 1600
MAX_UPLOAD_MB   = 15
APPLY_PREPROCESS = os.environ.get("APPLY_PREPROCESS", "0") == "1"
STATIC_DIR      = Path(__file__).parent / "static"
# ─────────────────────────────────────────────────────────

# global model state
STATE = {"model": None, "processor": None, "device": None, "loaded": False, "error": None,
         "master_seller": None, "master_buyer": None}


def load_model():
    """โหลด base 4-bit + LoRA adapter ครั้งเดียว"""
    if not torch.cuda.is_available():
        raise RuntimeError("ไม่พบ GPU/CUDA — inference โมเดลนี้ต้องใช้ CUDA")

    print(f"โหลด base: {schema.BASE_MODEL_ID} (4-bit)...", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = ModelClass.from_pretrained(
        schema.BASE_MODEL_ID,
        device_map={"": 0},
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        token=HF_TOKEN,
    )

    if os.path.isdir(ADAPTER_DIR):
        print(f"โหลด LoRA adapter: {ADAPTER_DIR}", flush=True)
        os.makedirs("offload_tmp", exist_ok=True)
        model = PeftModel.from_pretrained(
            base, ADAPTER_DIR, offload_folder="offload_tmp", offload_buffers=True,
        )
        proc_src = ADAPTER_DIR
    else:
        print(f"ไม่พบ adapter ที่ {ADAPTER_DIR} -> ใช้ base model เปล่า (baseline)", flush=True)
        model = base
        proc_src = schema.BASE_MODEL_ID

    model.eval()
    processor = ProcessorClass.from_pretrained(
        proc_src, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
        trust_remote_code=True, token=HF_TOKEN,
    )

    # build_master_live() อ่านจาก master_db (SQLite) -- seed ครั้งแรกจาก train
    # เท่านั้น (กัน data leakage เหมือนเดิม) แต่หลังจากนั้นโตขึ้นเรื่อยๆ จากพนักงาน
    # กดยืนยัน/แก้ไขข้อมูลจริงผ่าน POST /confirm (ต่างจาก build_master() ที่ eval
    # script ใช้ ซึ่งอ่านจากไฟล์ train ตรงๆ ทุกครั้ง ไม่โต เพื่อให้ผลวัด reproduce ได้)
    print("โหลดตารางอ้างอิงคู่ค้า (seed จาก train + ข้อมูลที่ยืนยันแล้วจากการใช้งานจริง)...", flush=True)
    master_seller = build_master_live("seller")
    master_buyer  = build_master_live("buyer")
    print(f"   ผู้ขาย {len(master_seller)} ราย, ผู้ซื้อ {len(master_buyer)} ราย", flush=True)

    STATE.update(model=model, processor=processor, device="cuda", loaded=True,
                master_seller=master_seller, master_buyer=master_buyer)
    print("โมเดลพร้อมใช้งาน", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception as e:
        STATE["error"] = str(e)
        print(f"โหลดโมเดลไม่สำเร็จ: {e}", flush=True)
    yield
    # cleanup
    STATE["model"] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="Thai Tax Invoice OCR API", version="1.0", lifespan=lifespan)

# เปิด CORS ให้ React frontend เรียกได้ (ปรับ origins ตอน deploy จริง)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _cap_image_size(img: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    """ย่อรูปให้ด้านยาวสุดไม่เกิน max_side (คงสัดส่วนเดิม) -- ป้องกัน VRAM ล้น
    จากรูปความละเอียดสูงมาก (กล้องมือถือ 12MP+) ก่อนเข้า image processor ของโมเดล
    ทำก่อนเสมอไม่ว่าจะเปิด APPLY_PREPROCESS หรือไม่"""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def _maybe_preprocess(img: Image.Image) -> Image.Image:
    """ถ้าเปิด APPLY_PREPROCESS -> ใช้ pipeline เดียวกับตอนทำ dataset"""
    if not APPLY_PREPROCESS:
        return img
    try:
        import preprocess  # ใช้ฟังก์ชันจาก preprocess.py ถ้ามี
        for fn_name in ("preprocess_image", "process_image", "run"):
            fn = getattr(preprocess, fn_name, None)
            if callable(fn):
                return fn(img)
    except Exception as e:
        print(f"preprocess ข้าม ({e})", flush=True)
    return img


@torch.no_grad()
def run_ocr(img: Image.Image) -> dict:
    model, processor = STATE["model"], STATE["processor"]
    messages = schema.build_inference_messages(img)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)

    # ต้องตรงกับ evaluate.py::run_inference เป๊ะ (ตัวที่ใช้วัด Field Accuracy 73.3%)
    # -- เดิมมี repetition_penalty/no_repeat_ngram_size เพิ่มมาจาก Finetune.py
    # (ทดสอบแค่ 1 ใบตอนนั้น) แต่ JSON ที่มีหลายรายการสินค้าจำเป็นต้อง "พูดซ้ำ" คีย์
    # เดิม (unit_price/discount/amount) no_repeat_ngram_size=5 ไปห้ามการพูดซ้ำนั้น
    # ทำให้โมเดลเบี่ยงไปสร้างคีย์ผิดจน JSON พังทุกใบที่มี >=2 รายการ (พบจากทดสอบจริง)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,                 # deterministic -> ผลซ้ำได้
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    raw = processor.tokenizer.decode(
        out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )
    fields = schema.parse_model_json(raw)
    return {"fields": fields, "raw": raw}


# ══════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "ok" if STATE["loaded"] else "model_not_loaded",
        "model_loaded": STATE["loaded"],
        "base_model": schema.BASE_MODEL_ID,
        "adapter": ADAPTER_DIR if os.path.isdir(ADAPTER_DIR) else None,
        "cuda": torch.cuda.is_available(),
        "error": STATE["error"],
    }


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    if not STATE["loaded"]:
        raise HTTPException(503, detail=f"โมเดลยังไม่พร้อม: {STATE['error'] or 'กำลังโหลด'}")

    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, detail=f"ไฟล์ใหญ่เกิน {MAX_UPLOAD_MB}MB")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception:
        raise HTTPException(400, detail="เปิดรูปไม่ได้ — ต้องเป็นไฟล์ภาพที่ถูกต้อง")

    img = _cap_image_size(img)
    img = _maybe_preprocess(img)

    t0 = time.time()
    try:
        result = run_ocr(img)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(507, detail="VRAM ไม่พอ — ลองลดขนาดรูปหรือ MAX_PIXELS")
    except Exception as e:
        raise HTTPException(500, detail=f"inference error: {e}")

    # schema.parse_model_json คืนค่าว่างเงียบๆ ถ้าโมเดลตอบ JSON ผิดรูปแบบ (เกิดได้
    # ~0-2% ของใบ ตามที่วัดไว้) -- เช็คแยกก่อนเอาไปใช้ต่อ กันฟอร์มว่างเปล่าไม่บอกอะไร
    json_ok = bool(result["fields"] and any(
        result["fields"].get(k) for k in schema.SCALAR_FIELDS))

    if not json_ok:
        return JSONResponse({
            "filename": file.filename,
            "elapsed_sec": round(time.time() - t0, 2),
            "fields": result["fields"],
            "canonical_json": schema.build_canonical_json(result["fields"]),
            "raw_output": result["raw"],
            "flags": ["โมเดลตอบไม่ถูกรูปแบบ JSON -- ลองถ่ายใหม่ให้ชัดขึ้น/ตรงขึ้น หรือกรอกด้วยมือ"],
            "master_list_fixes": [],
            "master_list_match": {},
        })

    # เติม/แก้ tax_id + address จากตารางอ้างอิงคู่ค้า (เชื่อโมเดลก่อนเสมอ
    # -- ดูนโยบายระมัดระวังใน master_list.apply_master)
    fields, ml_changes, master_match = apply_master(
        result["fields"], STATE["master_seller"], STATE["master_buyer"])

    # ตรวจความสมเหตุสมผล (checksum เลขภาษี + เลขคณิตใบกำกับ) ไม่ต้องมีเฉลย
    flags = postprocess.validate_fields(fields)

    # แจ้งเตือนกรณีค่าที่โมเดลอ่านได้ต่างจากที่เคยบันทึกในตารางอ้างอิง (type="notice"
    # ใน apply_master) -- ไม่ทับค่า แค่บอกผู้ใช้ให้เช็คทานเป็นพิเศษ
    for ch in ml_changes:
        if ch.get("type") == "notice":
            flags.append(
                f"{schema.LABELS.get(ch['field'], ch['field'])} ของ \"{ch['matched_name']}\" "
                f"ที่อ่านได้ ({ch['model_value']}) ต่างจากที่เคยบันทึกไว้ ({ch['master_value']}) "
                f"-- กรุณาตรวจสอบ"
            )

    return JSONResponse({
        "filename": file.filename,
        "elapsed_sec": round(time.time() - t0, 2),
        "fields": fields,
        "canonical_json": schema.build_canonical_json(fields),
        "raw_output": result["raw"],
        "flags": flags,
        "master_list_fixes": ml_changes,
        # ผลเทียบกับฐานข้อมูลของทุกช่อง (ไม่ใช่แค่ตอนมีปัญหา) -- ให้ frontend
        # โชว์ % ความมั่นใจต่อผู้ใช้ได้ เช่น "ชื่อผู้ขาย ตรงกับฐานข้อมูล 100%"
        "master_list_match": master_match,
    })


class ConfirmRequest(BaseModel):
    seller_name_th: str = ""
    seller_name_en: str = ""
    seller_tax_id: str = ""
    seller_address: str = ""
    buyer_name_th: str = ""
    buyer_name_en: str = ""
    buyer_tax_id: str = ""
    buyer_address: str = ""


@app.post("/confirm")
def confirm(body: ConfirmRequest):
    """บันทึกข้อมูลคู่ค้า (ชื่อ/ที่อยู่/เลขภาษี เท่านั้น) ที่ผู้ใช้ตรวจ/แก้ไขแล้วลง
    master_db ถาวร -- ใช้ได้ทั้งบริษัทใหม่ (เพิ่มรายการ) และบริษัทเก่า (แก้ให้ตรงปัจจุบัน)
    เรียกจากปุ่ม "บันทึก" ต่อใบในหน้าเว็บ หลังผู้ใช้ตรวจทานเสร็จ"""
    if not STATE["loaded"]:
        raise HTTPException(503, detail="โมเดลยังไม่พร้อม")

    saved = []
    for side in ("seller", "buyer"):
        name_th = getattr(body, f"{side}_name_th").strip()
        if not name_th:
            continue
        save_verified(side, name_th, getattr(body, f"{side}_name_en"),
                     getattr(body, f"{side}_tax_id"), getattr(body, f"{side}_address"))
        saved.append(side)

    # รีเฟรชตารางในหน่วยความจำทันที -- ไม่ต้อง restart server ก็เห็นผลตั้งแต่ใบถัดไป
    STATE["master_seller"] = build_master_live("seller")
    STATE["master_buyer"]  = build_master_live("buyer")

    return {"saved": saved}


@app.get("/meta")
def meta():
    """field list + label ภาษาไทย -- ให้หน้าเว็บ frontend ดึงไปสร้างฟอร์มโดยไม่ต้อง
    เขียนชื่อ field ซ้ำใน JS (schema.py เป็นแหล่งความจริงเดียวเหมือนเดิม)"""
    return {
        "scalar_fields": [{"key": k, "label": schema.LABELS.get(k, k)} for k in schema.SCALAR_FIELDS],
        "item_fields":   [{"key": k, "label": schema.LABELS.get(k, k)} for k in schema.ITEM_FIELDS],
        "summary_fields":[{"key": k, "label": schema.LABELS.get(k, k)} for k in schema.SUMMARY_FIELDS],
    }


@app.get("/", response_class=HTMLResponse)
def mobile_page():
    """หน้าเว็บสำหรับมือถือ -- ถ่าย/เลือกรูปหลายใบ อัปโหลดทีเดียว แสดงผลทีละใบ"""
    html_path = STATIC_DIR / "mobile.html"
    if not html_path.exists():
        return HTMLResponse("<h1>ไม่พบ static/mobile.html</h1>", status_code=500)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import socket
    import uvicorn
    try:
        # gethostbyname(gethostname()) มักได้ IP ของ VPN/virtual adapter (เช่น Radmin,
        # Hamachi) แทนที่จะเป็น WiFi จริง -- เปิด UDP socket เชื่อมต่อออกนอกเครื่อง
        # (ไม่ส่งข้อมูลจริง) เพื่อดูว่า route ผ่าน network interface ไหน ได้ IP ที่ถูกต้องกว่า
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            lan_ip = s.getsockname()[0]
    except Exception:
        lan_ip = "<ip เครื่องนี้ -- เช็คด้วย ipconfig, มองหา Wireless LAN adapter Wi-Fi>"
    print("\nเปิด API แล้ว:", flush=True)
    print(f"  ในเครื่องนี้      : http://localhost:8000", flush=True)
    print(f"  จากมือถือ (WiFi วงเดียวกัน) : http://{lan_ip}:8000", flush=True)
    print(f"  Swagger (ทดสอบ)  : http://localhost:8000/docs\n", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)

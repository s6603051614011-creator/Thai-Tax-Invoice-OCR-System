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
from contextlib import asynccontextmanager

import torch
from PIL import Image, ImageOps

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, BitsAndBytesConfig
from peft import PeftModel

import schema  # single source of truth: BASE_MODEL_ID + prompt + parser

# ── Config ───────────────────────────────────────────────
ADAPTER_DIR     = os.environ.get("ADAPTER_DIR", "models/best_model")
HF_TOKEN        = os.environ.get("HF_TOKEN") or None
MAX_NEW_TOKENS  = 768
MIN_PIXELS      = 128 * 28 * 28   # ต้องตรงกับตอน train (Finetune.py)
MAX_PIXELS      = 256 * 28 * 28
MAX_UPLOAD_MB   = 15
APPLY_PREPROCESS = os.environ.get("APPLY_PREPROCESS", "0") == "1"
# ─────────────────────────────────────────────────────────

# global model state
STATE = {"model": None, "processor": None, "device": None, "loaded": False, "error": None}


def load_model():
    """โหลด base 4-bit + LoRA adapter ครั้งเดียว"""
    if not torch.cuda.is_available():
        raise RuntimeError("ไม่พบ GPU/CUDA — inference โมเดลนี้ต้องใช้ CUDA")

    print(f"📦 โหลด base: {schema.BASE_MODEL_ID} (4-bit)...", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        schema.BASE_MODEL_ID,
        device_map={"": 0},
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        token=HF_TOKEN,
    )

    if os.path.isdir(ADAPTER_DIR):
        print(f"🔧 โหลด LoRA adapter: {ADAPTER_DIR}", flush=True)
        os.makedirs("offload_tmp", exist_ok=True)
        model = PeftModel.from_pretrained(
            base, ADAPTER_DIR, offload_folder="offload_tmp", offload_buffers=True,
        )
        proc_src = ADAPTER_DIR
    else:
        print(f"⚠️  ไม่พบ adapter ที่ {ADAPTER_DIR} -> ใช้ base model เปล่า (baseline)", flush=True)
        model = base
        proc_src = schema.BASE_MODEL_ID

    model.eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(
        proc_src, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
        trust_remote_code=True, token=HF_TOKEN,
    )
    STATE.update(model=model, processor=processor, device="cuda", loaded=True)
    print("✅ โมเดลพร้อมใช้งาน", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception as e:
        STATE["error"] = str(e)
        print(f"❌ โหลดโมเดลไม่สำเร็จ: {e}", flush=True)
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
        print(f"⚠️  preprocess ข้าม ({e})", flush=True)
    return img


@torch.no_grad()
def run_ocr(img: Image.Image) -> dict:
    model, processor = STATE["model"], STATE["processor"]
    messages = schema.build_inference_messages(img)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,                 # deterministic -> ผลซ้ำได้
        repetition_penalty=1.3,
        no_repeat_ngram_size=5,
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

    img = _maybe_preprocess(img)

    t0 = time.time()
    try:
        result = run_ocr(img)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(507, detail="VRAM ไม่พอ — ลองลดขนาดรูปหรือ MAX_PIXELS")
    except Exception as e:
        raise HTTPException(500, detail=f"inference error: {e}")

    return JSONResponse({
        "filename": file.filename,
        "elapsed_sec": round(time.time() - t0, 2),
        "fields": result["fields"],
        "canonical_json": schema.build_canonical_json(result["fields"]),
        "raw_output": result["raw"],
    })


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 เปิด API ที่ http://localhost:8000  (Swagger: /docs)\n", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)

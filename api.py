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
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from PIL import Image, ImageOps

from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from transformers import BitsAndBytesConfig
from peft import PeftModel

import schema  # single source of truth: BASE_MODEL_ID + prompt + parser
import postprocess
import store    # ฐานข้อมูล SQLite เก็บใบที่ตรวจแล้ว + สรุปยอดรายเดือน/รายปี
from master_list import build_master_live, apply_master, save_verified

# model/processor class เลือกอัตโนมัติตาม BASE_MODEL_ID (Qwen2.5-VL หรือ Qwen3-VL)
ModelClass     = schema.get_model_class()
ProcessorClass = schema.get_processor_class()

# ── Config ───────────────────────────────────────────────
ADAPTER_DIR     = os.environ.get("ADAPTER_DIR", "models/best_model")
HF_TOKEN        = os.environ.get("HF_TOKEN") or None
# ── เพดานความยาวคำตอบ: 1536 (ไม่ใช่ 768 เดิม) ──
# 768 ทำให้ใบที่มีรายการสินค้าเยอะ output ถูกตัดกลางคัน -> JSON ปิดวงเล็บไม่ครบ ->
# parse ไม่ผ่าน -> schema.parse_model_json คืนค่าว่างทุกช่อง = เสียทั้งใบ
# วัดจริงกับ 2 ใบที่พังใน test set (inv_095 11 รายการ, inv_272 7 รายการ):
#   @768  -> generate ครบ 768 ชนเพดานพอดี, JSON พัง, ได้ 0/10 ช่อง ทั้งคู่
#   @1536 -> generate 860/805 แล้วจบเอง, JSON ผ่าน, ได้ 9/10 และ 8/10 ช่อง
# ขยายเพดานไม่มีต้นทุน: โมเดลหยุดเองที่ EOS อยู่แล้ว ใบปกติ (~200-400 tokens)
# จึงไม่ช้าลงเลย และ KV cache ที่เพิ่มมาเป็นหลัก MB บนโมเดล 2B
MAX_NEW_TOKENS  = 1536
MIN_PIXELS      = 64 * 32 * 32     # ต้องตรงกับตอน train (Finetune.py) -- Qwen3-VL 32px/token
# ── ความละเอียด: 1600 tokens (ปรับขึ้นจาก 960 หลังเปลี่ยนเป็น sdpa) ──
# ตอนยังใช้ eager, VRAM ของรูป worst-case (แนวตั้งจำลองกล้องมือถือ 3024x4032)
# ที่ 960 tok กินไป 4.09GB และ 1280 tok ขึ้นไปเฉียด/ล้นการ์ด 6GB -- เลยต้องจำกัด
# ไว้แค่ 960 (แลกความแม่นยำเหลือ ~64-68%) พอเปลี่ยนเป็น sdpa (ดู attn_implementation
# ด้านล่าง) VRAM ลดฮวบมาก วัดซ้ำด้วยรูป worst-case เดียวกันพบว่า "ปลอดภัยหมดทุกค่า
# ที่ลองจนถึง 3072 tok" (reserved สูงสุดแค่ 4.58GB ที่ 3072tok) เลือก 1600 เพราะ
# ความแม่นยำเริ่ม plateau เกินจุดนี้ (ข้อจำกัดจากความละเอียดตอนเทรน 320 tok ไม่ใช่
# VRAM แล้ว) วัดจริงบน test set 50 ใบ (โค้ดปัจจุบัน: sdpa + MAX_NEW_TOKENS=1536):
#   960 tok  + Master List = 69.9% (ที่เคยใช้)
#   1600 tok + Master List = 73.8% (ตัวใหม่ที่ใช้อยู่นี้, CER 8.0%, JSON พัง 0/50)
MAX_PIXELS      = 1600 * 32 * 32
# ย่อรูปให้ด้านยาวสุดไม่เกินนี้ก่อนเข้าโมเดล (ชั้นป้องกันที่ 2) -- ควบคุมขนาด
# ด้วยมิติจริงแทนที่จะปล่อยให้ MAX_PIXELS คำนวณเองอย่างเดียว กัน edge case ที่
# สุดโต่งกว่าที่เคยทดสอบ (เช่นกล้องมือถือรุ่นใหม่ 48MP+) ไม่ให้ผลต่างกันตาม
# สัดส่วนภาพเหมือนที่เจอมา
MAX_IMAGE_SIDE  = 1600
MAX_UPLOAD_MB   = 15
APPLY_PREPROCESS = os.environ.get("APPLY_PREPROCESS", "0") == "1"
STATIC_DIR      = Path(__file__).parent / "static"
# API key -- ไม่บังคับตอนอยู่ใน LAN (ค่าเริ่มต้น ไม่ตั้ง env var ก็ไม่ต้องใส่ key
# เลย พฤติกรรมเดิมทุกอย่าง) แต่ต้องตั้งก่อนเปิดออกนอก LAN เสมอ ไม่งั้นใครก็เรียก
# /ocr (ใช้ GPU ฟรี), /invoices, /confirm (ยัดข้อมูลปลอมเข้าฐานข้อมูล) ได้หมด
# รันแบบมี auth:  API_KEY=xxxxx python api.py
API_KEY         = os.environ.get("API_KEY") or None
# ─────────────────────────────────────────────────────────

# global model state
STATE = {"model": None, "processor": None, "device": None, "loaded": False, "error": None,
         "master_seller": None, "master_buyer": None}

# ══════════════════════════════════════════════
# คิวใช้ GPU -- อ่านได้ทีละใบเท่านั้น
# ══════════════════════════════════════════════
# การ์ดมี VRAM 6GB, อ่าน 1 ใบใช้ ~4GB -> สองใบพร้อมกันล้นแน่นอน
#
# ก่อนหน้านี้ระบบ "รอด" มาได้เพราะบังเอิญ: route เป็น async def แต่ run_ocr()
# เป็นฟังก์ชัน blocking ธรรมดา -> มันบล็อก event loop ทั้งเส้นตอนทำงาน คำขอที่ 2
# จึงเข้ามาไม่ได้เลย ผลข้างเคียงคือ /health ก็ตอบไม่ได้ตลอด 40-140 วิด้วย และถ้า
# ใครแก้ async def -> def (ซึ่งถูกตามหลัก FastAPI สำหรับงาน blocking) threadpool
# จะรันขนานกันทันที = VRAM ล้น โดยไม่มีอะไรในโค้ดเตือนไว้เลย
#
# จึงเขียนเจตนาลงไปตรงๆ: semaphore ใบเดียว + ย้าย inference ไป threadpool
# -> กัน GPU ชนกันแบบตั้งใจ, event loop ว่างตอบ /health ได้, และบอกความยาวคิวได้
GPU_SEMAPHORE = asyncio.Semaphore(1)
QUEUE = {"waiting": 0, "busy": False}


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
    # ต้องตรงกับ evaluate.py::load_model() เป๊ะ -- ที่นั่นไม่ได้ระบุ attn_implementation
    # เลย ปล่อยให้ transformers auto-select ซึ่งวัดแล้วมันเลือก "sdpa" เสมอสำหรับ
    # โมเดลนี้ ในขณะที่ไฟล์นี้เคยล็อกไว้ที่ "eager" ตรงๆ -- ตัวเลข Field Accuracy
    # ทุกตัวที่รายงานอาจารย์ไป (69.9% ที่ 960 tok, ก่อนหน้านี้ก็เช่นกัน) วัดด้วย
    # sdpa มาตลอด แต่ api.py (ระบบที่ใช้งานจริง) กลับใช้ eager มาตลอดโดยไม่ตั้งใจ
    # -- เพิ่งพบจาก benchmark เทียบ 5 ใบ: คำตอบต่างกันจริง ไม่ใช่แค่เร็วขึ้น (เช่น
    # inv_095 eager อ่านชื่อบริษัทผิดเป็นคนละบริษัท sdpa อ่านถูก) เปลี่ยนเป็น sdpa
    # ให้ production ตรงกับตัวเลขที่วัดไว้จริงๆ เสียที + ได้เร็วขึ้น ~2 เท่า และ
    # VRAM เหลือเยอะขึ้น (~2.1GB จาก ~3.7GB ที่ 960 tok) เป็นผลพลอยได้
    base = ModelClass.from_pretrained(
        schema.BASE_MODEL_ID,
        device_map={"": 0},
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
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

    store.init()      # ฐานข้อมูลใบที่ตรวจแล้ว (คนละตัวกับ master_db ที่เก็บตารางคู่ค้า)
    print(f"ฐานข้อมูลใบกำกับ: {store.stats()['db']} ({store.stats()['invoices']} ใบที่บันทึกไว้)", flush=True)

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


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """ป้องกัน endpoint ที่ใช้ GPU หรือแตะฐานข้อมูลจริง -- เช็คเฉพาะตอนตั้ง API_KEY
    ไว้ (เช่นตอนเปิดออกนอก LAN) ถ้าไม่ได้ตั้ง (ค่าเริ่มต้นตอนใช้ใน LAN) ผ่านตลอด
    ไม่ต้องมี key เลย ไม่กระทบพฤติกรรมเดิมแม้แต่นิดเดียว
    ไม่ครอบ /health, /meta, /, /summary เพราะไม่ใช่ข้อมูลอ่อนไหว (แค่ label
    ฟิลด์/หน้า HTML เปล่า) และ /health ควรเช็คได้เสมอไม่ว่าจะมี key หรือไม่"""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, detail="ต้องใส่ API key ที่ถูกต้อง (header X-API-Key)")


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
        # สถานะคิว -- ตอบได้ตลอดแม้กำลังอ่านใบอยู่ (inference ไปอยู่ threadpool แล้ว)
        "busy": QUEUE["busy"],
        "queue_waiting": QUEUE["waiting"],
    }


@app.post("/ocr", dependencies=[Depends(require_api_key)])
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

    # ── เข้าคิวใช้ GPU (ทีละใบ) ──
    # run_in_threadpool ย้าย inference ที่ blocking ออกจาก event loop -> /health และ
    # คำขออื่นยังตอบได้ระหว่างอ่าน ส่วน semaphore เป็นตัวกันไม่ให้ชนกันบน GPU
    t_queued = time.time()
    QUEUE["waiting"] += 1
    still_waiting = True           # ยังติดหนี้ลดตัวนับอยู่ไหม (กันลดซ้ำ/ลดของคนอื่น)
    try:
        async with GPU_SEMAPHORE:
            QUEUE["waiting"] -= 1
            still_waiting = False
            QUEUE["busy"] = True
            t0 = time.time()
            try:
                result = await run_in_threadpool(run_ocr, img)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise HTTPException(507, detail="VRAM ไม่พอ — ลองลดขนาดรูปหรือ MAX_PIXELS")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(500, detail=f"inference error: {e}")
            finally:
                QUEUE["busy"] = False
    finally:
        # หลุดออกไปก่อนได้คิว (client ตัดการเชื่อมต่อระหว่างรอ) ก็ต้องคืนตัวนับ
        # ไม่งั้น waiting ค้างเพิ่มขึ้นเรื่อยๆ จนเลขคิวที่โชว์ผู้ใช้เพี้ยน
        if still_waiting:
            QUEUE["waiting"] -= 1

    queued_sec = round(t0 - t_queued, 2)

    # schema.parse_model_json คืนค่าว่างเงียบๆ ถ้าโมเดลตอบ JSON ผิดรูปแบบ (เกิดได้
    # ~0-2% ของใบ ตามที่วัดไว้) -- เช็คแยกก่อนเอาไปใช้ต่อ กันฟอร์มว่างเปล่าไม่บอกอะไร
    json_ok = bool(result["fields"] and any(
        result["fields"].get(k) for k in schema.SCALAR_FIELDS))

    if not json_ok:
        return JSONResponse({
            "filename": file.filename,
            "elapsed_sec": round(time.time() - t0, 2),
            "queued_sec": queued_sec,
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
        # เวลาที่รอคิว GPU ก่อนได้เริ่มอ่านจริง (0 = ได้คิวทันที) -- แยกจาก
        # elapsed_sec เพื่อให้ frontend บอกผู้ใช้ได้ว่าช้าเพราะรอคิวหรือเพราะอ่านนาน
        "queued_sec": queued_sec,
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


@app.post("/confirm", dependencies=[Depends(require_api_key)])
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
        # ตารางสินค้าใช้ ITEM_LABELS (ไม่ใช่ LABELS) -- "discount" ระดับรายการ
        # คือส่วนลดของบรรทัดนั้น ไม่ใช่ "ส่วนลดรวม" ของทั้งใบ
        "item_fields":   [{"key": k, "label": schema.ITEM_LABELS.get(k, k)} for k in schema.ITEM_FIELDS],
        "summary_fields":[{"key": k, "label": schema.LABELS.get(k, k)} for k in schema.SUMMARY_FIELDS],
        # กลุ่ม field สำหรับจัดหน้าฟอร์ม -- frontend ไม่ต้องรู้ว่า field ไหนอยู่กลุ่มไหน
        "ui_groups":     [{"title": g["title"],
                           "fields": [{"key": k, "label": schema.LABELS.get(k, k)} for k in g["fields"]]}
                          for g in schema.UI_GROUPS],
    }


@app.get("/", response_class=HTMLResponse)
def mobile_page():
    """หน้าเว็บสำหรับมือถือ -- ถ่าย/เลือกรูปหลายใบ อัปโหลดทีเดียว แสดงผลทีละใบ"""
    return _page("mobile.html")


@app.get("/summary", response_class=HTMLResponse)
def summary_page():
    """หน้าสรุปยอดรายเดือน/รายปี (กราฟ) -- ดึงตัวเลขจาก /api/summary"""
    return _page("summary.html")


@app.get("/history", response_class=HTMLResponse)
def history_page():
    """หน้าประวัติ -- รายการใบที่บันทึกแล้วทั้งหมด ดูรูปจริง + แก้ไขย้อนหลังได้"""
    return _page("history.html")


def _page(name: str):
    p = STATIC_DIR / name
    if not p.exists():
        return HTMLResponse(f"<h1>ไม่พบ static/{name}</h1>", status_code=500)
    return HTMLResponse(p.read_text(encoding="utf-8"))


# ══════════════════════════════════════════════
# ฐานข้อมูล: บันทึกใบที่ตรวจแล้ว + สรุปยอด
# ══════════════════════════════════════════════

@app.post("/invoices", dependencies=[Depends(require_api_key)])
def save_invoice(payload: dict = Body(...)):
    """
    บันทึกใบที่ผู้ใช้ตรวจ/แก้แล้วลงฐานข้อมูล (กดปุ่ม "บันทึก" ในหน้าตรวจ)
    payload: {"filename":..., "fields": {...}, "items": [...], "flags": [...]}

    บันทึก 2 อย่างในการกดครั้งเดียว:
      1. ตัวใบ -> store (data/invoices.db)  เอาไปสรุปยอดรายเดือน/รายปี
      2. ข้อมูลคู่ค้า -> master_db ผ่าน save_verified()  ให้ตารางอ้างอิงโตขึ้น
    เพราะ "พนักงานยืนยันว่าใบนี้ถูกต้องแล้ว" ก็แปลว่าชื่อ/เลขภาษี/ที่อยู่คู่ค้า
    ในใบนั้นถูกต้องด้วย -- ไม่ต้องให้ frontend จำว่าต้องยิงสอง endpoint
    (ยังมี POST /confirm แยกไว้เหมือนเดิม สำหรับกรณีอยากยืนยันคู่ค้าอย่างเดียว)
    """
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        raise HTTPException(400, detail="payload ต้องมี fields")

    try:
        result = store.save_invoice(payload)
    except Exception as e:
        raise HTTPException(500, detail=f"บันทึกไม่สำเร็จ: {e}")

    # ตารางคู่ค้าเป็นงานเสริม -- ถ้าพลาดต้องไม่ทำให้ "บันทึกใบ" ที่สำเร็จไปแล้วกลายเป็น error
    learned = []
    try:
        for side in ("seller", "buyer"):
            name_th = str(fields.get(f"{side}_name_th") or "").strip()
            if not name_th:
                continue
            save_verified(side, name_th,
                          str(fields.get(f"{side}_name_en") or ""),
                          str(fields.get(f"{side}_tax_id") or ""),
                          str(fields.get(f"{side}_address") or ""))
            learned.append(side)
        if learned and STATE["loaded"]:
            # รีเฟรชตารางในหน่วยความจำ -- ใบถัดไปได้ประโยชน์ทันที ไม่ต้อง restart
            STATE["master_seller"] = build_master_live("seller")
            STATE["master_buyer"] = build_master_live("buyer")
    except Exception as e:
        print(f"เตือน: บันทึกใบสำเร็จแล้ว แต่จำข้อมูลคู่ค้าไม่สำเร็จ: {e}", flush=True)
        return {**result, "learned": [], "learn_error": str(e)}

    return {**result, "learned": learned}


@app.delete("/api/invoices/{invoice_id}", dependencies=[Depends(require_api_key)])
def api_invoice_delete(invoice_id: int):
    """ลบใบที่บันทึกไว้ถาวร -- จากปุ่ม "ลบ" ในหน้าประวัติ ไม่มี undo"""
    if not store.delete_invoice(invoice_id):
        raise HTTPException(404, detail=f"ไม่พบใบเลขที่ {invoice_id}")
    return {"deleted": invoice_id}


@app.get("/api/summary", dependencies=[Depends(require_api_key)])
def api_summary():
    """ยอดรวมรายเดือน + รายปี สำหรับหน้ากราฟ"""
    return store.summary()


@app.get("/api/invoices", dependencies=[Depends(require_api_key)])
def api_invoices(limit: int = 500):
    """รายการใบที่บันทึกไว้ (ล่าสุดก่อน) -- ใช้กับหน้าประวัติ (ตัวย่อ ไม่รวมรูป/รายการสินค้า)"""
    return {"invoices": store.list_invoices(limit)}


@app.get("/api/invoices/{invoice_id}", dependencies=[Depends(require_api_key)])
def api_invoice_detail(invoice_id: int):
    """ใบเดียวแบบเต็ม (ทุกช่อง + รายการสินค้า) -- ให้หน้าประวัติเปิดแก้ไข"""
    rec = store.get_invoice(invoice_id)
    if rec is None:
        raise HTTPException(404, detail=f"ไม่พบใบเลขที่ {invoice_id}")
    return rec


@app.get("/api/invoices/{invoice_id}/image", dependencies=[Depends(require_api_key)])
def api_invoice_image(invoice_id: int):
    """รูปใบจริงที่บันทึกไว้ตอนกดยืนยัน -- ใบที่บันทึกก่อนมีหน้าประวัติจะไม่มีรูป (404)"""
    data = store.get_invoice_image(invoice_id)
    if data is None:
        raise HTTPException(404, detail="ใบนี้ไม่มีรูปเก็บไว้ (บันทึกไว้ก่อนมีหน้าประวัติ)")
    return Response(content=data, media_type="image/jpeg")


@app.put("/api/invoices/{invoice_id}", dependencies=[Depends(require_api_key)])
def api_invoice_update(invoice_id: int, payload: dict = Body(...)):
    """แก้ไขใบที่บันทึกไว้แล้วจากหน้าประวัติ (แก้ในแถวเดิม ไม่สร้างใบใหม่)"""
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        raise HTTPException(400, detail="payload ต้องมี fields")
    try:
        result = store.update_invoice(invoice_id, payload)
    except KeyError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"บันทึกไม่สำเร็จ: {e}")

    # เหมือน POST /invoices -- ถือว่าผู้ใช้ตรวจทานแล้วว่าถูกต้อง ให้ตารางอ้างอิงคู่ค้าโตขึ้นด้วย
    learned = []
    try:
        for side in ("seller", "buyer"):
            name_th = str(fields.get(f"{side}_name_th") or "").strip()
            if not name_th:
                continue
            save_verified(side, name_th,
                          str(fields.get(f"{side}_name_en") or ""),
                          str(fields.get(f"{side}_tax_id") or ""),
                          str(fields.get(f"{side}_address") or ""))
            learned.append(side)
        if learned and STATE["loaded"]:
            STATE["master_seller"] = build_master_live("seller")
            STATE["master_buyer"] = build_master_live("buyer")
    except Exception as e:
        print(f"เตือน: แก้ไขใบสำเร็จแล้ว แต่จำข้อมูลคู่ค้าไม่สำเร็จ: {e}", flush=True)
        return {**result, "learned": [], "learn_error": str(e)}

    return {**result, "learned": learned}


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
    print(f"  Swagger (ทดสอบ)  : http://localhost:8000/docs", flush=True)
    if API_KEY:
        print(f"  API key          : เปิดใช้งาน (ต้องแนบ header X-API-Key ทุกครั้งที่เรียก "
             f"/ocr, /confirm, /invoices, /api/summary, /api/invoices)\n", flush=True)
    else:
        print(f"  API key          : ไม่ได้ตั้งไว้ -- ใครก็เรียกได้ ใช้ได้เฉพาะใน LAN "
             f"เท่านั้น! ตั้ง API_KEY=xxxxx ก่อน python api.py หากจะเปิดออกนอก LAN\n", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
Step 3: QLoRA Fine-tune Typhoon OCR สำหรับใบกำกับภาษีไทย
==========================================================
ปรับแต่งสำหรับ: RTX 4050 Laptop 6GB VRAM | RAM 16GB | Windows 11

ติดตั้ง:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  pip install transformers>=4.49.0 accelerate peft==0.14.0 bitsandbytes>=0.46.1
  pip install qwen-vl-utils trl==0.15.0 datasets pillow tqdm

รัน:
  python Finetune.py --mode train
  python Finetune.py --mode test --image dataset/raw/inv_001.jpg
"""

import os

# ── ลด VRAM fragmentation (สำคัญมากบน 6GB) ต้องตั้งก่อน import torch ──
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import json
import base64
import gc
import torch

import schema  # single source of truth: BASE_MODEL_ID + prompt
from pathlib import Path
from dataclasses import dataclass
from io import BytesIO

print("✓ stdlib imported", flush=True)

from PIL import Image, ImageOps
from tqdm import tqdm

print("✓ PIL imported", flush=True)

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    set_seed,
)
print("✓ transformers imported", flush=True)

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
print("✓ peft imported", flush=True)

from torch.utils.data import Dataset
print("✓ all imports done", flush=True)


# ══════════════════════════════════════════════
# CONFIG — ปรับแต่งสำหรับ RTX 4050 6GB
# ══════════════════════════════════════════════
@dataclass
class Config:
    # ── Model ── (จาก schema เพื่อให้ train/eval/inference base ตรงกันเสมอ)
    MODEL_ID: str = schema.BASE_MODEL_ID

    # ── Reproducibility (สำคัญสำหรับ thesis: รันซ้ำได้ผลเดิม) ──
    SEED: int = 42

    # ── Dataset ──
    TRAIN_JSONL: str = "dataset/formatted/train.jsonl"
    VAL_JSONL:   str = "dataset/formatted/val.jsonl"

    # ── Output ──
    OUTPUT_DIR:     str = "models/typhoon-ocr-finetuned"
    CHECKPOINT_DIR: str = "models/checkpoints"
    BEST_MODEL_DIR: str = "models/best_model"

    # ── QLoRA Config ──
    LORA_R:       int   = 8           # data น้อย (~350) r=8 พอ ไม่ overfit
    LORA_ALPHA:   int   = 16          # alpha = 2*r (scaling มาตรฐาน)
    LORA_DROPOUT: float = 0.10        # 0.05 -> 0.10: regularize ขึ้นกัน overfit บน data เล็ก
    LORA_TARGETS: tuple = ("q_proj", "v_proj", "k_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj")

    # ── Training ──
    BATCH_SIZE:          int   = 1
    GRAD_ACCUM:          int   = 8    # effective batch = 8
    LEARNING_RATE:       float = 5e-5 # conservative -> loss นิ่ง เหมาะ data เล็ก
    NUM_EPOCHS:          int   = 10
    WARMUP_RATIO:        float = 0.05 # ใช้ ratio แทน fixed steps -> ปรับตาม dataset อัตโนมัติ
    MAX_SEQ_LEN:         int   = 1024 # Group 2: กลับไป MAX_PIXELS=256 (256px inference-only ให้ field
                                       # acc เพิ่มมหาศาลใน Group 1 experiment) วัด VRAM จริงด้วย sample
                                       # ยาวสุดจริง (256px) -> seq_len 1024 -> peak_reserved 6.17GB
                                       # (1152 -> 6.57GB, 1280 -> 6.98GB เกินขอบเขตที่เคยเทรนผ่านจริง
                                       # ที่ 6.36GB) format_dataset.py กรอง sample ยาวเกินทิ้ง (~12%)
    SAVE_STEPS:          int   = 50
    EVAL_STEPS:          int   = 50
    LOGGING_STEPS:       int   = 5
    EARLY_STOP_PATIENCE: int   = 5

    # ── Image ──
    MIN_PIXELS: int = 64 * 28 * 28
    MAX_PIXELS: int = 256 * 28 * 28  # Group 2: กลับขึ้น 256 (ดูเหตุผลที่ MAX_SEQ_LEN)


cfg = Config()
print("✓ Config created", flush=True)


# ══════════════════════════════════════════════
# HuggingFace Token
# ══════════════════════════════════════════════
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if HF_TOKEN:
    print(f"✓ HF Token: ...{HF_TOKEN[-4:]} (จาก env)", flush=True)
else:
    try:
        from huggingface_hub import get_token
        HF_TOKEN = get_token() or ""
        if HF_TOKEN:
            print(f"✓ HF Token: ...{HF_TOKEN[-4:]} (จาก HF cache)", flush=True)
    except Exception:
        pass

if not HF_TOKEN:
    print("⚠️  ไม่พบ HF Token — อาจโหลด model ไม่ได้", flush=True)


# ══════════════════════════════════════════════
# VRAM Management
# ══════════════════════════════════════════════
def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_vram(label=""):
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   VRAM {label}: {used:.2f} / {total:.1f} GB", flush=True)


# ══════════════════════════════════════════════
# STEP 1: โหลด Model
# ══════════════════════════════════════════════
def load_model_and_processor():
    print("\n📦 โหลด Model...", flush=True)
    print(f"   Model: {cfg.MODEL_ID}", flush=True)
    clear_vram()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.MODEL_ID,
        device_map={"": 0},
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        token=HF_TOKEN or None,
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    processor = Qwen2_5_VLProcessor.from_pretrained(
        cfg.MODEL_ID,
        min_pixels=cfg.MIN_PIXELS,
        max_pixels=cfg.MAX_PIXELS,
        trust_remote_code=True,
        token=HF_TOKEN or None,
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print(f"   ✓ โหลดสำเร็จ", flush=True)
    print_vram("หลังโหลด model")

    return model, processor


# ══════════════════════════════════════════════
# STEP 2: เพิ่ม LoRA Adapter
# ══════════════════════════════════════════════
def add_lora_adapter(model):
    lora_config = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        target_modules=list(cfg.LORA_TARGETS),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    trainable, total = 0, 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    print(f"\n🔧 LoRA Config:", flush=True)
    print(f"   r={cfg.LORA_R}, alpha={cfg.LORA_ALPHA}, targets={len(cfg.LORA_TARGETS)} layers", flush=True)
    print(f"   Trainable: {trainable:,} params ({100*trainable/total:.3f}%)", flush=True)
    print(f"   Frozen:    {total-trainable:,} params", flush=True)

    return model


# ══════════════════════════════════════════════
# STEP 3: Dataset
# ══════════════════════════════════════════════
class InvoiceDataset(Dataset):
    _trunc_warned = False  # เตือนเรื่อง target ถูกตัดแค่ครั้งเดียว

    def __init__(self, jsonl_path: str, processor, max_len: int):
        self.processor = processor
        self.max_len   = max_len
        self.samples   = []

        print(f"\n📂 โหลด Dataset: {jsonl_path}", flush=True)
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        print(f"   ✓ {len(self.samples)} samples", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample   = self.samples[idx]
        messages = sample["messages"]

        ground_truth = next(
            (m["content"] for m in messages if m["role"] == "assistant"), ""
        )

        user_content = messages[1]["content"]
        pil_image    = None
        text_prompt  = ""

        for item in user_content:
            if item["type"] == "image":
                b64 = item["image"].split(",")[-1]
                pil_image = ImageOps.exif_transpose(Image.open(BytesIO(base64.b64decode(b64)))).convert("RGB")
            elif item["type"] == "text":
                text_prompt = item["text"]

        if pil_image is None:
            pil_image = Image.new("RGB", (224, 224), color="white")

        conversation = [
            {"role": "system",    "content": messages[0]["content"]},
            {"role": "user",      "content": [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text": text_prompt}
            ]},
            {"role": "assistant", "content": ground_truth},
        ]

        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

        inputs = self.processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_len,
        )

        input_ids = inputs["input_ids"].squeeze(0)
        labels    = input_ids.clone()

        assistant_tokens = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n",
            add_special_tokens=False,
        )

        seq        = input_ids.tolist()
        mask_until = len(seq)

        for i in range(len(seq) - len(assistant_tokens) + 1):
            if seq[i:i+len(assistant_tokens)] == assistant_tokens:
                mask_until = i + len(assistant_tokens)
                break

        labels[:mask_until] = -100

        # เตือนถ้า target (ฝั่ง assistant) ถูกตัดทิ้งเพราะ MAX_SEQ_LEN สั้นไป
        # -> sample นั้นแทบไม่มี label ให้เรียน = เทรนเสียเปล่า
        answer_tokens = (labels != -100).sum().item()
        if answer_tokens < 5 and not InvoiceDataset._trunc_warned:
            InvoiceDataset._trunc_warned = True
            print(f"\n⚠️  พบ sample ที่ target ถูกตัด (เหลือ label {answer_tokens} tokens) "
                  f"-> เพิ่ม MAX_SEQ_LEN (ตอนนี้ {self.max_len}) หรือ MAX_PIXELS ให้พอ\n", flush=True)

        result = {
            "input_ids":      input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels":         labels,
        }

        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"]

        return result


# ══════════════════════════════════════════════
# STEP 4: Collate Function
# ══════════════════════════════════════════════
def make_collate_fn(pad_token_id: int):
    def collate_fn(batch):
        input_ids      = [item["input_ids"]      for item in batch]
        attention_mask = [item["attention_mask"]  for item in batch]
        labels         = [item["labels"]          for item in batch]

        max_len = max(x.shape[0] for x in input_ids)

        def pad1d(tensors, pad_val):
            out = torch.full((len(tensors), max_len), pad_val, dtype=torch.long)
            for i, t in enumerate(tensors):
                out[i, :t.shape[0]] = t
            return out

        result = {
            "input_ids":      pad1d(input_ids, pad_token_id),
            "attention_mask": pad1d(attention_mask, 0),
            "labels":         pad1d(labels, -100),
        }

        if "pixel_values" in batch[0]:
            result["pixel_values"]   = torch.cat([b["pixel_values"]   for b in batch], dim=0)
            result["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)

        return result
    return collate_fn


# ══════════════════════════════════════════════
# STEP 5: Training
# ══════════════════════════════════════════════
def train(model, processor):
    train_dataset = InvoiceDataset(cfg.TRAIN_JSONL, processor, cfg.MAX_SEQ_LEN)
    val_dataset   = InvoiceDataset(cfg.VAL_JSONL,   processor, cfg.MAX_SEQ_LEN)
    pad_id        = processor.tokenizer.pad_token_id or 0

    training_args = TrainingArguments(
        output_dir=cfg.CHECKPOINT_DIR,

        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRAD_ACCUM,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        learning_rate=cfg.LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.WARMUP_RATIO,
        num_train_epochs=cfg.NUM_EPOCHS,
        seed=cfg.SEED,
        data_seed=cfg.SEED,

        bf16=True,
        tf32=True,

        eval_strategy="steps",
        eval_steps=cfg.EVAL_STEPS,
        save_strategy="steps",
        save_steps=cfg.SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=cfg.LOGGING_STEPS,
        logging_first_step=True,
        report_to="none",

        optim="adamw_torch_fused",
        weight_decay=0.01,
        max_grad_norm=1.0,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=cfg.EARLY_STOP_PATIENCE,
            early_stopping_threshold=0.001,
        )],
        data_collator=make_collate_fn(pad_id),
    )

    total_steps = (len(train_dataset) // (cfg.BATCH_SIZE * cfg.GRAD_ACCUM)) * cfg.NUM_EPOCHS
    print(f"\n🚀 เริ่ม Fine-tune", flush=True)
    print(f"   Train:        {len(train_dataset)} samples", flush=True)
    print(f"   Val:          {len(val_dataset)} samples", flush=True)
    print(f"   Batch:        {cfg.BATCH_SIZE} × accum {cfg.GRAD_ACCUM} = {cfg.BATCH_SIZE*cfg.GRAD_ACCUM}", flush=True)
    print(f"   Epochs:       {cfg.NUM_EPOCHS}", flush=True)
    print(f"   Total steps:  ~{total_steps}", flush=True)
    print(f"   LR:           {cfg.LEARNING_RATE}", flush=True)
    print(f"   LoRA r:       {cfg.LORA_R}", flush=True)
    print_vram("ก่อน train")
    print("", flush=True)

    # auto-resume: ถ้ามี checkpoint ค้างจาก run ก่อน (เช่นโดน process/session
    # ตัดตอนกลางคัน) ให้ต่อจากจุดนั้นแทนเริ่มใหม่ทั้งหมด
    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = get_last_checkpoint(cfg.CHECKPOINT_DIR)
    if last_checkpoint:
        print(f"♻️  พบ checkpoint ค้าง -> resume จาก {last_checkpoint}", flush=True)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    print(f"\n💾 บันทึก Best Model → {cfg.BEST_MODEL_DIR}", flush=True)
    trainer.save_model(cfg.BEST_MODEL_DIR)
    processor.save_pretrained(cfg.BEST_MODEL_DIR)

    print(f"\n{'='*50}", flush=True)
    print(f"✅ Fine-tune เสร็จแล้ว!", flush=True)
    print(f"📁 Best model → {cfg.BEST_MODEL_DIR}/", flush=True)
    print(f"➡️  ต่อไป: python Finetune.py --mode test --image dataset/raw/inv_001.jpg", flush=True)
    print(f"{'='*50}", flush=True)

    return trainer


# ══════════════════════════════════════════════
# STEP 6: Inference Test  ← แก้ไขตรงนี้
# ══════════════════════════════════════════════
def test_inference(model_path: str, image_path: str):
    from peft import PeftModel
    from PIL import ImageOps

    print(f"\n🔍 ทดสอบ Inference: {image_path}", flush=True)
    clear_vram()

    # สร้าง offload folder สำหรับ layers ที่ล้น VRAM
    offload_dir = "offload_tmp"
    os.makedirs(offload_dir, exist_ok=True)

    # ใช้ 4-bit เหมือนตอน train → VRAM พอ
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("📦 โหลด base model (4-bit)...", flush=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.MODEL_ID,
        device_map={"": 0},           # โหลดเข้า GPU ตรงๆ เหมือนตอน train
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=HF_TOKEN or None,
    )

    print("🔧 โหลด LoRA adapter...", flush=True)
    model = PeftModel.from_pretrained(
        base,
        model_path,
        offload_folder=offload_dir,
        offload_buffers=True,
    )
    model.eval()

    processor = Qwen2_5_VLProcessor.from_pretrained(
        model_path,
        min_pixels=cfg.MIN_PIXELS,
        max_pixels=cfg.MAX_PIXELS,
        token=HF_TOKEN or None,
    )

    print("🖼️  โหลดรูปภาพ...", flush=True)
    image = ImageOps.exif_transpose(Image.open(image_path).convert("RGB"))

    # ใช้ prompt ชุดเดียวกับตอน train (JSON) เพื่อให้ผลตรง ไม่ใช่ Markdown
    messages = schema.build_inference_messages(image)

    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

    print("⚙️  กำลัง generate...", flush=True)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=768,
            do_sample=False,
            repetition_penalty=1.3,      # เพิ่มจาก 1.1 → 1.3
            no_repeat_ngram_size=5,      # ห้ามซ้ำ n-gram ขนาด 5
            pad_token_id=processor.tokenizer.eos_token_id,
)

    output_text = processor.tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    print("\n📄 ผลลัพธ์ (raw):", flush=True)
    print("─" * 40, flush=True)
    print(output_text, flush=True)
    print("─" * 40, flush=True)

    # parse เป็น fields ตาม schema -> ดูว่า extract ได้ครบไหม
    fields = schema.parse_model_json(output_text)
    print("\n📋 Parsed fields (canonical):", flush=True)
    print(schema.build_canonical_json(fields, indent=2), flush=True)

    return output_text


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["train", "test"], default="train")
    parser.add_argument("--image", help="path รูปสำหรับ test mode")
    args = parser.parse_args()

    print(f"\n{'='*50}", flush=True)
    print(f"MODE: {args.mode}", flush=True)

    if not torch.cuda.is_available():
        print("❌ ไม่พบ GPU — QLoRA ต้องใช้ CUDA")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🖥️  GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)", flush=True)

    for path in [cfg.TRAIN_JSONL, cfg.VAL_JSONL]:
        if not Path(path).exists():
            print(f"❌ ไม่พบ {path} — รัน format_dataset.py ก่อน")
            sys.exit(1)
        print(f"   ✓ พบ {path}", flush=True)

    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(cfg.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        set_seed(cfg.SEED)   # reproducible: shuffle, dropout, init เหมือนเดิมทุกครั้ง
        model, processor = load_model_and_processor()
        model = add_lora_adapter(model)
        train(model, processor)

    elif args.mode == "test":
        if not args.image:
            print("❌ ระบุ --image path")
            sys.exit(1)
        test_inference(cfg.BEST_MODEL_DIR, args.image)
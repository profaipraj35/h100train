#!/usr/bin/env python3
"""
QLoRA Fine-tuning — Multi-Model Document Processing
====================================================
Supports 30+ specialist and generalist models across 8 architecture types
for OCR, KIE, NER, Classification, Table Understanding tasks.

Usage
-----
    python finetune_qlora.py                            # default config
    python finetune_qlora.py --config my_config.yaml    # custom config
    python finetune_qlora.py --resume                    # force resume
    python finetune_qlora.py --model gemma3_4b           # override model

Features: configurable via YAML, real-time RAM/VRAM monitoring,
automatic checkpointing, resume from interruption.
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

# ── Architecture groupings ────────────────────────────────────────────────────

VISION_CHAT_ARCHS = {
    "qwen_vl", "gemma", "phi_vision", "llava", "got_ocr",
    "llama_vision", "internvl", "mplug", "vision_chat",
}
ENC_DEC_ARCHS = {"vision_enc_dec"}
TEXT_ARCHS = {"text_causal"}
IMAGE_CLS_ARCHS = {"image_cls"}
SEQ_CLS_ARCHS = {"seq_cls"}
TOKEN_CLS_ARCHS = {"token_cls"}
DETR_ARCHS = {"detr"}
UNSUPPORTED_ARCHS = {"paddle", "external"}

ALL_GENERATIVE_ARCHS = VISION_CHAT_ARCHS | ENC_DEC_ARCHS | TEXT_ARCHS
ALL_CLASSIFICATION_ARCHS = IMAGE_CLS_ARCHS | SEQ_CLS_ARCHS | TOKEN_CLS_ARCHS

DEFAULT_TARGET_MODULES = {
    "qwen_vl":        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma":          ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi_vision":     ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    "llava":          ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama_vision":   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "internvl":       ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mplug":          ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "vision_chat":    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "text_causal":    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "vision_enc_dec": ["q_proj", "k_proj", "v_proj", "out_proj"],
    "got_ocr":        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "image_cls":      ["query", "key", "value"],
    "seq_cls":        ["query", "key", "value", "dense"],
    "token_cls":      ["query", "key", "value", "dense"],
    "detr":           ["q_proj", "k_proj", "v_proj", "out_proj"],
}

SCRIPT_DIR = Path(__file__).resolve().parent

# ── Utilities ─────────────────────────────────────────────────────────────────


def print_banner(text: str):
    w = max(len(text) + 4, 60)
    print(f"\n{'=' * w}")
    print(f"  {text}")
    print(f"{'=' * w}\n")


def print_resource_usage():
    vm = psutil.virtual_memory()
    lines = [
        f"  [System RAM]  Used: {vm.used / 1e9:.1f} GB  |  "
        f"Free: {vm.available / 1e9:.1f} GB  |  "
        f"Total: {vm.total / 1e9:.1f} GB  ({vm.percent}%)"
    ]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            alloc = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = props.total_memory / 1e9
            free = total - alloc
            lines.append(
                f"  [GPU {i}: {props.name}]  Allocated: {alloc:.1f} GB  |  "
                f"Reserved: {reserved:.1f} GB  |  Free: {free:.1f} GB  |  "
                f"Total: {total:.1f} GB"
            )
    else:
        lines.append("  [GPU]  No CUDA device available")

    border = "-" * 64
    print(f"\n{border}")
    print("\n".join(lines))
    print(f"{border}\n")


# ── Config ────────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_model_config(cfg: dict, override_model: Optional[str] = None) -> dict:
    key = override_model or cfg["active_model"]
    if key not in cfg["models"]:
        available = ", ".join(sorted(cfg["models"].keys()))
        sys.exit(f"ERROR: model key '{key}' not found.\n  Available: {available}")
    model_cfg = cfg["models"][key]
    model_cfg["key"] = key
    return model_cfg


# ── BitsAndBytes ──────────────────────────────────────────────────────────────


def make_bnb_config(qlora_cfg: dict):
    from transformers import BitsAndBytesConfig

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(
        qlora_cfg.get("bnb_4bit_compute_dtype", "bfloat16"), torch.bfloat16
    )

    if qlora_cfg.get("quantization_bits", 4) == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=qlora_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=qlora_cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=compute_dtype,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


# ── Label map helpers ─────────────────────────────────────────────────────────


def build_classification_label_map(records: List[dict]) -> tuple:
    labels = set()
    for rec in records:
        cls = rec.get("document_class") or rec.get("response", "").strip()
        if cls:
            labels.add(cls)
    labels = sorted(labels)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def build_ner_label_map(records: List[dict]) -> tuple:
    entity_types = set()
    for rec in records:
        resp = rec.get("response", "")
        try:
            entities = json.loads(resp) if isinstance(resp, str) else resp
            if isinstance(entities, list):
                for ent in entities:
                    if isinstance(ent, dict) and "type" in ent:
                        entity_types.add(ent["type"])
        except (json.JSONDecodeError, TypeError):
            pass
    entity_types = sorted(entity_types)
    bio_labels = ["O"]
    for t in entity_types:
        bio_labels.append(f"B-{t}")
        bio_labels.append(f"I-{t}")
    label2id = {l: i for i, l in enumerate(bio_labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


# ── Model loading ─────────────────────────────────────────────────────────────


def load_model_and_processor(model_cfg: dict, qlora_cfg: dict, num_labels: int = 0):
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageClassification,
        AutoModelForObjectDetection,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoProcessor,
        AutoTokenizer,
    )
    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        AutoModelForVision2Seq = None

    arch = model_cfg["arch"]
    hf_id = model_cfg["hf_id"]
    trust = model_cfg.get("trust_remote_code", False)
    bnb_config = make_bnb_config(qlora_cfg)

    if arch in UNSUPPORTED_ARCHS:
        note = model_cfg.get("note", "")
        sys.exit(
            f"ERROR: '{model_cfg['key']}' ({hf_id}) cannot be fine-tuned "
            f"with this script.\n  {note}"
        )

    common = dict(
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust,
        torch_dtype="auto",
    )

    print(f"Loading {hf_id}  (arch={arch}) ...")
    print_resource_usage()

    # ── Text-only causal LM (Mistral, Llama, Phi-mini) ───────────────────────
    if arch in TEXT_ARCHS:
        processor = AutoTokenizer.from_pretrained(
            hf_id, trust_remote_code=trust, padding_side="right"
        )
        if processor.pad_token is None:
            processor.pad_token = processor.eos_token
        model = AutoModelForCausalLM.from_pretrained(hf_id, **common)

    # ── Vision encoder-decoder (TrOCR, Donut) ────────────────────────────────
    elif arch in ENC_DEC_ARCHS:
        if AutoModelForVision2Seq is None:
            sys.exit("ERROR: AutoModelForVision2Seq not available in this transformers version. Upgrade transformers to use TrOCR/Donut.")
        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=trust)
        model = AutoModelForVision2Seq.from_pretrained(hf_id, **common)

    # ── Image classification (DiT, BEiT) ─────────────────────────────────────
    elif arch in IMAGE_CLS_ARCHS:
        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=trust)
        model = AutoModelForImageClassification.from_pretrained(
            hf_id, num_labels=num_labels, ignore_mismatched_sizes=True, **common
        )

    # ── Sequence classification (BERT, RoBERTa) ─────────────────────────────
    elif arch in SEQ_CLS_ARCHS:
        processor = AutoTokenizer.from_pretrained(
            hf_id, trust_remote_code=trust, padding_side="right"
        )
        if processor.pad_token is None:
            processor.pad_token = processor.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_id, num_labels=num_labels, ignore_mismatched_sizes=True, **common
        )

    # ── Token classification (LayoutLMv3, LayoutLMv2, BROS) ──────────────────
    elif arch in TOKEN_CLS_ARCHS:
        processor = AutoProcessor.from_pretrained(
            hf_id, trust_remote_code=trust, apply_ocr=False
        )
        model = AutoModelForTokenClassification.from_pretrained(
            hf_id, num_labels=num_labels, ignore_mismatched_sizes=True, **common
        )

    # ── Object detection (Table Transformer / DETR) ──────────────────────────
    elif arch in DETR_ARCHS:
        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=trust)
        model = AutoModelForObjectDetection.from_pretrained(hf_id, **common)

    # ── Vision-language chat models (Qwen-VL, Gemma, Phi, LLaVA, etc.) ──────
    else:
        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=trust)
        if AutoModelForVision2Seq is not None:
            try:
                model = AutoModelForVision2Seq.from_pretrained(hf_id, **common)
            except (ValueError, KeyError, ImportError):
                model = AutoModelForCausalLM.from_pretrained(hf_id, **common)
        else:
            # AutoModelForVision2Seq unavailable — try arch-specific classes
            model = None
            if arch == "qwen_vl":
                for cls_name in ("Qwen2_5_VLForConditionalGeneration", "Qwen2VLForConditionalGeneration"):
                    try:
                        import importlib
                        cls = getattr(importlib.import_module("transformers"), cls_name)
                        model = cls.from_pretrained(hf_id, **common)
                        break
                    except (ImportError, AttributeError, ValueError):
                        continue
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(hf_id, **common)

    print(f"Model loaded.  Total parameters: {model.num_parameters():,}")
    print_resource_usage()
    return model, processor


# ── PEFT / LoRA ───────────────────────────────────────────────────────────────


def apply_qlora(model, qlora_cfg: dict, arch: str, gradient_checkpointing: bool):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing
    )

    target = qlora_cfg.get("target_modules", "auto")
    if target == "auto":
        target = DEFAULT_TARGET_MODULES.get(arch, ["q_proj", "v_proj"])

    if arch in ENC_DEC_ARCHS:
        task_type = TaskType.SEQ_2_SEQ_LM
    elif arch in SEQ_CLS_ARCHS or arch in IMAGE_CLS_ARCHS:
        task_type = TaskType.SEQ_CLS
    elif arch in TOKEN_CLS_ARCHS:
        task_type = TaskType.TOKEN_CLS
    elif arch in DETR_ARCHS:
        task_type = TaskType.FEATURE_EXTRACTION
    else:
        task_type = TaskType.CAUSAL_LM

    lora_config = LoraConfig(
        r=qlora_cfg["rank"],
        lora_alpha=qlora_cfg["alpha"],
        lora_dropout=qlora_cfg.get("dropout", 0.0),
        bias=qlora_cfg.get("bias", "none"),
        target_modules=target,
        task_type=task_type,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── Datasets ──────────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def gather_records(data_cfg: dict, base_dir: Path):
    train_all, val_all, test_all = [], [], []
    data_dir = base_dir / data_cfg["data_dir"]

    for ds in data_cfg["datasets"]:
        ds_dir = data_dir / ds
        for split_name, target in [("train", train_all), ("val", val_all), ("test", test_all)]:
            fp = ds_dir / f"{split_name}.jsonl"
            if fp.exists():
                recs = load_jsonl(fp)
                if recs:
                    target.extend(recs)
                    print(f"  {ds}/{split_name}: {len(recs):,} records")

    if not val_all and train_all:
        random.seed(42)
        shuffled = list(train_all)
        random.shuffle(shuffled)
        split_n = max(1, int(len(shuffled) * data_cfg.get("val_split_ratio", 0.1)))
        val_all = shuffled[:split_n]
        train_all = shuffled[split_n:]
        print(f"  (split {split_n:,} records from train for validation)")

    max_train = data_cfg.get("max_train_samples")
    if max_train:
        train_all = train_all[:max_train]

    return train_all, val_all, test_all


# ── Dataset: vision-language chat models ──────────────────────────────────────


class VisionChatDataset(Dataset):
    """For VLMs: Qwen-VL, Gemma, Phi-Vision, LLaVA, Llama-Vision, InternVL, mPLUG, etc."""

    def __init__(self, records, processor, base_dir, max_length, arch):
        self.records = records
        self.processor = processor
        self.base_dir = Path(base_dir)
        self.max_length = max_length
        self.arch = arch

        # Resolve image-pad token ID for Qwen2.5-VL mm_token_type_ids
        self._img_token_id = None
        if arch == "qwen_vl" and hasattr(processor, "tokenizer"):
            vocab = processor.tokenizer.get_vocab()
            for tok in ("<|image_pad|>", "<image_pad>", "<image>", "<img>"):
                if tok in vocab:
                    self._img_token_id = vocab[tok]
                    break

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = os.path.normpath(self.base_dir / rec["image_path"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"WARNING: cannot open {img_path}: {e} — using blank image")
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        # Resize to fixed size so image token count is predictable and fits in max_length
        image = image.resize((224, 224), Image.LANCZOS)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": rec["instruction"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": rec["response"]}],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # No truncation/padding here — the collator handles per-batch padding.
        # Passing max_length causes image-token count mismatch in Qwen2.5-VL.
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        # Truncate only if the sequence exceeds max_length (no padding)
        seq_len = inputs["input_ids"].shape[0] if "input_ids" in inputs else 0
        if seq_len > self.max_length:
            for key in ("input_ids", "attention_mask"):
                if key in inputs and inputs[key].dim() == 1:
                    inputs[key] = inputs[key][:self.max_length]

        # Build labels: mask prompt tokens so loss is only on the response
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": rec["instruction"]},
                ],
            },
        ]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = self.processor.tokenizer(prompt_text)
        prompt_len = min(len(prompt_tokens["input_ids"]), inputs["input_ids"].shape[0])

        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None) or 0
        labels = inputs["input_ids"].clone()
        labels[:prompt_len] = -100
        labels[labels == pad_id] = -100
        inputs["labels"] = labels

        # Explicit mm_token_type_ids for Qwen2.5-VL: avoids broken internal computation
        # that mismatches shapes when sequences are padded across a batch.
        if self.arch == "qwen_vl":
            mm = torch.zeros_like(inputs["input_ids"])
            if self._img_token_id is not None:
                mm[inputs["input_ids"] == self._img_token_id] = 1
            inputs["mm_token_type_ids"] = mm

        return inputs


# ── Dataset: vision encoder-decoder ──────────────────────────────────────────


class VisionEncDecDataset(Dataset):
    """For TrOCR, Donut."""

    def __init__(self, records, processor, base_dir, max_length):
        self.records = records
        self.processor = processor
        self.base_dir = Path(base_dir)
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = os.path.normpath(self.base_dir / rec["image_path"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (384, 384), (128, 128, 128))

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        tokenizer = (
            self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        )
        labels = tokenizer(
            rec["response"],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        ).input_ids.squeeze(0)

        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ── Dataset: text-only causal LM ─────────────────────────────────────────────


class TextChatDataset(Dataset):
    """For Mistral, Llama, Phi-mini. Uses OCR text in lieu of images."""

    def __init__(self, records, tokenizer, base_dir, max_length):
        self.records = records
        self.tokenizer = tokenizer
        self.base_dir = Path(base_dir)
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        doc_text = rec.get("full_text", "")
        if not doc_text and "entities" in rec:
            doc_text = json.dumps(rec["entities"], ensure_ascii=False)
        if not doc_text:
            doc_text = rec.get("response", "")

        user_content = f"{rec['instruction']}\n\nDocument text:\n{doc_text}"

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": rec["response"]},
        ]

        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_enc = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        prompt_messages = [{"role": "user", "content": user_content}]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_enc = self.tokenizer(prompt_text, truncation=True, max_length=self.max_length)
        prompt_len = len(prompt_enc["input_ids"])

        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ── Dataset: image classification ─────────────────────────────────────────────


class ImageClassificationDataset(Dataset):
    """For DiT, BEiT. Filters to classification-task records only."""

    def __init__(self, records, processor, base_dir, label2id):
        self.records = [
            r for r in records
            if r.get("task") == "classification" and (r.get("document_class") or r.get("response", "").strip())
        ]
        self.processor = processor
        self.base_dir = Path(base_dir)
        self.label2id = label2id

        if not self.records:
            print("  WARNING: no classification records found for ImageClassificationDataset")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = os.path.normpath(self.base_dir / rec["image_path"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        cls = rec.get("document_class") or rec.get("response", "").strip()
        inputs["labels"] = torch.tensor(self.label2id.get(cls, 0), dtype=torch.long)
        return inputs


# ── Dataset: sequence classification ──────────────────────────────────────────


class SequenceClassificationDataset(Dataset):
    """For BERT, RoBERTa. Uses OCR text for classification."""

    def __init__(self, records, tokenizer, base_dir, label2id, max_length):
        self.records = [
            r for r in records
            if r.get("task") == "classification" and (r.get("document_class") or r.get("response", "").strip())
        ]
        self.tokenizer = tokenizer
        self.base_dir = Path(base_dir)
        self.label2id = label2id
        self.max_length = max_length

        if not self.records:
            print("  WARNING: no classification records found for SequenceClassificationDataset")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        text = rec.get("full_text", "")
        if not text:
            text = rec.get("response", "").strip()
        if not text:
            text = rec.get("instruction", "")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        cls = rec.get("document_class") or rec.get("response", "").strip()
        inputs["labels"] = torch.tensor(self.label2id.get(cls, 0), dtype=torch.long)
        return inputs


# ── Dataset: token classification ─────────────────────────────────────────────


class TokenClassificationDataset(Dataset):
    """For LayoutLMv3, LayoutLMv2, BROS. Extracts words via OCR for NER tasks.

    Uses the record's response (entity list) to assign BIO labels.
    For records without parseable entities, falls back to all-O labels.
    """

    def __init__(self, records, processor, base_dir, label2id, max_length):
        self.records = [r for r in records if r.get("task") in ("ner", "kie")]
        self.processor = processor
        self.base_dir = Path(base_dir)
        self.label2id = label2id
        self.max_length = max_length

        self._ocr_reader = None

        if not self.records:
            print("  WARNING: no NER/KIE records found for TokenClassificationDataset")

    def _get_ocr_reader(self):
        if self._ocr_reader is None:
            try:
                import easyocr
                self._ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
            except ImportError:
                sys.exit(
                    "ERROR: token_cls models require easyocr for word extraction.\n"
                    "  Install: pip install easyocr"
                )
        return self._ocr_reader

    def __len__(self):
        return len(self.records)

    def _run_ocr(self, img_path: str):
        reader = self._get_ocr_reader()
        results = reader.readtext(img_path)
        words, boxes = [], []
        for bbox, text, _conf in results:
            if not text.strip():
                continue
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x0, y0 = int(min(x_coords)), int(min(y_coords))
            x1, y1 = int(max(x_coords)), int(max(y_coords))
            words.append(text.strip())
            boxes.append([x0, y0, x1, y1])
        return words, boxes

    def _parse_entities(self, rec):
        resp = rec.get("response", "")
        try:
            entities = json.loads(resp) if isinstance(resp, str) else resp
            if isinstance(entities, list):
                return entities
        except (json.JSONDecodeError, TypeError):
            pass
        return []

    def _assign_bio_labels(self, words, entities):
        labels = [self.label2id.get("O", 0)] * len(words)
        words_lower = [w.lower() for w in words]

        for ent in entities:
            if not isinstance(ent, dict):
                continue
            ent_type = ent.get("type", "")
            ent_text = ent.get("entity", "")
            if not ent_type or not ent_text:
                continue

            ent_words = ent_text.lower().split()
            for i in range(len(words_lower) - len(ent_words) + 1):
                if words_lower[i : i + len(ent_words)] == ent_words:
                    b_label = self.label2id.get(f"B-{ent_type}", 0)
                    i_label = self.label2id.get(f"I-{ent_type}", 0)
                    labels[i] = b_label
                    for j in range(1, len(ent_words)):
                        labels[i + j] = i_label
                    break

        return labels

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = os.path.normpath(self.base_dir / rec["image_path"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        words, boxes = self._run_ocr(img_path)

        if not words:
            words, boxes = ["[EMPTY]"], [[0, 0, 1, 1]]

        entities = self._parse_entities(rec)
        word_labels = self._assign_bio_labels(words, entities)

        w, h = image.size
        norm_boxes = [
            [
                min(1000, max(0, int(b[0] * 1000 / w))),
                min(1000, max(0, int(b[1] * 1000 / h))),
                min(1000, max(0, int(b[2] * 1000 / w))),
                min(1000, max(0, int(b[3] * 1000 / h))),
            ]
            for b in boxes
        ]

        try:
            encoding = self.processor(
                image,
                words,
                boxes=norm_boxes,
                word_labels=word_labels,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )
        except Exception:
            encoding = self.processor(
                image,
                words,
                boxes=norm_boxes,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )
            labels_tensor = torch.full(
                (encoding["input_ids"].shape[-1],), -100, dtype=torch.long
            )
            encoding["labels"] = labels_tensor.unsqueeze(0)

        return {k: v.squeeze(0) for k, v in encoding.items() if isinstance(v, torch.Tensor)}


# ── Dataset builder ───────────────────────────────────────────────────────────


def build_datasets(arch, records_train, records_val, processor, base_dir, max_length,
                   label2id=None):
    if arch in VISION_CHAT_ARCHS:
        ds_train = VisionChatDataset(records_train, processor, base_dir, max_length, arch)
        ds_val = VisionChatDataset(records_val, processor, base_dir, max_length, arch)
    elif arch in ENC_DEC_ARCHS:
        ds_train = VisionEncDecDataset(records_train, processor, base_dir, max_length)
        ds_val = VisionEncDecDataset(records_val, processor, base_dir, max_length)
    elif arch in TEXT_ARCHS:
        ds_train = TextChatDataset(records_train, processor, base_dir, max_length)
        ds_val = TextChatDataset(records_val, processor, base_dir, max_length)
    elif arch in IMAGE_CLS_ARCHS:
        ds_train = ImageClassificationDataset(records_train, processor, base_dir, label2id)
        ds_val = ImageClassificationDataset(records_val, processor, base_dir, label2id)
    elif arch in SEQ_CLS_ARCHS:
        ds_train = SequenceClassificationDataset(
            records_train, processor, base_dir, label2id, max_length
        )
        ds_val = SequenceClassificationDataset(
            records_val, processor, base_dir, label2id, max_length
        )
    elif arch in TOKEN_CLS_ARCHS:
        ds_train = TokenClassificationDataset(
            records_train, processor, base_dir, label2id, max_length
        )
        ds_val = TokenClassificationDataset(
            records_val, processor, base_dir, label2id, max_length
        )
    elif arch in DETR_ARCHS:
        sys.exit(
            "ERROR: DETR / Table Transformer requires bounding-box annotations in "
            "COCO format.\n  The current formatted_data does not include table bbox "
            "annotations.\n  Provide a COCO-format annotation file to proceed."
        )
    else:
        sys.exit(f"ERROR: unknown architecture '{arch}'")
    return ds_train, ds_val


# ── Data collator ─────────────────────────────────────────────────────────────


@dataclass
class MultimodalCollator:
    """Batches multimodal samples.

    - pixel_values: concatenated along dim=0 (Qwen2.5-VL expects all patches
      in a single [total_patches, C] tensor, not stacked per image).
    - 1-D sequence tensors: padded to the longest sequence in the batch.
    - Same-shape tensors: stacked normally.
    """

    pad_token_id: int = 0

    # Keys whose patches must be concatenated across the batch, not stacked.
    _CONCAT_KEYS = frozenset({"pixel_values"})

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch: Dict[str, Any] = {}
        keys = features[0].keys()

        for key in keys:
            values = [f[key] for f in features]

            if not isinstance(values[0], torch.Tensor):
                batch[key] = values
                continue

            # Qwen2.5-VL: pixel_values is [P, C] per item → [B*P, C] in batch
            if key in self._CONCAT_KEYS:
                batch[key] = torch.cat(values, dim=0)
                continue

            shapes_match = all(v.shape == values[0].shape for v in values)

            if shapes_match:
                batch[key] = torch.stack(values)
            elif values[0].dim() == 1:
                max_len = max(v.shape[0] for v in values)
                padded = []
                for v in values:
                    gap = max_len - v.shape[0]
                    if gap > 0:
                        if key == "labels":
                            pad_val = -100
                        elif key in ("attention_mask", "mm_token_type_ids"):
                            pad_val = 0
                        else:
                            pad_val = self.pad_token_id
                        v = torch.nn.functional.pad(v, (0, gap), value=pad_val)
                    padded.append(v)
                batch[key] = torch.stack(padded)
            else:
                try:
                    batch[key] = torch.stack(values)
                except RuntimeError:
                    batch[key] = torch.cat(values, dim=0)

        return batch


# ── Trainer callback for resource monitoring ──────────────────────────────────

from transformers import TrainerCallback  # noqa: E402


class ResourceMonitorCallback(TrainerCallback):
    """Prints system RAM and GPU VRAM usage at a configurable interval."""

    def __init__(self, interval_seconds: int = 30):
        super().__init__()
        self.interval = interval_seconds
        self._last_time = 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        now = time.time()
        if self.interval > 0 and now - self._last_time >= self.interval:
            self._last_time = now
            print_resource_usage()

    def on_train_begin(self, args, state, control, **kwargs):
        print_banner("Initial resource snapshot")
        print_resource_usage()

    def on_train_end(self, args, state, control, **kwargs):
        print_banner("Final resource snapshot")
        print_resource_usage()


# ── Checkpoint discovery ──────────────────────────────────────────────────────


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    out = Path(output_dir)
    if not out.exists():
        return None
    ckpts = sorted(out.glob("checkpoint-*"), key=os.path.getmtime)
    return str(ckpts[-1]) if ckpts else None


# ── Training ──────────────────────────────────────────────────────────────────


def build_trainer(model, processor, ds_train, ds_val, cfg, arch, output_dir):
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        Trainer,
        TrainingArguments,
    )

    tcfg = cfg["training"]
    ecfg = cfg["evaluation"]
    ccfg = cfg["checkpointing"]
    hcfg = cfg["hardware"]
    ocfg = cfg["output"]

    # save_steps must be a multiple of eval_steps for load_best_model_at_end
    eval_steps = ecfg["eval_steps"]
    raw_save = ccfg["save_steps"]
    save_steps = max(eval_steps, (raw_save // eval_steps) * eval_steps) or eval_steps

    pad_token_id = 0
    if hasattr(processor, "pad_token_id") and processor.pad_token_id is not None:
        pad_token_id = processor.pad_token_id
    elif hasattr(processor, "tokenizer") and processor.tokenizer.pad_token_id is not None:
        pad_token_id = processor.tokenizer.pad_token_id

    collator = MultimodalCollator(pad_token_id=pad_token_id)
    monitor = ResourceMonitorCallback(interval_seconds=hcfg.get("ram_monitor_interval", 30))

    shared_args = dict(
        output_dir=output_dir,
        per_device_train_batch_size=tcfg["batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        num_train_epochs=tcfg["num_epochs"],
        learning_rate=float(tcfg["learning_rate"]),
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        optim=tcfg["optimizer"],
        fp16=tcfg.get("fp16", False),
        bf16=tcfg.get("bf16", True),
        eval_strategy=ecfg["eval_strategy"],
        eval_steps=ecfg["eval_steps"],
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=ccfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=ocfg["logging_steps"],
        report_to=ocfg["report_to"],
        seed=hcfg["seed"],
        dataloader_num_workers=hcfg["dataloader_num_workers"],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    if arch in ENC_DEC_ARCHS:
        args = Seq2SeqTrainingArguments(**shared_args, predict_with_generate=False)
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            data_collator=collator,
            callbacks=[monitor],
        )
    else:
        args = TrainingArguments(**shared_args)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            data_collator=collator,
            callbacks=[monitor],
        )

    return trainer


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for document processing models")
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR / "finetune_config.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument("--model", default=None, help="Override active_model from config")
    parser.add_argument(
        "--resume", action="store_true", help="Force resume from latest checkpoint"
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    model_cfg = resolve_model_config(cfg, args.model)

    arch = model_cfg["arch"]
    hf_id = model_cfg["hf_id"]
    output_dir = str(SCRIPT_DIR / cfg["output"]["output_dir"] / model_cfg["key"])
    save_dir = str(Path(output_dir) / "final_adapter")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    print_banner(f"QLoRA Fine-tuning: {model_cfg['key']}")
    print(f"  Model    : {hf_id}")
    print(f"  Arch     : {arch}")
    print(f"  Category : {model_cfg.get('category', '?')}")
    print(f"  Size     : {model_cfg.get('size', '?')}")
    print(f"  Output   : {output_dir}")
    print(f"  Config   : {args.config}")
    if model_cfg.get("note"):
        print(f"  Note     : {model_cfg['note']}")
    print()

    with open(os.path.join(output_dir, "resolved_config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ── Load data ─────────────────────────────────────────────────────────────
    print_banner("Loading data")
    train_recs, val_recs, test_recs = gather_records(cfg["data"], SCRIPT_DIR)
    print(f"\n  Totals:  train={len(train_recs):,}  val={len(val_recs):,}  test={len(test_recs):,}")

    if not train_recs:
        sys.exit("ERROR: no training records found. Check data.datasets in config.")

    max_eval = cfg["evaluation"].get("max_eval_samples")
    if max_eval and len(val_recs) > max_eval:
        random.seed(42)
        val_recs = random.sample(val_recs, max_eval)
        print(f"  (capped validation to {max_eval} samples)")

    # ── Build label maps for classification / NER architectures ───────────────
    label2id, id2label, num_labels = None, None, 0

    if arch in IMAGE_CLS_ARCHS or arch in SEQ_CLS_ARCHS:
        all_recs = train_recs + val_recs + test_recs
        label2id, id2label = build_classification_label_map(all_recs)
        num_labels = len(label2id)
        print(f"\n  Classification labels ({num_labels}): {list(label2id.keys())[:10]}...")

    elif arch in TOKEN_CLS_ARCHS:
        all_recs = train_recs + val_recs + test_recs
        label2id, id2label = build_ner_label_map(all_recs)
        num_labels = len(label2id)
        print(f"\n  NER BIO labels ({num_labels}): {list(label2id.keys())[:10]}...")

    # Save label maps for inference
    if label2id:
        label_map_path = os.path.join(output_dir, "label_map.json")
        with open(label_map_path, "w") as f:
            json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
        print(f"  Label map saved to: {label_map_path}")

    # ── Load model ────────────────────────────────────────────────────────────
    print_banner("Loading model")
    model, processor = load_model_and_processor(model_cfg, cfg["qlora"], num_labels=num_labels)

    # ── Apply QLoRA ───────────────────────────────────────────────────────────
    print_banner("Applying QLoRA adapters")
    grad_ckpt = cfg["training"].get("use_gradient_checkpointing", True)
    model = apply_qlora(model, cfg["qlora"], arch, grad_ckpt)
    print_resource_usage()

    # ── Build datasets ────────────────────────────────────────────────────────
    print_banner("Building datasets")
    max_seq = cfg["training"].get("max_seq_length", 1024)
    ds_train, ds_val = build_datasets(
        arch, train_recs, val_recs, processor, SCRIPT_DIR, max_seq, label2id=label2id
    )
    print(f"  Train dataset: {len(ds_train):,} samples")
    print(f"  Val   dataset: {len(ds_val):,} samples")

    # ── Trainer ───────────────────────────────────────────────────────────────
    print_banner("Setting up trainer")
    trainer = build_trainer(model, processor, ds_train, ds_val, cfg, arch, output_dir)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    resume_cfg = cfg["checkpointing"].get("resume_from_checkpoint", False)
    resume_path = None

    if args.resume or resume_cfg:
        if isinstance(resume_cfg, str) and os.path.isdir(resume_cfg):
            resume_path = resume_cfg
        else:
            resume_path = find_latest_checkpoint(output_dir)

        if resume_path:
            print(f"  Resuming from checkpoint: {resume_path}")
        else:
            print("  No checkpoint found — starting from scratch.")

    # ── Train ─────────────────────────────────────────────────────────────────
    qlora_info = cfg["qlora"]
    print_banner("Training")
    print(
        f"  LoRA rank={qlora_info['rank']}  alpha={qlora_info['alpha']}  "
        f"dropout={qlora_info.get('dropout', 0)}"
    )
    print(
        f"  Batch={cfg['training']['batch_size']}  "
        f"GradAccum={cfg['training']['gradient_accumulation_steps']}  "
        f"Epochs={cfg['training']['num_epochs']}  "
        f"LR={cfg['training']['learning_rate']}"
    )
    print(f"  MaxSeqLen={max_seq}  Optimizer={cfg['training']['optimizer']}")
    print()

    train_result = trainer.train(resume_from_checkpoint=resume_path)

    metrics = train_result.metrics
    print(f"\nTraining complete.")
    print(f"  Runtime     : {metrics.get('train_runtime', 0):.0f}s")
    print(f"  Samples/sec : {metrics.get('train_samples_per_second', 0):.2f}")
    print(f"  Final loss  : {metrics.get('train_loss', '?')}")

    # ── Save final adapter ────────────────────────────────────────────────────
    print_banner("Saving final adapter")
    model.save_pretrained(save_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(save_dir)
    print(f"  Adapter saved to: {save_dir}")

    metrics_path = os.path.join(output_dir, "train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Metrics saved to: {metrics_path}")

    print_resource_usage()
    print_banner("Done")


if __name__ == "__main__":
    main()

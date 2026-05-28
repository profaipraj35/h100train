"""
evaluate_all_models.py — Evaluate all specialist & generalist models on
OCR / Classification / KIE / NER using eval_datasets/ ground truth.

Reads  : eval_datasets/{task}/eval_test.jsonl   (from build_eval_datasets.py)
Writes : eval_results/{model_key}/{task}_results.json
         eval_results/summary.csv
         eval_results/eval_checkpoint.json

Usage:
  python evaluate_all_models.py                              # all non-stub models
  python evaluate_all_models.py --models dit_base qwen_vl_3b
  python evaluate_all_models.py --tasks classification ner
  python evaluate_all_models.py --max 200                    # cap records per task
  python evaluate_all_models.py --resume                     # skip completed
  python evaluate_all_models.py --list                       # show all models
"""

import argparse
import concurrent.futures
import csv
import gc
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

SAMPLE_TIMEOUT_SEC = 120  # kill a single inference call after this many seconds

BASE_DIR    = Path(__file__).parent.resolve()
EVAL_DIR    = BASE_DIR / "eval_datasets"
RESULTS_DIR = BASE_DIR / "eval_results"
CKPT_FILE   = RESULTS_DIR / "eval_checkpoint.json"

# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

RVL_CLASSES = [
    "letter", "form", "email", "handwritten", "advertisement",
    "scientific_report", "scientific_publication", "specification",
    "file_folder", "news_article", "budget", "invoice",
    "presentation", "questionnaire", "resume", "memo",
]

MODEL_REGISTRY: dict[str, dict] = {
    # ── SPECIALIST ─────────────────────────────────────────────────────────────
    "donut_cls": {
        "display":  "Donut-base (RVL-CDIP classification)",
        "hf_id":    "naver-clova-ix/donut-base-finetuned-rvlcdip",
        "arch":     "donut_cls",
        "tasks":    ["classification"],
        "category": "specialist", "size_m": 200,
    },
    "donut_kie": {
        "display":  "Donut-base (CORD v2 KIE)",
        "hf_id":    "naver-clova-ix/donut-base-finetuned-cord-v2",
        "arch":     "donut_kie",
        "tasks":    ["kie"],
        "category": "specialist", "size_m": 200,
    },
    "layoutlmv3": {
        "display":  "LayoutLMv3-base (FUNSD NER)",
        "hf_id":    "microsoft/layoutlmv3-base-finetuned-funsd",
        "arch":     "layoutlmv3_ner",
        "tasks":    ["ner"],
        "category": "specialist", "size_m": 125,
    },
    "layoutlmv2": {
        "display":  "LayoutLMv2-base",
        "hf_id":    "microsoft/layoutlmv2-base-uncased",
        "arch":     "stub",
        "tasks":    [],
        "category": "specialist", "size_m": 200,
        "notes": "Needs task-specific fine-tuning; no public rvlcdip checkpoint",
    },
    "bros": {
        "display":  "BROS-base",
        "hf_id":    "naver-clova-ix/bros-base-uncased",
        "arch":     "stub",
        "tasks":    [],
        "category": "specialist", "size_m": 110,
        "notes": "Needs KIE/NER fine-tuning before evaluation",
    },
    "tilt": {
        "display":  "TILT-base",
        "hf_id":    None,
        "arch":     "stub",
        "tasks":    [],
        "category": "specialist", "size_m": 230,
        "notes": "Not publicly available on HuggingFace Hub",
    },
    "dit_base": {
        "display":  "DiT-base (RVL-CDIP fine-tuned)",
        "hf_id":    "microsoft/dit-base-finetuned-rvlcdip",
        "arch":     "dit",
        "tasks":    ["classification"],
        "category": "specialist", "size_m": 86,
    },
    "dit_large": {
        "display":  "DiT-large (RVL-CDIP fine-tuned)",
        "hf_id":    "microsoft/dit-large-finetuned-rvlcdip",
        "arch":     "dit",
        "tasks":    ["classification"],
        "category": "specialist", "size_m": 307,
    },
    "beit": {
        "display":  "BEiT-base",
        "hf_id":    "microsoft/beit-base-patch16-224-pt22k-ft22k",
        "arch":     "dit",
        "tasks":    ["classification"],
        "category": "specialist", "size_m": 86,
        "notes": "ImageNet-21k model; replace hf_id with a document-tuned checkpoint for fair comparison",
    },
    "tatr": {
        "display":  "TATR — Table Transformer",
        "hf_id":    "microsoft/table-transformer-detection",
        "arch":     "stub",
        "tasks":    [],
        "category": "specialist", "size_m": 110,
        "notes": "Table task eval dataset not yet available",
    },
    "detectron2": {
        "display":  "Detectron2 / Mask R-CNN",
        "hf_id":    None,
        "arch":     "stub",
        "tasks":    [],
        "category": "specialist", "size_m": None,
        "notes": "Layout task eval dataset not yet available",
    },
    "mplug_owl": {
        "display":  "mPLUG-DocOwl-7B",
        "hf_id":    "MAGAer13/mplug-owl2-llama2-7b",
        "arch":     "stub",
        "tasks":    [],
        "category": "specialist", "size_m": 7000,
        "notes": "Requires custom inference code from MAGAer13/mPLUG-Owl repo",
    },
    "mplug_owl2": {
        "display":  "mPLUG-DocOwl2-7B",
        "hf_id":    "mPLUG/DocOwl2",
        "arch":     "stub",
        "tasks":    [],
        "category": "specialist", "size_m": 7000,
        "notes": "Requires custom inference code from mPLUG/DocOwl repo",
    },
    "textmonkey": {
        "display":  "TextMonkey-7B",
        "hf_id":    "Yuliang-Liu/TextMonkey",
        "arch":     "stub",
        "tasks":    [],
        "category": "specialist", "size_m": 7000,
        "notes": "Requires custom inference code from Yuliang-Liu/TextMonkey repo",
    },
    "vary": {
        "display":  "Vary-base (1.8B)",
        "hf_id":    "HaoranWei/Vary-base",
        "arch":     "stub",
        "tasks":    [],
        "category": "specialist", "size_m": 1800,
        "notes": "Requires Vary-specific inference code; limited standard HF support",
    },
    "bert": {
        "display":  "BERT-base-uncased",
        "hf_id":    "bert-base-uncased",
        "arch":     "bert_zs",
        "tasks":    ["classification"],
        "category": "specialist", "size_m": 110,
        "requires_ocr": True,
        "notes": "Zero-shot via [CLS] cosine similarity; needs OCR text",
    },
    "roberta": {
        "display":  "RoBERTa-base",
        "hf_id":    "roberta-base",
        "arch":     "bert_zs",
        "tasks":    ["classification"],
        "category": "specialist", "size_m": 125,
        "requires_ocr": True,
        "notes": "Zero-shot via [CLS] cosine similarity; needs OCR text",
    },
    "tableformer": {
        "display":  "TableFormer (custom)",
        "hf_id":    None,
        "arch":     "stub",
        "tasks":    [],
        "category": "specialist", "size_m": None,
        "notes": "Custom model; table task eval dataset not yet available",
    },
    "trocr_base": {
        "display":  "TrOCR-base-printed (334M)",
        "hf_id":    "microsoft/trocr-base-printed",
        "arch":     "trocr",
        "tasks":    ["ocr"],
        "category": "specialist", "size_m": 334,
    },
    "trocr_large": {
        "display":  "TrOCR-large-printed (558M)",
        "hf_id":    "microsoft/trocr-large-printed",
        "arch":     "trocr",
        "tasks":    ["ocr"],
        "category": "specialist", "size_m": 558,
    },
    "got_ocr2": {
        "display":  "GOT-OCR2.0 (580M)",
        "hf_id":    "stepfun-ai/GOT-OCR2_0",
        "arch":     "got_ocr",
        "tasks":    ["ocr"],
        "category": "specialist", "size_m": 580,
    },
    "paddleocr_vl": {
        "display":  "PaddleOCR-VL (0.9B)",
        "hf_id":    None,
        "arch":     "paddleocr_vl",
        "tasks":    ["ocr"],
        "category": "specialist", "size_m": 900,
        "notes": "Requires: pip install paddlepaddle paddleocr",
    },
    # ── GENERALIST ─────────────────────────────────────────────────────────────
    "qwen_vl_3b": {
        "display":  "Qwen2.5-VL-3B-Instruct",
        "hf_id":    "Qwen/Qwen2.5-VL-3B-Instruct",
        "arch":     "qwen_vl",
        "tasks":    ["ocr", "classification", "kie", "ner"],
        "category": "generalist", "size_m": 3000,
    },
    "qwen25vl_3b_ft": {
        "display":  "Qwen2.5-VL-3B Fine-tuned (cls-only, r=32)",
        "hf_id":    "Qwen/Qwen2.5-VL-3B-Instruct",
        "adapter_path": str(BASE_DIR / "finetune_outputs_cls" / "qwen25vl_3b" / "final_adapter"),
        "arch":     "qwen_vl_lora",
        "tasks":    ["classification"],
        "category": "finetuned", "size_m": 3000,
    },
    "qwen25vl_3b_lora_cls": {
        "display":  "Qwen2.5-VL-3B CLS Expert (lora_cls)",
        "hf_id":    "Qwen/Qwen2.5-VL-3B-Instruct",
        "adapter_path": str(BASE_DIR / "finetune_outputs" / "lora_cls" / "qwen25vl_3b" / "final_adapter"),
        "arch":     "qwen_vl_lora",
        "tasks":    ["classification"],
        "category": "finetuned", "size_m": 3000,
        "notes": "Classification-only LoRA. Trained on full RVL-CDIP (320K) + Tobacco (2437).",
    },
    "qwen25vl_3b_lora_ocr": {
        "display":  "Qwen2.5-VL-3B OCR Expert (lora_ocr)",
        "hf_id":    "Qwen/Qwen2.5-VL-3B-Instruct",
        "adapter_path": str(BASE_DIR / "finetune_outputs" / "lora_ocr" / "qwen25vl_3b" / "final_adapter"),
        "arch":     "qwen_vl_lora",
        "tasks":    ["ocr"],
        "category": "finetuned", "size_m": 3000,
        "notes": "OCR-only LoRA. Trained on full IAM (6482) + SROIE (626).",
    },
    "qwen25vl_3b_mixed_ck400": {
        "display":  "Qwen2.5-VL-3B Mixed Fine-tuned (checkpoint-400, best val)",
        "hf_id":    "Qwen/Qwen2.5-VL-3B-Instruct",
        "adapter_path": str(BASE_DIR / "finetune_outputs" / "qwen25vl_3b" / "checkpoint-400"),
        "arch":     "qwen_vl_lora",
        "tasks":    ["ocr", "classification", "kie", "ner"],
        "category": "finetuned", "size_m": 3000,
        "notes": "Mixed OCR+KIE+NER+CLS training. Best val checkpoint (step 400, loss 3.45 vs 3.93 final).",
    },
    "qwen_vl_7b": {
        "display":  "Qwen2.5-VL-7B-Instruct",
        "hf_id":    "Qwen/Qwen2.5-VL-7B-Instruct",
        "arch":     "qwen_vl",
        "tasks":    ["ocr", "classification", "kie", "ner"],
        "category": "generalist", "size_m": 7000,
    },
    "qwen35_4b": {
        "display":  "Qwen3.5-4B (vision-language)",
        "hf_id":    "Qwen/Qwen3.5-4B",
        "arch":     "qwen3_5_vl",
        "tasks":    ["ocr", "classification", "kie", "ner"],
        "category": "generalist", "size_m": 4000,
    },
    "gemma3_4b": {
        "display":  "Gemma-3-4B-IT",
        "hf_id":    "google/gemma-3-4b-it",
        "arch":     "gemma_vl",
        "tasks":    ["ocr", "classification", "kie", "ner"],
        "category": "generalist", "size_m": 4000,
    },
    "gemma4_e4b": {
        "display":  "Gemma-4-E4B-IT",
        "hf_id":    "google/gemma-4-e4b-it",
        "arch":     "gemma_vl",
        "tasks":    ["ocr", "classification", "kie", "ner"],
        "category": "generalist", "size_m": 4000,
    },
    "phi_vision": {
        "display":  "Phi-3.5-Vision-Instruct (4.2B)",
        "hf_id":    "microsoft/Phi-3.5-vision-instruct",
        "arch":     "phi_vl",
        "tasks":    ["ocr", "classification", "kie", "ner"],
        "category": "generalist", "size_m": 4200,
    },
    "phi_mini": {
        "display":  "Phi-3.5-Mini-Instruct (3.8B, text)",
        "hf_id":    "microsoft/Phi-3.5-mini-instruct",
        "arch":     "text_llm",
        "tasks":    ["classification", "kie", "ner"],
        "category": "generalist", "size_m": 3800,
        "requires_ocr": True,
    },
    "mistral_7b": {
        "display":  "Mistral-7B-Instruct (text, post-OCR)",
        "hf_id":    "mistralai/Mistral-7B-Instruct-v0.3",
        "arch":     "text_llm",
        "tasks":    ["classification", "kie", "ner"],
        "category": "generalist", "size_m": 7000,
        "requires_ocr": True,
    },
    "llama31_7b": {
        "display":  "Llama-3.1-8B-Instruct (text, post-OCR)",
        "hf_id":    "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "arch":     "text_llm",
        "tasks":    ["classification", "kie", "ner"],
        "category": "generalist", "size_m": 8000,
        "requires_ocr": True,
    },
    "llama32_3b": {
        "display":  "Llama-3.2-3B-Instruct (text, post-OCR)",
        "hf_id":    "meta-llama/Llama-3.2-3B-Instruct",
        "arch":     "text_llm",
        "tasks":    ["classification", "kie", "ner"],
        "category": "generalist", "size_m": 3000,
        "requires_ocr": True,
    },
    "llama32_11b": {
        "display":  "Llama-3.2-11B-Vision-Instruct",
        "hf_id":    "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "arch":     "llama_vl",
        "tasks":    ["ocr", "classification", "kie", "ner"],
        "category": "generalist", "size_m": 11000,
    },
    "internvl2_4b": {
        "display":  "InternVL2-4B",
        "hf_id":    "OpenGVLab/InternVL2-4B",
        "arch":     "internvl",
        "tasks":    ["ocr", "classification", "kie", "ner"],
        "category": "generalist", "size_m": 4000,
    },
    "llava15_7b": {
        "display":  "LLaVA-1.5-7B",
        "hf_id":    "llava-hf/llava-1.5-7b-hf",
        "arch":     "llava",
        "tasks":    ["ocr", "classification", "kie", "ner"],
        "category": "generalist", "size_m": 7000,
    },
    "llava_next_7b": {
        "display":  "LLaVA-Next-7B",
        "hf_id":    "llava-hf/llava-v1.6-vicuna-7b-hf",
        "arch":     "llava_next",
        "tasks":    ["ocr", "classification", "kie", "ner"],
        "category": "generalist", "size_m": 7000,
    },
    "qwen3_4b": {
        "display":  "Qwen3-4B (text, post-OCR)",
        "hf_id":    "Qwen/Qwen3-4B",
        "arch":     "text_llm",
        "tasks":    ["classification", "kie", "ner"],
        "category": "generalist", "size_m": 4000,
        "requires_ocr": True,
    },
    "paligemma": {
        "display":  "PaliGemma-3B",
        "hf_id":    "google/paligemma-3b-mix-448",
        "arch":     "paligemma",
        "tasks":    ["ocr", "classification", "kie", "ner"],
        "category": "generalist", "size_m": 3000,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# TASK PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

PROMPTS = {
    "ocr": (
        "Transcribe all text visible in this document image exactly as it appears, "
        "preserving line breaks and structure."
    ),
    "classification": (
        "Classify this document into exactly one of these categories: {classes}. "
        "Reply with only the category name, nothing else."
    ),
    "kie": (
        "Extract all key-value information fields from this document. "
        "Return a JSON object mapping field names to their values. "
        "Return only valid JSON, no explanation."
    ),
    "ner": (
        "Identify named entities in this document. "
        "Return a JSON list: [{\"entity\":\"...\",\"type\":\"...\",\"start\":0,\"end\":0}]. "
        "Types: ANSWER, QUESTION, HEADER, OTHER. Return only valid JSON."
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_easyocr_reader = None

def _get_ocr_text(image: Image.Image, words: Optional[list] = None) -> str:
    if words:
        return " ".join(str(w) for w in words)
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    import numpy as np
    results = _easyocr_reader.readtext(np.array(image))
    return " ".join(r[1] for r in results)


def _closest_class(pred: str, classes: list[str]) -> str:
    p = pred.lower().strip()
    for cls in classes:
        if cls.lower() == p:
            return cls
    for cls in classes:
        if cls.lower() in p or p in cls.lower():
            return cls
    return p


def _bio_to_spans(words: list, tags: list) -> list[dict]:
    spans, i = [], 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{etype}":
                j += 1
            spans.append({"entity": " ".join(words[i:j]),
                          "type": etype, "start": i, "end": j - 1})
            i = j
        else:
            i += 1
    return spans


def _parse_json_response(text: str, fallback):
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        # Try extracting first {...} or [...]
        for pat in (r"\{.*\}", r"\[.*\]"):
            m = re.search(pat, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except Exception:
                    pass
    return fallback


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS  (return a "loaded" dict passed to runners)
# ══════════════════════════════════════════════════════════════════════════════

def _quant_kwargs(size_m: int = 0, force: bool = False) -> dict:
    """Return 4-bit NF4 config when the model footprint exceeds 50% of total VRAM, or when forced."""
    if not torch.cuda.is_available():
        return {}
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    model_bf16_gb = size_m * 2 / 1000  # bfloat16: 2 bytes per param
    if force or model_bf16_gb > total_gb * 0.50:
        try:
            from transformers import BitsAndBytesConfig
            reason = "forced" if force else f"~{model_bf16_gb:.1f} GB > 50% of {total_gb:.1f} GB VRAM"
            print(f"  [quant] {reason} — using 4-bit NF4")
            return {"quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")}
        except ImportError:
            print("  [WARN] bitsandbytes not installed — model will offload to CPU. "
                  "Fix: pip install bitsandbytes")
    return {}


def _run_with_timeout(fn, timeout_sec: float, *args, **kwargs):
    """Run fn(*args, **kwargs) in a thread; raise TimeoutError without blocking."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = executor.submit(fn, *args, **kwargs)
    try:
        return fut.result(timeout=timeout_sec)
    except concurrent.futures.TimeoutError:
        fut.cancel()
        executor.shutdown(wait=False)  # abandon the thread — don't block
        raise
    except Exception:
        executor.shutdown(wait=False)
        raise
    finally:
        # normal completion path
        try:
            executor.shutdown(wait=False)
        except Exception:
            pass


def _load_qwen_vl(cfg: dict) -> dict:
    from transformers import AutoProcessor
    qkw = _quant_kwargs(cfg.get("size_m", 0))
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg["hf_id"], torch_dtype=torch.bfloat16, device_map="auto", **qkw)
    except (ImportError, AttributeError):
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg["hf_id"], torch_dtype=torch.bfloat16, device_map="auto", **qkw)
    # cap image tokens: 256*28*28 ≈ 200K pixels max per image to save VRAM
    proc = AutoProcessor.from_pretrained(
        cfg["hf_id"], min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
    return {"model": model, "processor": proc}


def _load_qwen_vl_lora(cfg: dict) -> dict:
    """Load Qwen2.5-VL base + QLoRA adapter from a training checkpoint."""
    from transformers import AutoProcessor
    from peft import PeftModel

    base_id    = cfg["hf_id"]
    adapter_path = cfg["adapter_path"]
    qkw = _quant_kwargs(cfg.get("size_m", 0))

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_id, torch_dtype=torch.bfloat16, device_map="auto", **qkw)
    except (ImportError, AttributeError):
        from transformers import Qwen2VLForConditionalGeneration
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            base_id, torch_dtype=torch.bfloat16, device_map="auto", **qkw)

    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    proc = AutoProcessor.from_pretrained(
        base_id, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
    return {"model": model, "processor": proc}


def _load_qwen3_5_vl(cfg: dict) -> dict:
    from transformers import AutoProcessor
    # Try known class names in order; fall back to generic causal LM
    for cls_name in ("Qwen2_5_VLForConditionalGeneration", "Qwen2VLForConditionalGeneration"):
        try:
            import importlib
            cls = getattr(importlib.import_module("transformers"), cls_name)
            model = cls.from_pretrained(cfg["hf_id"], torch_dtype=torch.bfloat16, device_map="auto")
            proc = AutoProcessor.from_pretrained(cfg["hf_id"])
            return {"model": model, "processor": proc}
        except (ImportError, AttributeError, OSError):
            continue
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(cfg["hf_id"])
    return {"model": model, "processor": proc}


def _load_gemma_vl(cfg: dict) -> dict:
    from transformers import AutoModelForCausalLM, AutoProcessor
    # Gemma 4 MoE models use ~14GB+ at bf16 during inference — force 4-bit quant
    force_quant = "gemma-4" in cfg.get("hf_id", "")
    qkw = _quant_kwargs(cfg.get("size_m", 0), force=force_quant)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"], torch_dtype=torch.bfloat16, device_map="auto", **qkw)
    proc = AutoProcessor.from_pretrained(cfg["hf_id"])
    return {"model": model, "processor": proc}


def _load_phi_vl(cfg: dict) -> dict:
    from transformers import AutoModelForCausalLM, AutoProcessor
    qkw = _quant_kwargs(cfg.get("size_m", 0))
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, _attn_implementation="eager", **qkw)
    proc = AutoProcessor.from_pretrained(cfg["hf_id"], trust_remote_code=True)
    return {"model": model, "processor": proc}


def _load_llama_vl(cfg: dict) -> dict:
    from transformers import MllamaForConditionalGeneration, AutoProcessor
    qkw = _quant_kwargs(cfg.get("size_m", 0))
    model = MllamaForConditionalGeneration.from_pretrained(
        cfg["hf_id"], torch_dtype=torch.bfloat16, device_map="auto", **qkw)
    proc = AutoProcessor.from_pretrained(cfg["hf_id"])
    return {"model": model, "processor": proc}


def _load_internvl(cfg: dict) -> dict:
    from transformers import AutoModel, AutoTokenizer
    qkw = _quant_kwargs(cfg.get("size_m", 0))
    model = AutoModel.from_pretrained(
        cfg["hf_id"], torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="auto", **qkw).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"], trust_remote_code=True)
    return {"model": model, "tokenizer": tokenizer}


def _load_llava(cfg: dict) -> dict:
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    model = LlavaForConditionalGeneration.from_pretrained(
        cfg["hf_id"], torch_dtype=torch.float16, device_map="auto")
    proc = AutoProcessor.from_pretrained(cfg["hf_id"])
    return {"model": model, "processor": proc}


def _load_llava_next(cfg: dict) -> dict:
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
    model = LlavaNextForConditionalGeneration.from_pretrained(
        cfg["hf_id"], torch_dtype=torch.float16, device_map="auto")
    proc = LlavaNextProcessor.from_pretrained(cfg["hf_id"])
    return {"model": model, "processor": proc}


def _load_paligemma(cfg: dict) -> dict:
    from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        cfg["hf_id"], torch_dtype=torch.bfloat16, device_map="auto")
    proc = AutoProcessor.from_pretrained(cfg["hf_id"])
    return {"model": model, "processor": proc}


def _load_text_llm(cfg: dict) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["hf_id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"], torch_dtype=torch.float16, device_map="auto")
    return {"model": model, "tokenizer": tok}


def _load_dit(cfg: dict) -> dict:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    feat  = AutoImageProcessor.from_pretrained(cfg["hf_id"])
    model = AutoModelForImageClassification.from_pretrained(cfg["hf_id"]).to("cuda")
    model.eval()
    return {"model": model, "processor": feat}


def _load_bert_zs(cfg: dict) -> dict:
    from transformers import AutoModel, AutoTokenizer
    tok   = AutoTokenizer.from_pretrained(cfg["hf_id"])
    model = AutoModel.from_pretrained(cfg["hf_id"]).to("cuda")
    model.eval()
    return {"model": model, "tokenizer": tok}


def _load_donut(cfg: dict) -> dict:
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    proc  = DonutProcessor.from_pretrained(cfg["hf_id"])
    model = VisionEncoderDecoderModel.from_pretrained(cfg["hf_id"]).to("cuda")
    model.eval()
    return {"model": model, "processor": proc}


def _load_layoutlmv3_ner(cfg: dict) -> dict:
    from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
    proc  = LayoutLMv3Processor.from_pretrained(cfg["hf_id"], apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(cfg["hf_id"]).to("cuda")
    model.eval()
    return {"model": model, "processor": proc}


def _load_trocr(cfg: dict) -> dict:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    proc  = TrOCRProcessor.from_pretrained(cfg["hf_id"])
    model = VisionEncoderDecoderModel.from_pretrained(cfg["hf_id"]).to("cuda")
    model.eval()
    return {"model": model, "processor": proc}


def _load_got_ocr(cfg: dict) -> dict:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok   = AutoTokenizer.from_pretrained(cfg["hf_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"], torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True).eval()
    return {"model": model, "tokenizer": tok}


def _load_paddleocr_vl(cfg: dict) -> dict:
    import importlib
    if importlib.util.find_spec("paddleocr") is None:
        raise ImportError("paddleocr not installed — run: pip install paddlepaddle paddleocr")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return {"ocr": ocr}


# ── Loader dispatch ────────────────────────────────────────────────────────────
ARCH_LOADERS = {
    "qwen_vl":        _load_qwen_vl,
    "qwen_vl_lora":   _load_qwen_vl_lora,
    "qwen3_5_vl":     _load_qwen3_5_vl,
    "gemma_vl":       _load_gemma_vl,
    "phi_vl":         _load_phi_vl,
    "llama_vl":       _load_llama_vl,
    "internvl":       _load_internvl,
    "llava":          _load_llava,
    "llava_next":     _load_llava_next,
    "paligemma":      _load_paligemma,
    "text_llm":       _load_text_llm,
    "dit":            _load_dit,
    "bert_zs":        _load_bert_zs,
    "donut_cls":      _load_donut,
    "donut_kie":      _load_donut,
    "layoutlmv3_ner": _load_layoutlmv3_ner,
    "trocr":          _load_trocr,
    "got_ocr":        _load_got_ocr,
    "paddleocr_vl":   _load_paddleocr_vl,
}


def _load_model(key: str, cfg: dict) -> dict:
    arch = cfg["arch"]
    if arch not in ARCH_LOADERS:
        raise ValueError(f"No loader for arch={arch}")
    print(f"  Loading {cfg['display']} ...", flush=True)
    loaded = ARCH_LOADERS[arch](cfg)
    loaded["_arch"] = arch
    loaded["_cfg"]  = cfg
    return loaded


def _unload_model(loaded: dict):
    for k in ("model", "processor", "tokenizer"):
        if k in loaded:
            del loaded[k]
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass  # CUDA context may already be corrupted — safe to ignore here


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE RUNNERS  (each returns a plain str)
# ══════════════════════════════════════════════════════════════════════════════

def _run_qwen_vl(loaded: dict, image: Image.Image, instruction: str,
                 max_tokens: int = 256, **_) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text":  instruction},
    ]}]
    proc = loaded["processor"]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = loaded["model"].generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    decoded = proc.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return decoded.strip()


def _run_qwen3_5_vl(loaded: dict, image: Image.Image, instruction: str,
                    max_tokens: int = 256, **_) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text":  instruction},
    ]}]
    proc = loaded["processor"]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = loaded["model"].generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return proc.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def _run_gemma_vl(loaded: dict, image: Image.Image, instruction: str,
                  max_tokens: int = 256, **_) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text":  instruction},
    ]}]
    proc = loaded["processor"]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=text, images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = loaded["model"].generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return proc.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def _run_phi_vl(loaded: dict, image: Image.Image, instruction: str,
                max_tokens: int = 256, **_) -> str:
    proc = loaded["processor"]
    messages = [{"role": "user", "content": f"<|image_1|>\n{instruction}"}]
    prompt = proc.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(prompt, [image], return_tensors="pt").to("cuda")
    eos_id  = proc.tokenizer.eos_token_id
    with torch.no_grad():
        out = loaded["model"].generate(**inputs, max_new_tokens=max_tokens,
                                       eos_token_id=eos_id, do_sample=False, use_cache=False)
    return proc.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def _run_llama_vl(loaded: dict, image: Image.Image, instruction: str,
                  max_tokens: int = 256, **_) -> str:
    proc = loaded["processor"]
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction},
    ]}]
    text   = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(images=image, text=text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = loaded["model"].generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _run_internvl(loaded: dict, image: Image.Image, instruction: str,
                  max_tokens: int = 256, **_) -> str:
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).to("cuda")
    gen_cfg = {"max_new_tokens": max_tokens, "do_sample": False}
    response = loaded["model"].chat(
        loaded["tokenizer"], pixel_values, f"<image>\n{instruction}", gen_cfg)
    return response.strip()


def _run_llava(loaded: dict, image: Image.Image, instruction: str,
               max_tokens: int = 256, **_) -> str:
    proc = loaded["processor"]
    conv = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction},
    ]}]
    prompt = proc.apply_chat_template(conv, add_generation_prompt=True)
    inputs = proc(images=image, text=prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = loaded["model"].generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return proc.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def _run_llava_next(loaded: dict, image: Image.Image, instruction: str,
                    max_tokens: int = 256, **_) -> str:
    proc = loaded["processor"]
    conv = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction},
    ]}]
    prompt = proc.apply_chat_template(conv, add_generation_prompt=True)
    inputs = proc(images=image, text=prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = loaded["model"].generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return proc.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def _run_paligemma(loaded: dict, image: Image.Image, instruction: str,
                   max_tokens: int = 256, **_) -> str:
    proc   = loaded["processor"]
    inputs = proc(text=instruction, images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = loaded["model"].generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return proc.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def _run_text_llm(loaded: dict, text_context: str, instruction: str,
                  max_tokens: int = 256, **_) -> str:
    tok  = loaded["tokenizer"]
    body = f"Document text:\n{text_context[:2000]}\n\n{instruction}"
    msgs = [{"role": "user", "content": body}]
    if hasattr(tok, "apply_chat_template"):
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        prompt = body
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    with torch.no_grad():
        out = loaded["model"].generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def _run_dit(loaded: dict, image: Image.Image, **_) -> str:
    inputs = loaded["processor"](images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = loaded["model"](**inputs).logits
    pred_id = logits.argmax(-1).item()
    label   = loaded["model"].config.id2label.get(pred_id, str(pred_id))
    return label.lower().replace(" ", "_").replace("-", "_")


def _run_bert_zs(loaded: dict, text_context: str, classes: list[str], **_) -> str:
    tok   = loaded["tokenizer"]
    model = loaded["model"]

    def _embed(text: str) -> torch.Tensor:
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=512, padding=True).to("cuda")
        with torch.no_grad():
            return model(**enc).last_hidden_state[:, 0, :]  # [CLS]

    doc_emb = _embed(text_context[:512])
    best_cls, best_sim = "", -1.0
    for cls in classes:
        cls_emb = _embed(f"This document is a {cls}.")
        sim = torch.cosine_similarity(doc_emb, cls_emb).item()
        if sim > best_sim:
            best_sim = sim
            best_cls = cls
    return best_cls


def _run_donut_cls(loaded: dict, image: Image.Image, **_) -> str:
    proc  = loaded["processor"]
    model = loaded["model"]
    task_prompt = "<s_rvlcdip>"
    decoder_ids = proc.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    pixel_values = proc(image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad():
        out = model.generate(
            pixel_values, decoder_input_ids=decoder_ids,
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=proc.tokenizer.pad_token_id,
            eos_token_id=proc.tokenizer.eos_token_id)
    seq = proc.batch_decode(out.tolist())[0]
    seq = seq.replace(proc.tokenizer.eos_token, "").replace(proc.tokenizer.pad_token, "")
    m = re.search(r"<s_class_type>(.*?)</s_class_type>", seq)
    return m.group(1).strip() if m else re.sub(r"<.*?>", "", seq).strip()


def _run_donut_kie(loaded: dict, image: Image.Image, **_) -> str:
    proc  = loaded["processor"]
    model = loaded["model"]
    task_prompt = "<s_cord-v2>"
    decoder_ids = proc.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    pixel_values = proc(image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad():
        out = model.generate(
            pixel_values, decoder_input_ids=decoder_ids,
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=proc.tokenizer.pad_token_id,
            eos_token_id=proc.tokenizer.eos_token_id)
    seq = proc.batch_decode(out.tolist())[0]
    seq = seq.replace(proc.tokenizer.eos_token, "").replace(proc.tokenizer.pad_token, "")
    parsed = proc.token2json(seq)
    flat: dict = {}
    def _flatten(d, prefix=""):
        for k, v in (d.items() if isinstance(d, dict) else {}.items()):
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, dict):
                _flatten(v, key)
            elif isinstance(v, list):
                flat[key] = str(v)
            else:
                flat[key] = str(v)
    _flatten(parsed)
    return json.dumps(flat, ensure_ascii=False)


def _run_layoutlmv3_ner(loaded: dict, image: Image.Image,
                        words: Optional[list], bboxes: Optional[list], **_) -> str:
    if not words:
        return "[]"
    proc  = loaded["processor"]
    model = loaded["model"]
    inputs = proc(
        image, words, boxes=bboxes,
        return_tensors="pt", truncation=True, padding="max_length", max_length=512
    ).to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    pred_ids = logits.argmax(-1).cpu().numpy()
    id2label = model.config.id2label
    pred_tags = [id2label.get(int(p), "O") for p in pred_ids[:len(words)]]
    spans = _bio_to_spans(words, pred_tags)
    return json.dumps(spans, ensure_ascii=False)


def _run_trocr(loaded: dict, image: Image.Image, **_) -> str:
    proc  = loaded["processor"]
    model = loaded["model"]
    pixel_values = proc(images=image.convert("RGB"), return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=256)
    return proc.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def _run_got_ocr(loaded: dict, image: Image.Image, **_) -> str:
    import tempfile, os
    model = loaded["model"]
    tok   = loaded["tokenizer"]
    # GOT-OCR2 requires a file path; save image to a temp file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        image.convert("RGB").save(tmp_path)
        result = model.chat(tok, tmp_path, ocr_type="ocr")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return result.strip() if isinstance(result, str) else str(result).strip()


def _run_paddleocr_vl(loaded: dict, image: Image.Image, **_) -> str:
    import numpy as np
    ocr    = loaded["ocr"]
    result = ocr.ocr(np.array(image.convert("RGB")), cls=True)
    if not result or not result[0]:
        return ""
    return "\n".join(line[1][0] for line in result[0] if line and line[1])


# ── Runner dispatch ────────────────────────────────────────────────────────────
ARCH_RUNNERS = {
    "qwen_vl":        _run_qwen_vl,
    "qwen_vl_lora":   _run_qwen_vl,   # same inference path as base qwen_vl
    "qwen3_5_vl":     _run_qwen3_5_vl,
    "gemma_vl":       _run_gemma_vl,
    "phi_vl":         _run_phi_vl,
    "llama_vl":       _run_llama_vl,
    "internvl":       _run_internvl,
    "llava":          _run_llava,
    "llava_next":     _run_llava_next,
    "paligemma":      _run_paligemma,
    "text_llm":       _run_text_llm,
    "dit":            _run_dit,
    "bert_zs":        _run_bert_zs,
    "donut_cls":      _run_donut_cls,
    "donut_kie":      _run_donut_kie,
    "layoutlmv3_ner": _run_layoutlmv3_ner,
    "trocr":          _run_trocr,
    "got_ocr":        _run_got_ocr,
    "paddleocr_vl":   _run_paddleocr_vl,
}


def _run_one(loaded: dict, image: Image.Image, instruction: str,
             text_context: str, words: Optional[list],
             bboxes: Optional[list], max_tokens: int,
             classes: Optional[list] = None) -> str:
    arch   = loaded["_arch"]
    runner = ARCH_RUNNERS[arch]
    return runner(loaded, image=image, instruction=instruction,
                  text_context=text_context, words=words,
                  bboxes=bboxes, max_tokens=max_tokens, classes=classes)


# ══════════════════════════════════════════════════════════════════════════════
# PER-TASK EVALUATORS
# ══════════════════════════════════════════════════════════════════════════════

def _load_eval_records(task: str, max_n: Optional[int]) -> list[dict]:
    path = EVAL_DIR / task / f"eval_test.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                if max_n and len(records) >= max_n:
                    break
    return records


def evaluate_task(model_key: str, task: str, loaded: dict,
                  cfg: dict, max_n: Optional[int]) -> dict:
    from build_eval_datasets import (
        ocr_metrics, classification_metrics, kie_metrics, ner_metrics)

    records = _load_eval_records(task, max_n)
    if not records:
        return {"status": "no_eval_file", "count": 0}

    predictions: list[str] = []
    timings_ms:  list[float] = []
    gpu_gb_list: list[float] = []
    errors = 0

    # IAM line OCR needs ≤80 tokens; 256 is still generous and halves KV-cache vs 512
    max_tokens_for_task = 256 if task == "ocr" else 128
    t_loop_start = time.perf_counter()

    for idx, rec in enumerate(records):
        img_abs = BASE_DIR / rec["image_path"]
        try:
            image = Image.open(img_abs).convert("RGB")
        except Exception:
            errors += 1
            predictions.append("")
            timings_ms.append(0.0)
            gpu_gb_list.append(0.0)
            continue

        # Build instruction
        if task == "classification":
            classes = rec.get("all_classes", RVL_CLASSES)
            instruction = PROMPTS["classification"].format(classes=", ".join(classes))
        else:
            instruction = PROMPTS[task]

        text_context = ""
        if cfg.get("requires_ocr"):
            text_context = _get_ocr_text(image, rec.get("words"))

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            pred = _run_with_timeout(
                _run_one, SAMPLE_TIMEOUT_SEC,
                loaded, image, instruction, text_context,
                rec.get("words"), rec.get("bboxes"), max_tokens_for_task,
                classes if task == "classification" else None,
            )
        except concurrent.futures.TimeoutError:
            print(f"    [TIMEOUT] sample {rec['id']} exceeded {SAMPLE_TIMEOUT_SEC}s — skipping")
            pred = ""
            errors += 1
        except Exception as exc:
            print(f"    [WARN] inference error sample {rec['id']}: {exc}")
            pred = ""
            errors += 1
            # CUDA context corruption: once broken, every subsequent call will also fail.
            # Bail out early, fill remaining slots with empty predictions, and let the
            # next model start in a fresh state.
            if "CUDA error" in str(exc) or "CUDA out of memory" in str(exc):
                remaining = len(records) - idx - 1
                print(f"    [ERROR] CUDA context corrupted — aborting task, skipping {remaining} remaining samples")
                predictions.extend([""] * remaining)
                timings_ms.extend([0.0] * remaining)
                gpu_gb_list.extend([0.0] * remaining)
                errors += remaining
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000
        peak_gb    = (torch.cuda.max_memory_allocated() / 1e9
                      if torch.cuda.is_available() else 0.0)

        predictions.append(pred)
        timings_ms.append(elapsed_ms)
        gpu_gb_list.append(peak_gb)

        # Periodic GPU memory clearing to prevent fragmentation over long eval runs
        if (idx + 1) % 50 == 0 and torch.cuda.is_available():
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Progress every 25 samples
        if (idx + 1) % 25 == 0 or (idx + 1) == len(records):
            done = idx + 1
            elapsed_total = time.perf_counter() - t_loop_start
            rate = done / elapsed_total if elapsed_total > 0 else 0
            eta  = (len(records) - done) / rate if rate > 0 else 0
            print(f"    [{task}] {done}/{len(records)}  "
                  f"{elapsed_ms:.0f}ms/sample  "
                  f"ETA {eta/60:.1f}min  errors={errors}", flush=True)

    # ── Score ────────────────────────────────────────────────────────────────
    if task == "ocr":
        scores = ocr_metrics(predictions, records)

    elif task == "classification":
        norm_preds = [_closest_class(p, rec.get("all_classes", RVL_CLASSES))
                      for p, rec in zip(predictions, records)]
        scores = classification_metrics(norm_preds, records)

    elif task == "kie":
        parsed: list[dict] = []
        for p in predictions:
            parsed.append(_parse_json_response(p, {}) or {})
        scores = kie_metrics(parsed, records)

    elif task == "ner":
        parsed_spans: list[list] = []
        for p in predictions:
            result = _parse_json_response(p, [])
            parsed_spans.append(result if isinstance(result, list) else [])
        scores = ner_metrics(parsed_spans, records)

    else:
        scores = {}

    n = len(timings_ms)
    scores.setdefault("aggregate", {}).update({
        "mean_inference_ms": round(sum(timings_ms) / n, 1) if n else 0.0,
        "max_gpu_gb":        round(max(gpu_gb_list), 3) if gpu_gb_list else 0.0,
        "n_errors":          errors,
        "n_samples":         n,
    })
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT & RESULTS I/O
# ══════════════════════════════════════════════════════════════════════════════

def _load_checkpoint() -> dict:
    if CKPT_FILE.exists():
        with open(CKPT_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_checkpoint(ckpt: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CKPT_FILE, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, indent=2)


def _save_task_results(model_key: str, task: str, scores: dict):
    out_dir = RESULTS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{task}_results.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)


def _print_scores(model_key: str, task: str, scores: dict):
    agg = scores.get("aggregate", {})
    parts = [f"{k}={v}" for k, v in agg.items()
             if not k.startswith("n_") and isinstance(v, (int, float))]
    print(f"    [{task}] " + "  ".join(parts[:6]))


def _save_summary_csv(all_scores: dict):
    """Write eval_results/summary.csv — one row per (model, task)."""
    rows = []
    for model_key, task_scores in all_scores.items():
        cfg = MODEL_REGISTRY.get(model_key, {})
        for task, scores in task_scores.items():
            agg = scores.get("aggregate", {})
            row = {
                "model_key":  model_key,
                "display":    cfg.get("display", model_key),
                "category":   cfg.get("category", ""),
                "size_m":     cfg.get("size_m", ""),
                "task":       task,
                **{k: v for k, v in agg.items() if isinstance(v, (int, float, str))},
            }
            rows.append(row)

    if not rows:
        return
    fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    with open(RESULTS_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Summary CSV: {RESULTS_DIR / 'summary.csv'}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _print_registry():
    print(f"\n{'─'*72}")
    print(f"  {'Key':<18} {'Display':<38} {'Arch':<18} {'Tasks'}")
    print(f"{'─'*72}")
    for key, cfg in MODEL_REGISTRY.items():
        stub  = " [STUB]" if cfg["arch"] == "stub" else ""
        tasks = ", ".join(cfg["tasks"]) or "—"
        print(f"  {key:<18} {cfg['display'][:36]:<38} {cfg['arch']:<18} {tasks}{stub}")
    print(f"{'─'*72}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models on eval_datasets/")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model keys to run (default: all non-stub)")
    parser.add_argument("--tasks",  nargs="*", default=None,
                        choices=["ocr", "classification", "kie", "ner"])
    parser.add_argument("--max",    type=int, default=500,
                        help="Max records per task per model (default 500)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed (model, task) pairs")
    parser.add_argument("--list",   action="store_true",
                        help="Print model registry and exit")
    args = parser.parse_args()

    if args.list:
        _print_registry()
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = _load_checkpoint() if args.resume else {}

    # Select models
    if args.models:
        model_keys = args.models
    else:
        model_keys = [k for k, v in MODEL_REGISTRY.items() if v["arch"] != "stub"]

    # Pre-populate all_scores from existing result files so --resume produces a
    # complete CSV, not just results from the current session.
    all_scores: dict = {}
    if args.resume:
        for model_dir in sorted(RESULTS_DIR.iterdir()):
            if not model_dir.is_dir():
                continue
            for result_file in sorted(model_dir.glob("*_results.json")):
                task = result_file.stem.replace("_results", "")
                mkey = model_dir.name
                all_scores.setdefault(mkey, {})[task] = json.loads(result_file.read_text())

    t_total = time.time()

    for model_key in model_keys:
        cfg = MODEL_REGISTRY.get(model_key)
        if cfg is None:
            print(f"[WARN] Unknown model key: {model_key}")
            continue
        if cfg["arch"] == "stub":
            note = cfg.get("notes", "not implemented")
            print(f"\n[SKIP] {cfg['display']} — {note}")
            continue

        # Which tasks to run for this model
        run_tasks = args.tasks if args.tasks else cfg["tasks"]
        run_tasks = [t for t in run_tasks if t in cfg["tasks"]]
        if not run_tasks:
            print(f"\n[SKIP] {cfg['display']} — no matching tasks")
            continue

        print(f"\n{'='*65}")
        print(f"  {cfg['display']}  [{cfg['category']}, {cfg.get('size_m','?')}M params]")
        print(f"  Tasks: {run_tasks}   max_samples: {args.max}")
        print(f"{'='*65}")

        # Check if all tasks already done
        if args.resume and all(
            checkpoint.get(f"{model_key}/{t}", {}).get("status") == "complete"
            for t in run_tasks
        ):
            print(f"  All tasks already complete — skipping.")
            continue

        # Load model once for all tasks
        try:
            loaded = _load_model(model_key, cfg)
        except Exception as exc:
            print(f"  [ERROR] Load failed: {exc}")
            traceback.print_exc()
            continue

        model_scores: dict = {}
        for task in run_tasks:
            ck_key = f"{model_key}/{task}"
            if args.resume and checkpoint.get(ck_key, {}).get("status") == "complete":
                print(f"  [{task}] Already complete — skipping")
                continue

            print(f"\n  [{task}] Evaluating {len(_load_eval_records(task, args.max))} records ...")
            try:
                scores = evaluate_task(model_key, task, loaded, cfg, args.max)
                model_scores[task] = scores
                _save_task_results(model_key, task, scores)
                _print_scores(model_key, task, scores)
                checkpoint[ck_key] = {
                    "status": "complete",
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                _save_checkpoint(checkpoint)
            except Exception as exc:
                print(f"  [ERROR] {task}: {exc}")
                traceback.print_exc()

        all_scores[model_key] = model_scores
        _unload_model(loaded)

    # Final summary
    _save_summary_csv(all_scores)
    print(f"\n{'='*65}")
    print(f"  Done in {time.time() - t_total:.0f}s")
    print(f"  Results : {RESULTS_DIR}")
    print(f"  Resume  : python evaluate_all_models.py --resume")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()

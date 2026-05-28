#!/usr/bin/env python3
"""
download_and_format_data.py
Downloads datasets from HuggingFace and formats them into the unified JSONL
schema used by finetune_qlora.py and evaluate_all_models.py.

Supports resume — a checkpoint file tracks the last completed record index
per (dataset, split) pair. Re-running is always safe.

Usage:
  python download_and_format_data.py                       # all datasets
  python download_and_format_data.py --datasets rvlcdip    # specific dataset
  python download_and_format_data.py --resume              # skip completed
  python download_and_format_data.py --max 5000            # cap records (debug)
  python download_and_format_data.py --list                # show available datasets
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.resolve()
DATA_DIR    = BASE_DIR / "formatted_data"
LOG_DIR     = BASE_DIR / "logs"
CKPT_FILE   = DATA_DIR / "download_format_checkpoint.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "download_format.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Label definitions ──────────────────────────────────────────────────────────
RVL_CLASSES = [
    "letter", "form", "email", "handwritten", "advertisement",
    "scientific_report", "scientific_publication", "specification",
    "file_folder", "news_article", "budget", "invoice",
    "presentation", "questionnaire", "resume", "memo",
]

TOBACCO_CLASSES = [
    "ADVE", "Email", "Form", "Letter", "Memo",
    "News", "Note", "Report", "Resume", "Scientific",
]

# ── Instruction templates ──────────────────────────────────────────────────────
OCR_INSTRUCTION = (
    "Transcribe all text visible in this document image exactly as it appears, "
    "preserving line breaks and structure."
)
CLS_INSTRUCTION_TEMPLATE = (
    "Classify this document into exactly one of these categories: {classes}. "
    "Reply with only the category name, nothing else."
)
KIE_INSTRUCTION = (
    "Extract all key-value information fields from this document. "
    "Return a JSON object mapping field names to their values. "
    "Return only valid JSON, no explanation."
)
NER_INSTRUCTION = (
    "Identify named entities in this document. "
    "Return a JSON list: [{\"entity\":\"...\",\"type\":\"...\",\"start\":0,\"end\":0}]. "
    "Types: ANSWER, QUESTION, HEADER, OTHER. Return only valid JSON."
)

# ── Dataset registry ───────────────────────────────────────────────────────────
DATASETS = {
    "rvlcdip": {
        "label":    "RVL-CDIP — document classification (16 classes, 320K)",
        "hf_id":    "aharley/rvl_cdip",
        "task":     "classification",
        "splits":   ["train", "validation", "test"],
        "streaming": False,      # full download — needed for 320K images
    },
    "iam": {
        "label":    "IAM Handwriting — OCR (line level)",
        "hf_id":    "Teklia/IAM-line",
        "task":     "ocr",
        "splits":   ["train", "validation", "test"],
        "streaming": False,
    },
    "sroie": {
        "label":    "SROIE v2 — receipt OCR / KIE",
        "hf_id":    "rth/sroie-2019-v2",
        "task":     "ocr",
        "splits":   ["train", "test"],
        "streaming": False,
    },
    "cord": {
        "label":    "CORD v2 — receipt KIE",
        "hf_id":    "naver-clova-ix/cord-v2",
        "task":     "kie",
        "splits":   ["train", "validation", "test"],
        "streaming": False,
    },
    "funsd": {
        "label":    "FUNSD — form NER",
        "hf_id":    "nielsr/funsd",
        "task":     "ner",
        "splits":   ["train", "test"],
        "streaming": False,
    },
    "tobacco": {
        "label":    "Tobacco3482 — classification (10 classes)",
        "hf_id":    "maveriq/tobacco3482",
        "task":     "classification",
        "splits":   ["train", "test"],
        "streaming": False,
    },
}

SPLIT_MAP = {"validation": "val"}   # normalise HF split names → our schema


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_checkpoint() -> dict:
    if CKPT_FILE.exists():
        try:
            return json.loads(CKPT_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_checkpoint(ckpt: dict) -> None:
    CKPT_FILE.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")


def _ckpt_key(ds_key: str, split: str) -> str:
    return f"{ds_key}/{split}"


# ══════════════════════════════════════════════════════════════════════════════
# Image helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save_image(img: Image.Image, path: Path) -> str:
    """Save image as JPEG (quality 90). Returns relative path string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = img.convert("RGB")
    img.save(path, format="JPEG", quality=90, optimize=True)
    return str(path.relative_to(BASE_DIR))


# ══════════════════════════════════════════════════════════════════════════════
# Per-dataset converters
# ══════════════════════════════════════════════════════════════════════════════

def _record_rvlcdip(ex: dict, idx: int, split: str, img_dir: Path) -> Optional[dict]:
    classes = RVL_CLASSES
    try:
        label_int = int(ex["label"])
        class_name = classes[label_int]
    except (KeyError, IndexError, ValueError):
        return None

    img_raw = ex.get("image") or ex.get("img")
    if img_raw is None:
        return None
    img = img_raw if isinstance(img_raw, Image.Image) else Image.open(img_raw)
    w, h = img.size
    out_path = img_dir / f"{idx:06d}.jpg"
    rel = _save_image(img, out_path)

    return {
        "id":             f"rvlcdip__{split}__{idx:06d}",
        "source":         "aharley/rvl_cdip",
        "task":           "classification",
        "split":          split,
        "image_path":     rel,
        "instruction":    CLS_INSTRUCTION_TEMPLATE.format(classes=", ".join(classes)),
        "response":       class_name,
        "image_width":    w,
        "image_height":   h,
        "language":       "en",
        "document_class": class_name,
        "is_handwritten": False,
        "is_scanned":     True,
        "quality":        "high",
    }


def _record_iam(ex: dict, idx: int, split: str, img_dir: Path) -> Optional[dict]:
    text = ex.get("text") or ex.get("transcription") or ex.get("label") or ""
    img_raw = ex.get("image") or ex.get("img")
    if img_raw is None or not text:
        return None
    img = img_raw if isinstance(img_raw, Image.Image) else Image.open(img_raw)
    w, h = img.size
    out_path = img_dir / f"{idx:06d}.jpg"
    rel = _save_image(img, out_path)
    return {
        "id":             f"iam__{split}__{idx:06d}",
        "source":         "Teklia/IAM-line",
        "task":           "ocr",
        "split":          split,
        "image_path":     rel,
        "instruction":    OCR_INSTRUCTION,
        "response":       str(text).strip(),
        "image_width":    w,
        "image_height":   h,
        "language":       "en",
        "full_text":      str(text).strip(),
        "is_handwritten": True,
        "is_scanned":     False,
        "quality":        "good",
    }


def _record_sroie(ex: dict, idx: int, split: str, img_dir: Path) -> Optional[dict]:
    img_raw = ex.get("image") or ex.get("img")
    if img_raw is None:
        return None
    # SROIE can yield OCR text from the words field
    words = ex.get("words") or ex.get("tokens") or []
    text  = " ".join(str(w) for w in words) if words else ex.get("text", "")
    if not text:
        return None
    img = img_raw if isinstance(img_raw, Image.Image) else Image.open(img_raw)
    w, h = img.size
    out_path = img_dir / f"{idx:06d}.jpg"
    rel = _save_image(img, out_path)
    return {
        "id":             f"sroie__{split}__{idx:06d}",
        "source":         "rth/sroie-2019-v2",
        "task":           "ocr",
        "split":          split,
        "image_path":     rel,
        "instruction":    OCR_INSTRUCTION,
        "response":       text.strip(),
        "image_width":    w,
        "image_height":   h,
        "language":       "en",
        "full_text":      text.strip(),
        "is_handwritten": False,
        "is_scanned":     True,
        "quality":        "good",
    }


def _record_cord(ex: dict, idx: int, split: str, img_dir: Path) -> Optional[dict]:
    img_raw = ex.get("image") or ex.get("img")
    if img_raw is None:
        return None
    gt_parse = ex.get("ground_truth")
    if gt_parse:
        if isinstance(gt_parse, str):
            try:
                gt_parse = json.loads(gt_parse)
            except Exception:
                gt_parse = {}
        if isinstance(gt_parse, dict):
            gt_parse = gt_parse.get("gt_parse", gt_parse)
    else:
        gt_parse = {}
    img = img_raw if isinstance(img_raw, Image.Image) else Image.open(img_raw)
    w, h = img.size
    out_path = img_dir / f"{idx:06d}.jpg"
    rel = _save_image(img, out_path)
    return {
        "id":           f"cord__{split}__{idx:06d}",
        "source":       "naver-clova-ix/cord-v2",
        "task":         "kie",
        "split":        split,
        "image_path":   rel,
        "instruction":  KIE_INSTRUCTION,
        "response":     json.dumps(gt_parse, ensure_ascii=False),
        "image_width":  w,
        "image_height": h,
        "language":     "en",
        "is_scanned":   True,
        "quality":      "good",
    }


def _record_funsd(ex: dict, idx: int, split: str, img_dir: Path) -> Optional[dict]:
    img_raw = ex.get("image") or ex.get("img")
    if img_raw is None:
        return None
    words    = ex.get("words") or []
    ner_tags = ex.get("ner_tags") or []
    # ner_tags may be ints — map via dataset features if possible (done in caller)
    img = img_raw if isinstance(img_raw, Image.Image) else Image.open(img_raw)
    w, h = img.size
    out_path = img_dir / f"{idx:06d}.jpg"
    rel = _save_image(img, out_path)
    entities = [{"word": str(words[i]), "tag": str(ner_tags[i])}
                for i in range(min(len(words), len(ner_tags)))]
    return {
        "id":           f"funsd__{split}__{idx:06d}",
        "source":       "nielsr/funsd",
        "task":         "ner",
        "split":        split,
        "image_path":   rel,
        "instruction":  NER_INSTRUCTION,
        "response":     json.dumps(entities, ensure_ascii=False),
        "image_width":  w,
        "image_height": h,
        "language":     "en",
        "words":        [str(w) for w in words],
        "is_scanned":   True,
        "quality":      "good",
    }


def _record_tobacco(ex: dict, idx: int, split: str, img_dir: Path) -> Optional[dict]:
    classes = TOBACCO_CLASSES
    img_raw = ex.get("image") or ex.get("img")
    label   = ex.get("label") or ex.get("class") or ex.get("category")
    if img_raw is None or label is None:
        return None
    # label can be int or string
    if isinstance(label, int):
        if 0 <= label < len(classes):
            label = classes[label]
        else:
            return None
    img = img_raw if isinstance(img_raw, Image.Image) else Image.open(img_raw)
    w, h = img.size
    out_path = img_dir / f"{idx:06d}.jpg"
    rel = _save_image(img, out_path)
    all_classes = ", ".join(classes)
    return {
        "id":             f"tobacco__{split}__{idx:06d}",
        "source":         "maveriq/tobacco3482",
        "task":           "classification",
        "split":          split,
        "image_path":     rel,
        "instruction":    CLS_INSTRUCTION_TEMPLATE.format(classes=all_classes),
        "response":       str(label),
        "image_width":    w,
        "image_height":   h,
        "language":       "en",
        "document_class": str(label),
        "is_handwritten": False,
        "is_scanned":     True,
        "quality":        "good",
    }


CONVERTERS = {
    "rvlcdip": _record_rvlcdip,
    "iam":     _record_iam,
    "sroie":   _record_sroie,
    "cord":    _record_cord,
    "funsd":   _record_funsd,
    "tobacco": _record_tobacco,
}


# ══════════════════════════════════════════════════════════════════════════════
# Core processing
# ══════════════════════════════════════════════════════════════════════════════

def process_dataset(ds_key: str, cfg: dict, resume: bool,
                    max_n: Optional[int], ckpt: dict) -> None:
    from datasets import load_dataset

    converter = CONVERTERS[ds_key]
    hf_id     = cfg["hf_id"]
    log.info("=" * 70)
    log.info(f"Dataset : {cfg['label']}")
    log.info(f"HF ID   : {hf_id}")

    for hf_split in cfg["splits"]:
        schema_split = SPLIT_MAP.get(hf_split, hf_split)
        ck = _ckpt_key(ds_key, schema_split)

        if resume and ckpt.get(ck) == "done":
            log.info(f"  [{schema_split}] already complete — skipping")
            continue

        start_idx = ckpt.get(ck + "/last_idx", 0) if resume else 0
        out_dir   = DATA_DIR / ds_key
        img_dir   = out_dir / "images" / schema_split
        jsonl_path = out_dir / f"{schema_split}.jsonl"
        out_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"  [{schema_split}] loading from HuggingFace (split={hf_split}) ...")
        t0 = time.time()

        try:
            ds_split = load_dataset(hf_id, split=hf_split,
                                    streaming=cfg["streaming"],
                                    trust_remote_code=True)
        except Exception as exc:
            log.error(f"  [{schema_split}] load failed: {exc}")
            continue

        total = getattr(ds_split, "num_rows", None)
        log.info(f"  [{schema_split}] total records: {total or 'unknown (streaming)'}")

        mode = "a" if start_idx > 0 else "w"
        written = skipped = errors = 0

        with open(jsonl_path, mode, encoding="utf-8") as fout:
            for idx, ex in enumerate(ds_split):
                if idx < start_idx:
                    continue
                if max_n and (written + skipped) >= max_n:
                    break
                try:
                    rec = converter(ex, idx, schema_split, img_dir)
                    if rec is None:
                        skipped += 1
                        continue
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                except Exception as exc:
                    log.warning(f"    sample {idx} error: {exc}")
                    errors += 1

                if (idx + 1) % 1000 == 0:
                    elapsed = time.time() - t0
                    rate = (idx - start_idx + 1) / max(elapsed, 1)
                    remaining_str = ""
                    if total:
                        rem = (total - idx - 1) / max(rate, 0.001)
                        remaining_str = f"  ETA {rem/60:.0f}m"
                    log.info(
                        f"    [{schema_split}] {idx+1}/{total or '?'}  "
                        f"written={written}  skipped={skipped}  errors={errors}  "
                        f"{rate:.1f} rec/s{remaining_str}"
                    )
                    # Save progress checkpoint every 1000 records
                    ckpt[ck + "/last_idx"] = idx + 1
                    _save_checkpoint(ckpt)

        elapsed = time.time() - t0
        log.info(
            f"  [{schema_split}] done — written={written}  skipped={skipped}  "
            f"errors={errors}  time={elapsed/60:.1f}m"
        )
        ckpt[ck] = "done"
        ckpt.pop(ck + "/last_idx", None)
        _save_checkpoint(ckpt)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Download and format datasets for fine-tuning")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=None,
                        help="Datasets to process (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed splits")
    parser.add_argument("--max", type=int, default=None, metavar="N",
                        help="Cap records per split (for debugging)")
    parser.add_argument("--list", action="store_true",
                        help="List available datasets and exit")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:")
        for k, v in DATASETS.items():
            print(f"  {k:<12}  {v['label']}")
        return

    keys = args.datasets or list(DATASETS.keys())
    ckpt = _load_checkpoint()

    log.info("=" * 70)
    log.info(f"Starting download_and_format_data.py")
    log.info(f"Datasets : {keys}")
    log.info(f"Resume   : {args.resume}")
    log.info(f"Max/split: {args.max or 'unlimited'}")
    log.info(f"Output   : {DATA_DIR}")
    log.info("=" * 70)

    t_total = time.time()
    for key in keys:
        if key not in DATASETS:
            log.warning(f"Unknown dataset key '{key}' — skipping")
            continue
        try:
            process_dataset(key, DATASETS[key], args.resume, args.max, ckpt)
        except KeyboardInterrupt:
            log.warning("Interrupted by user — progress saved to checkpoint")
            sys.exit(1)
        except Exception as exc:
            log.error(f"Dataset '{key}' failed: {exc}", exc_info=True)

    log.info("=" * 70)
    log.info(f"All done  total time={( time.time()-t_total)/60:.1f}m")
    log.info(f"Checkpoint: {CKPT_FILE}")


if __name__ == "__main__":
    main()

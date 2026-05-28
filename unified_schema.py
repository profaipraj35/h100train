"""
unified_schema.py — Canonical record format for all 8 IDP tasks.

Every dataset, regardless of source, is converted into this structure before
training or evaluation. Fields are nullable; a record only populates the fields
relevant to its task(s). This allows a single DataLoader to serve all tasks.

FIELD JUSTIFICATIONS (why each field exists):
─────────────────────────────────────────────
CORE (every record):
  id              – Unique key for deduplication, cross-dataset joins, and
                    result backtracking. Format: {source}__{split}__{index}
  source          – Dataset provenance (e.g. "cord_v2", "rvl_cdip"). Needed
                    to weight loss per domain and analyse per-dataset accuracy.
  task            – Which of the 8 tasks this record serves. Enables multi-task
                    batching and task-conditioned evaluation.
  split           – train/val/test. Keeps contamination impossible when merging
                    multiple datasets into one loader.
  image_path      – All 8 tasks are image-based. Relative to BASE_DIR.
  image_width /
  image_height    – Required to normalise bounding boxes to 0-1000 scale
                    (LayoutLM standard) and to resize for vision encoders.
  instruction     – The prompt fed to the model. Task-specific template stored
                    here so the schema is self-contained and reproducible.
  response        – Ground-truth output as a plain string. Keeps the schema
                    agnostic to model type: LLMs generate this string directly;
                    classifiers map their label to it; structured tasks use JSON.
  language        – ISO-639-1 code (en, de, zh …). Multilingual is a !! gap for
                    KIE and NER. Needed to filter or weight language-specific data.

DOCUMENT-LEVEL (Classification, Splitting):
  document_class  – Class label (RVL-CDIP 16 classes). Primary output for
                    Classification. Also used by Splitting to label each page.
  is_doc_boundary – True when this page starts a new logical document. Primary
                    signal for the Splitting / Triage task (!! gap: multi-page).
  page_number     – 1-based index within the parent document. Required for
                    Splitting and multi-page VQA where page order matters.
  doc_id          – Groups pages that belong to the same physical document.
                    Allows multi-page context windows for VQA and Splitting.
  page_count      – Total pages in the parent document. Helps models know
                    whether they are reading a one-page letter or a 50-page batch.

OCR / TEXT (OCR, KIE, NER, Layout — everything word-level):
  full_text       – Complete OCR ground truth as a single string. Primary
                    output for OCR task. Also the input text for NER when
                    working in text-only mode.
  words           – List of word strings aligned with bboxes. Required by
                    LayoutLMv3, BROS, and any layout-aware model that needs
                    both text and position simultaneously.
  bboxes          – Parallel list of [x0,y0,x1,y1] in 0-1000 normalised scale.
                    Used by OCR (word location output), KIE (field grounding),
                    NER (entity span location), Layout (text element positions).
  ner_tags        – BIO tags aligned with `words` (e.g. B-TOTAL, I-ORG, O).
                    Primary output for NER. Also populates KIE entity spans.

INFORMATION EXTRACTION (KIE):
  entities        – Dict of {field_name: value} e.g. {"total":"$45","date":"..."}.
                    Primary structured output for KIE. Serialised to JSON string
                    in `response` for LLM training; kept as dict here for metrics.

LAYOUT (Layout Segmentation, Table Understanding):
  layout_regions  – List of {label, bbox, text?} dicts. Primary output for
                    Layout Segmentation. Labels: text, title, table, figure,
                    list, header, footer. Bbox in 0-1000 scale.
  table_structure – Structured dict {rows, cols, cells:[{row,col,text,bbox}]}.
                    Primary output for Table Understanding (!! gap: spanning
                    cells, hierarchical headers). Also serialised in `response`.

VQA:
  question        – The natural-language question. Primary input for VQA.
                    Appended to `instruction` at training time.
  answer          – Short answer string. Primary output for VQA.
  answer_bbox     – [x0,y0,x1,y1] of the evidence region. Enables grounding-
                    aware VQA models to locate evidence (! gap: cross-doc VQA).

QUALITY FLAGS (affect model performance across all tasks):
  is_handwritten  – True for IAM, handwritten FUNSD samples. Directly maps to
                    the !! Handwritten gap in OCR, Classification, and KIE.
  is_scanned      – True for photographed/scanned docs vs born-digital PDFs.
                    Scanned docs have worse OCR, which cascades into KIE and NER.
  quality         – "high" | "medium" | "low". Low = noisy scan, degraded ink.
                    Drives stratified evaluation to expose model weaknesses.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class BBox:
    """Bounding box in 0-1000 normalised scale (LayoutLM standard)."""
    x0: int
    y0: int
    x1: int
    y1: int

    def to_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass
class LayoutRegion:
    label: str                        # text|title|table|figure|list|header|footer
    bbox: list[int]                   # [x0,y0,x1,y1] 0-1000 scale
    text: Optional[str] = None        # text content of the region if available


@dataclass
class TableCell:
    row: int
    col: int
    text: str
    row_span: int = 1
    col_span: int = 1
    bbox: Optional[list[int]] = None


@dataclass
class TableStructure:
    n_rows: int
    n_cols: int
    cells: list[TableCell] = field(default_factory=list)

    def to_dict(self):
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "cells": [asdict(c) for c in self.cells],
        }


# ── Valid values ───────────────────────────────────────────────────────────────
VALID_TASKS = {
    "ocr",            # OCR / Text Extraction
    "classification", # Document Classification
    "kie",            # Key Information Extraction
    "layout",         # Layout Segmentation
    "vqa",            # Document VQA
    "splitting",      # Document Splitting / Triage
    "ner",            # Named Entity Recognition
    "table",          # Table Understanding
}

VALID_SPLITS   = {"train", "val", "test"}
VALID_QUALITY  = {"high", "medium", "low"}


@dataclass
class UnifiedRecord:
    # ── CORE (required for every record) ───────────────────────────────────────
    id:           str
    source:       str
    task:         str                         # one of VALID_TASKS
    split:        str                         # train | val | test
    image_path:   str                         # relative to BASE_DIR
    instruction:  str                         # prompt for model
    response:     str                         # expected output as string

    # ── IMAGE ──────────────────────────────────────────────────────────────────
    image_width:  Optional[int]  = None
    image_height: Optional[int]  = None

    # ── LANGUAGE ───────────────────────────────────────────────────────────────
    language:     str            = "en"       # ISO-639-1

    # ── DOCUMENT-LEVEL ─────────────────────────────────────────────────────────
    document_class:   Optional[str]  = None   # RVL-CDIP label or custom
    is_doc_boundary:  Optional[bool] = None   # Splitting task
    page_number:      Optional[int]  = None   # 1-based
    doc_id:           Optional[str]  = None   # groups pages of same document
    page_count:       Optional[int]  = None

    # ── OCR / TEXT ─────────────────────────────────────────────────────────────
    full_text:    Optional[str]       = None   # complete OCR ground truth
    words:        Optional[list[str]] = None   # word list (aligned with bboxes)
    bboxes:       Optional[list]      = None   # [[x0,y0,x1,y1] …] 0-1000 scale
    ner_tags:     Optional[list[str]] = None   # BIO tags aligned with words

    # ── KIE ────────────────────────────────────────────────────────────────────
    entities:     Optional[dict]      = None   # {field: value}

    # ── LAYOUT ─────────────────────────────────────────────────────────────────
    layout_regions:  Optional[list]   = None   # list of LayoutRegion dicts
    table_structure: Optional[dict]   = None   # TableStructure.to_dict()

    # ── VQA ────────────────────────────────────────────────────────────────────
    question:     Optional[str]       = None
    answer:       Optional[str]       = None
    answer_bbox:  Optional[list[int]] = None   # [x0,y0,x1,y1]

    # ── QUALITY FLAGS ──────────────────────────────────────────────────────────
    is_handwritten: bool  = False
    is_scanned:     bool  = False
    quality:        str   = "high"             # high | medium | low

    # ──────────────────────────────────────────────────────────────────────────

    def validate(self):
        assert self.task  in VALID_TASKS,  f"Invalid task: {self.task}"
        assert self.split in VALID_SPLITS, f"Invalid split: {self.split}"
        assert self.quality in VALID_QUALITY, f"Invalid quality: {self.quality}"
        assert self.image_path, "image_path is required"
        assert self.instruction, "instruction is required"
        assert self.response is not None, "response is required (may be empty string)"

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ── Per-task field map (which fields are populated per task) ──────────────────
TASK_FIELD_MAP = {
    "ocr": {
        "required": ["image_path", "instruction", "response", "full_text"],
        "optional": ["words", "bboxes", "is_handwritten", "is_scanned", "quality", "language"],
        "response_format": "plain text — the full OCR transcript",
    },
    "classification": {
        "required": ["image_path", "instruction", "response", "document_class"],
        "optional": ["is_handwritten", "is_scanned", "quality", "language"],
        "response_format": "single class label string, e.g. 'invoice'",
    },
    "kie": {
        "required": ["image_path", "instruction", "response", "entities"],
        "optional": ["words", "bboxes", "full_text", "language", "quality"],
        "response_format": "JSON string of {field: value} pairs",
    },
    "layout": {
        "required": ["image_path", "instruction", "response", "layout_regions"],
        "optional": ["words", "bboxes", "quality", "is_scanned"],
        "response_format": "JSON list of {label, bbox} region dicts",
    },
    "vqa": {
        "required": ["image_path", "instruction", "response", "question", "answer"],
        "optional": ["answer_bbox", "doc_id", "page_number", "page_count", "language"],
        "response_format": "free-text answer string",
    },
    "splitting": {
        "required": ["image_path", "instruction", "response",
                     "document_class", "is_doc_boundary", "page_number"],
        "optional": ["doc_id", "page_count", "quality"],
        "response_format": 'JSON: {"document_type": "...", "is_new_document": true/false}',
    },
    "ner": {
        "required": ["image_path", "instruction", "response",
                     "words", "bboxes", "ner_tags"],
        "optional": ["full_text", "entities", "language", "quality"],
        "response_format": "JSON list of {entity, type, start, end} spans",
    },
    "table": {
        "required": ["image_path", "instruction", "response", "table_structure"],
        "optional": ["layout_regions", "words", "bboxes", "quality", "is_scanned"],
        "response_format": "JSON of {n_rows, n_cols, cells:[{row,col,text}]}",
    },
}


def print_schema_summary():
    """Print a human-readable field-to-task mapping table."""
    all_fields = [
        "full_text", "words", "bboxes", "ner_tags",
        "document_class", "entities", "layout_regions", "table_structure",
        "question", "answer", "answer_bbox",
        "is_doc_boundary", "page_number", "doc_id", "page_count",
        "is_handwritten", "is_scanned", "quality", "language",
    ]
    tasks = ["ocr", "classification", "kie", "layout", "vqa", "splitting", "ner", "table"]

    header = f"{'Field':<20}" + "".join(f"{t[:5]:>8}" for t in tasks)
    print(header)
    print("-" * len(header))
    for f in all_fields:
        row = f"{f:<20}"
        for t in tasks:
            req = TASK_FIELD_MAP[t]["required"]
            opt = TASK_FIELD_MAP[t]["optional"]
            if f in req:
                row += f"{'  R':>8}"
            elif f in opt:
                row += f"{'  o':>8}"
            else:
                row += f"{'  .':>8}"
        print(row)
    print("\nR = required   o = optional   . = not used")


if __name__ == "__main__":
    print_schema_summary()

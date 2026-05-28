"""
build_eval_datasets.py — Build per-task evaluation datasets + metric utilities.

Reads  : formatted_data/{dataset}/{split}.jsonl  (output of format_datasets.py)
Writes : eval_datasets/{task}/eval_{split}.jsonl  — ground-truth eval records
         eval_datasets/{task}/metadata.json        — metric specs & dataset stats
         eval_datasets/build_log.json              — resume checkpoint

Each output record carries all ground-truth fields required to compute every
paper metric for that task.  Metric-computation functions at the bottom of
this file can be imported directly into your inference / scoring scripts.

Usage:
  python build_eval_datasets.py                        # all 4 tasks, test split
  python build_eval_datasets.py --tasks ocr ner        # specific tasks
  python build_eval_datasets.py --split val            # val split instead
  python build_eval_datasets.py --max 500              # cap per source dataset
  python build_eval_datasets.py --resume               # skip completed tasks
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

BASE_DIR  = Path(__file__).parent.resolve()
DATA_DIR  = BASE_DIR / "formatted_data"
EVAL_DIR  = BASE_DIR / "eval_datasets"
BUILD_LOG = EVAL_DIR / "build_log.json"

# ── Task → metric names ────────────────────────────────────────────────────────

TASK_METRICS: dict[str, list[str]] = {
    "ocr": [
        "cer", "wer", "ned", "f1_word",
        "inference_time_ms", "gpu_memory_gb",
    ],
    "classification": [
        "accuracy", "macro_f1", "weighted_f1",
        "per_class_precision", "per_class_recall", "confusion_matrix",
        "inference_latency_ms", "gpu_memory_gb", "finetuned_vs_zeroshot_gap",
    ],
    "kie": [
        "field_level_f1", "exact_match", "levenshtein_similarity",
        "zeroshot_template_f1", "normalized_f1", "inference_time_per_doc",
    ],
    "ner": [
        "entity_f1_strict", "precision_per_type", "recall_per_type",
        "span_accuracy", "cross_domain_f1", "f1_unseen_types",
        "inference_latency_ms",
    ],
}

METRIC_DESCRIPTIONS: dict[str, str] = {
    # OCR
    "cer":                         "Character Error Rate = edit_distance(pred, gt) / len(gt)",
    "wer":                         "Word Error Rate = edit_distance(pred_tokens, gt_tokens) / len(gt_tokens)",
    "ned":                         "Normalized Edit Distance = edit_distance / max(len(pred), len(gt))",
    "f1_word":                     "Token-level F1 on bag-of-words overlap between prediction and ground truth",
    "inference_time_ms":           "Wall-clock time per image in milliseconds (measured during inference)",
    "gpu_memory_gb":               "Peak GPU memory during inference in GB (measured during inference)",
    # Classification
    "accuracy":                    "Fraction of correctly classified documents",
    "macro_f1":                    "Unweighted mean F1 across all classes",
    "weighted_f1":                 "Class-frequency-weighted mean F1",
    "per_class_precision":         "Precision per document class",
    "per_class_recall":            "Recall per document class",
    "confusion_matrix":            "N×N matrix: rows=actual class, cols=predicted class",
    "inference_latency_ms":        "Wall-clock latency per image in milliseconds (measured during inference)",
    "finetuned_vs_zeroshot_gap":   "Accuracy(fine-tuned) − Accuracy(zero-shot) for the same model",
    # KIE
    "field_level_f1":              "Token-overlap F1 per field, averaged across all fields in the document",
    "exact_match":                 "Fraction of fields where normalized prediction == normalized ground truth",
    "levenshtein_similarity":      "1 − NED averaged across all fields (higher = more similar)",
    "zeroshot_template_f1":        "Field F1 when the document template was unseen at training time",
    "normalized_f1":               "Field F1 with partial-credit normalization for near-matches",
    "inference_time_per_doc":      "End-to-end wall-clock latency per document in milliseconds",
    # NER
    "entity_f1_strict":            "F1 requiring exact span boundary AND entity type match (CoNLL-style)",
    "precision_per_type":          "Precision for each entity type separately",
    "recall_per_type":             "Recall for each entity type separately",
    "span_accuracy":               "Fraction of word tokens with a correctly predicted BIO tag",
    "cross_domain_f1":             "Entity F1 on held-out source domain (domain not in fine-tune set)",
    "f1_unseen_types":             "Entity F1 restricted to entity types absent from the fine-tune training set",
}

# ── Known class/type lists per HF source ──────────────────────────────────────

_RVL_CLASSES = [
    "letter", "form", "email", "handwritten", "advertisement",
    "scientific_report", "scientific_publication", "specification",
    "file_folder", "news_article", "budget", "invoice",
    "presentation", "questionnaire", "resume", "memo",
]

_KNOWN_CLASSES: dict[str, list[str]] = {
    "aharley/rvl_cdip":   _RVL_CLASSES,
    "jordyvl/RVL-CDIP-N": _RVL_CLASSES,
}

_KNOWN_NER_TYPES: dict[str, list[str]] = {
    "nielsr/funsd": ["ANSWER", "QUESTION", "HEADER", "OTHER"],
}


# ══════════════════════════════════════════════════════════════════════════════
# BUILD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _read_jsonl(path: Path, max_n: Optional[int] = None):
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1
            if max_n and count >= max_n:
                break


def _find_split_files(split: str) -> list[Path]:
    """Return all formatted_data/**/{split}.jsonl paths, sorted."""
    return sorted(DATA_DIR.rglob(f"{split}.jsonl"))


def _collect_classes(records: list[dict], source: str) -> list[str]:
    if source in _KNOWN_CLASSES:
        return _KNOWN_CLASSES[source]
    seen = sorted({
        r.get("document_class") or r.get("response", "")
        for r in records
        if r.get("document_class") or r.get("response", "")
    })
    return seen


def _collect_entity_types(records: list[dict], source: str) -> list[str]:
    if source in _KNOWN_NER_TYPES:
        return _KNOWN_NER_TYPES[source]
    types: set[str] = set()
    for r in records:
        for tag in r.get("ner_tags") or []:
            if isinstance(tag, str) and tag not in ("O", ""):
                base = tag[2:] if tag.startswith(("B-", "I-")) else tag
                types.add(base)
    return sorted(types)


# ── Per-task record converters ─────────────────────────────────────────────────

def _to_ocr(rec: dict) -> Optional[dict]:
    gt = rec.get("full_text") or rec.get("response", "")
    if not gt:
        return None
    return {
        "id":             rec["id"],
        "source":         rec["source"],
        "task":           "ocr",
        "split":          rec.get("split", ""),
        "image_path":     rec["image_path"],
        "ground_truth":   gt,
        "char_count":     len(gt),
        "word_count":     len(gt.split()),
        "is_handwritten": rec.get("is_handwritten", False),
        "is_scanned":     rec.get("is_scanned", True),
        "quality":        rec.get("quality", "medium"),
        "image_width":    rec.get("image_width"),
        "image_height":   rec.get("image_height"),
        "metrics":        TASK_METRICS["ocr"],
    }


def _to_classification(rec: dict, all_classes: list[str]) -> Optional[dict]:
    gt = rec.get("document_class") or rec.get("response", "")
    if not gt:
        return None
    return {
        "id":               rec["id"],
        "source":           rec["source"],
        "task":             "classification",
        "split":            rec.get("split", ""),
        "image_path":       rec["image_path"],
        "ground_truth":     gt,
        "ground_truth_idx": all_classes.index(gt) if gt in all_classes else -1,
        "all_classes":      all_classes,
        "quality":          rec.get("quality", "high"),
        "image_width":      rec.get("image_width"),
        "image_height":     rec.get("image_height"),
        "metrics":          TASK_METRICS["classification"],
    }


def _to_kie(rec: dict) -> Optional[dict]:
    entities = rec.get("entities")
    if not entities:
        try:
            entities = json.loads(rec.get("response", "{}"))
        except Exception:
            return None
    if not isinstance(entities, dict) or not entities:
        return None
    return {
        "id":           rec["id"],
        "source":       rec["source"],
        "task":         "kie",
        "split":        rec.get("split", ""),
        "image_path":   rec["image_path"],
        "ground_truth": entities,
        "field_names":  list(entities.keys()),
        "words":        rec.get("words"),
        "bboxes":       rec.get("bboxes"),
        "quality":      rec.get("quality", "medium"),
        "image_width":  rec.get("image_width"),
        "image_height": rec.get("image_height"),
        "metrics":      TASK_METRICS["kie"],
    }


def _to_ner(rec: dict, entity_types: list[str]) -> Optional[dict]:
    words    = rec.get("words") or []
    ner_tags = rec.get("ner_tags") or []
    spans: list = []
    try:
        parsed = json.loads(rec.get("response", "[]"))
        if isinstance(parsed, list):
            spans = parsed
    except Exception:
        pass
    if not words and not spans:
        return None
    return {
        "id":                  rec["id"],
        "source":              rec["source"],
        "task":                "ner",
        "split":               rec.get("split", ""),
        "image_path":          rec["image_path"],
        "ground_truth_spans":  spans,
        "ground_truth_tags":   ner_tags,
        "words":               words,
        "bboxes":              rec.get("bboxes"),
        "entity_types":        entity_types,
        "quality":             rec.get("quality", "high"),
        "image_width":         rec.get("image_width"),
        "image_height":        rec.get("image_height"),
        "metrics":             TASK_METRICS["ner"],
    }


_GT_SCHEMA: dict[str, dict] = {
    "ocr": {
        "ground_truth": "str — full text transcription",
        "char_count":   "int — number of characters in ground truth",
        "word_count":   "int — number of whitespace-separated tokens",
    },
    "classification": {
        "ground_truth":     "str — class name",
        "ground_truth_idx": "int — 0-based index into all_classes",
        "all_classes":      "list[str] — ordered class list",
    },
    "kie": {
        "ground_truth": "dict — {field_name: value} pairs extracted from the document",
        "field_names":  "list[str] — keys present in this specific record",
    },
    "ner": {
        "ground_truth_spans": "list[{entity, type, start, end}] — entity spans",
        "ground_truth_tags":  "list[str] — BIO tag per word token",
        "words":              "list[str] — tokenized word list",
        "entity_types":       "list[str] — entity types present in this source",
    },
}


# ── Core build function ────────────────────────────────────────────────────────

def build_task_eval(task: str, split: str, max_n: Optional[int],
                    resume: bool, log: dict) -> dict:
    log_key = f"{task}/{split}"
    if resume and log.get(log_key, {}).get("status") == "complete":
        n = log[log_key]["count"]
        print(f"  [{task}/{split}] Already complete ({n} records) — skipping")
        return log[log_key]

    split_files = _find_split_files(split)
    if not split_files:
        print(f"  [WARN] No {split}.jsonl files found in {DATA_DIR}")
        print(f"         Run format_datasets.py first.")
        return {"status": "error", "error": "no source files found"}

    out_dir  = EVAL_DIR / task
    out_file = out_dir / f"eval_{split}.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    # First pass for classification/NER: collect classes & entity types per source
    classes_by_source: dict[str, list[str]]      = {}
    ner_types_by_source: dict[str, list[str]]    = {}
    if task in ("classification", "ner"):
        for sf in split_files:
            buf = [r for r in _read_jsonl(sf) if r.get("task") == task]
            if buf:
                src = buf[0]["source"]
                if task == "classification":
                    classes_by_source[src] = _collect_classes(buf, src)
                else:
                    ner_types_by_source[src] = _collect_entity_types(buf, src)

    count    = 0
    skipped  = 0
    sources: set[str] = set()
    t0 = time.time()

    with open(out_file, "w", encoding="utf-8") as fout:
        for sf in split_files:
            src_count = 0
            for rec in _read_jsonl(sf):
                if rec.get("task") != task:
                    continue
                if max_n and src_count >= max_n:
                    break
                src = rec.get("source", "unknown")
                sources.add(src)

                if task == "ocr":
                    eval_rec = _to_ocr(rec)
                elif task == "classification":
                    eval_rec = _to_classification(rec, classes_by_source.get(src, []))
                elif task == "kie":
                    eval_rec = _to_kie(rec)
                elif task == "ner":
                    eval_rec = _to_ner(rec, ner_types_by_source.get(src, []))
                else:
                    eval_rec = None

                if eval_rec is None:
                    skipped += 1
                    continue
                fout.write(json.dumps(eval_rec, ensure_ascii=False) + "\n")
                count    += 1
                src_count += 1

    elapsed = time.time() - t0
    print(f"  [{task}/{split}]  {count:>6} records  |  {len(sources)} source(s)  |  "
          f"{skipped} skipped  |  {elapsed:.1f}s")

    # Write metadata.json
    meta: dict = {
        "task":                  task,
        "split":                 split,
        "count":                 count,
        "sources":               sorted(sources),
        "eval_file":             str(out_file.relative_to(BASE_DIR).as_posix()),
        "metrics":               TASK_METRICS[task],
        "metric_descriptions":   {m: METRIC_DESCRIPTIONS[m] for m in TASK_METRICS[task]},
        "ground_truth_schema":   _GT_SCHEMA[task],
        "notes": {
            "inference_time_ms":  "Fill in timing_ms field on each record during inference.",
            "gpu_memory_gb":      "Fill in gpu_mem_gb field on each record during inference.",
        },
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if task == "classification" and classes_by_source:
        meta["classes_by_source"] = classes_by_source
    if task == "ner" and ner_types_by_source:
        meta["entity_types_by_source"] = ner_types_by_source

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    entry = {
        "status":  "complete",
        "count":   count,
        "skipped": skipped,
        "sources": sorted(sources),
        "file":    str(out_file.relative_to(BASE_DIR).as_posix()),
        "ts":      time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return entry


# ══════════════════════════════════════════════════════════════════════════════
# METRIC COMPUTATION UTILITIES
# Import these into your inference/scoring script:
#   from build_eval_datasets import ocr_metrics, classification_metrics, ...
# ══════════════════════════════════════════════════════════════════════════════

# ── Levenshtein (no external deps) ────────────────────────────────────────────

def _lev_chars(a: str, b: str) -> int:
    if a == b:  return 0
    if not a:   return len(b)
    if not b:   return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[-1] + 1,
                            prev[j - 1] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def _lev_tokens(a: list, b: list) -> int:
    if a == b:  return 0
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, 1):
        curr = [i]
        for j, tb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[-1] + 1,
                            prev[j - 1] + (0 if ta == tb else 1)))
        prev = curr
    return prev[-1]


# ── OCR ───────────────────────────────────────────────────────────────────────

def compute_cer(pred: str, gt: str) -> float:
    """Character Error Rate."""
    if not gt: return 0.0 if not pred else 1.0
    return _lev_chars(pred, gt) / len(gt)


def compute_wer(pred: str, gt: str) -> float:
    """Word Error Rate."""
    gt_w, pred_w = gt.split(), pred.split()
    if not gt_w: return 0.0 if not pred_w else 1.0
    return _lev_tokens(pred_w, gt_w) / len(gt_w)


def compute_ned(pred: str, gt: str) -> float:
    """Normalized Edit Distance (0=identical, 1=fully different)."""
    denom = max(len(pred), len(gt))
    return 0.0 if denom == 0 else _lev_chars(pred, gt) / denom


def compute_word_f1(pred: str, gt: str) -> float:
    """Bag-of-words token F1."""
    p_toks, g_toks = set(pred.lower().split()), set(gt.lower().split())
    if not p_toks and not g_toks: return 1.0
    if not p_toks or  not g_toks: return 0.0
    common = p_toks & g_toks
    prec = len(common) / len(p_toks)
    rec  = len(common) / len(g_toks)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def ocr_metrics(predictions: list[str], records: list[dict]) -> dict:
    """
    Score OCR predictions against eval records.

    predictions : list[str]  — one predicted text string per record
    records     : list[dict] — loaded from eval_datasets/ocr/eval_{split}.jsonl

    Returns:
      {
        "aggregate": { mean_cer, mean_wer, mean_ned, mean_f1_word, n_samples },
        "per_record": [ { id, cer, wer, ned, f1_word }, ... ]
      }
    """
    assert len(predictions) == len(records), "length mismatch"
    per: list[dict] = []
    for pred, rec in zip(predictions, records):
        gt = rec["ground_truth"]
        per.append({
            "id":      rec["id"],
            "cer":     round(compute_cer(pred, gt),      4),
            "wer":     round(compute_wer(pred, gt),      4),
            "ned":     round(compute_ned(pred, gt),      4),
            "f1_word": round(compute_word_f1(pred, gt),  4),
        })
    n = len(per)
    def _avg(k): return sum(r[k] for r in per) / n if n else 0.0
    return {
        "aggregate": {
            "mean_cer":     round(_avg("cer"),     4),
            "mean_wer":     round(_avg("wer"),     4),
            "mean_ned":     round(_avg("ned"),     4),
            "mean_f1_word": round(_avg("f1_word"), 4),
            "n_samples":    n,
        },
        "per_record": per,
    }


# ── Classification ────────────────────────────────────────────────────────────

def classification_metrics(predictions: list[str], records: list[dict]) -> dict:
    """
    Score classification predictions against eval records.

    predictions : list[str]  — one predicted class name per record
    records     : list[dict] — loaded from eval_datasets/classification/eval_{split}.jsonl

    Returns:
      {
        "aggregate":       { accuracy, macro_f1, weighted_f1, n_samples },
        "per_class":       { class_name: { precision, recall, f1 } },
        "confusion_matrix":{ classes, matrix }
      }
    """
    assert len(predictions) == len(records), "length mismatch"
    gts     = [r["ground_truth"] for r in records]
    classes = records[0]["all_classes"] if records else []

    tp: dict = defaultdict(int)
    fp: dict = defaultdict(int)
    fn: dict = defaultdict(int)
    for pred, gt in zip(predictions, gts):
        if pred == gt:
            tp[gt]   += 1
        else:
            fp[pred] += 1
            fn[gt]   += 1

    class_counts: dict = defaultdict(int)
    for g in gts: class_counts[g] += 1
    total = len(gts)

    per_class: dict = {}
    for cls in classes:
        _tp, _fp, _fn = tp[cls], fp[cls], fn[cls]
        p  = _tp / (_tp + _fp) if (_tp + _fp) else 0.0
        r  = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_class[cls] = {
            "precision": round(p,  4),
            "recall":    round(r,  4),
            "f1":        round(f1, 4),
            "support":   class_counts[cls],
        }

    macro_f1    = sum(v["f1"] for v in per_class.values()) / len(classes) if classes else 0.0
    weighted_f1 = (sum(per_class[c]["f1"] * class_counts[c] for c in classes) / total
                   if total else 0.0)
    accuracy    = sum(p == g for p, g in zip(predictions, gts)) / total if total else 0.0

    cls_idx = {c: i for i, c in enumerate(classes)}
    n_cls   = len(classes)
    cm      = [[0] * n_cls for _ in range(n_cls)]
    for pred, gt in zip(predictions, gts):
        if gt in cls_idx and pred in cls_idx:
            cm[cls_idx[gt]][cls_idx[pred]] += 1

    return {
        "aggregate": {
            "accuracy":    round(accuracy,    4),
            "macro_f1":    round(macro_f1,    4),
            "weighted_f1": round(weighted_f1, 4),
            "n_samples":   total,
        },
        "per_class":        per_class,
        "confusion_matrix": {"classes": classes, "matrix": cm},
    }


def finetuned_vs_zeroshot_gap(finetuned: dict, zeroshot: dict) -> dict:
    """
    Compute gap between fine-tuned and zero-shot classification results.
    Both inputs should be the output of classification_metrics().

    Returns { accuracy_gap, macro_f1_gap, finetuned_accuracy, zeroshot_accuracy }
    """
    ft_acc = finetuned["aggregate"]["accuracy"]
    zs_acc = zeroshot["aggregate"]["accuracy"]
    ft_f1  = finetuned["aggregate"]["macro_f1"]
    zs_f1  = zeroshot["aggregate"]["macro_f1"]
    return {
        "accuracy_gap":       round(ft_acc - zs_acc, 4),
        "macro_f1_gap":       round(ft_f1  - zs_f1,  4),
        "finetuned_accuracy": ft_acc,
        "zeroshot_accuracy":  zs_acc,
    }


# ── KIE ───────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def _field_f1(pred_val: str, gt_val: str) -> float:
    p_toks = set(_norm(pred_val).split())
    g_toks = set(_norm(gt_val).split())
    if not p_toks and not g_toks: return 1.0
    if not p_toks or  not g_toks: return 0.0
    common = p_toks & g_toks
    prec = len(common) / len(p_toks)
    rec  = len(common) / len(g_toks)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def kie_metrics(predictions: list[dict], records: list[dict]) -> dict:
    """
    Score KIE predictions against eval records.

    predictions : list[dict] — one {field: predicted_value} dict per record
    records     : list[dict] — loaded from eval_datasets/kie/eval_{split}.jsonl

    Returns:
      {
        "aggregate":    { mean_field_level_f1, mean_exact_match,
                          mean_levenshtein_similarity, mean_normalized_f1, n_samples },
        "per_field_f1": { field_name: mean_f1 },
        "per_record":   [ { id, field_level_f1, exact_match,
                            levenshtein_similarity, normalized_f1 } ]
      }
    """
    assert len(predictions) == len(records), "length mismatch"
    per: list[dict] = []
    field_f1s: dict = defaultdict(list)

    for pred_dict, rec in zip(predictions, records):
        gt_dict  = rec["ground_truth"]
        all_keys = set(gt_dict) | set(pred_dict)

        ff1s, ems, sims, nf1s = [], [], [], []
        for key in all_keys:
            gt_v   = str(gt_dict.get(key, ""))
            pred_v = str(pred_dict.get(key, ""))
            ff1  = _field_f1(pred_v, gt_v)
            em   = 1.0 if _norm(pred_v) == _norm(gt_v) else 0.0
            sim  = 1.0 - compute_ned(pred_v.lower(), gt_v.lower())
            ff1s.append(ff1); ems.append(em); sims.append(sim); nf1s.append(ff1)
            field_f1s[key].append(ff1)

        n = len(all_keys) or 1
        per.append({
            "id":                    rec["id"],
            "field_level_f1":        round(sum(ff1s) / n, 4),
            "exact_match":           round(sum(ems)  / n, 4),
            "levenshtein_similarity": round(sum(sims) / n, 4),
            "normalized_f1":         round(sum(nf1s) / n, 4),
        })

    n = len(per)
    def _avg(k): return sum(r[k] for r in per) / n if n else 0.0
    return {
        "aggregate": {
            "mean_field_level_f1":          round(_avg("field_level_f1"),          4),
            "mean_exact_match":             round(_avg("exact_match"),             4),
            "mean_levenshtein_similarity":  round(_avg("levenshtein_similarity"),  4),
            "mean_normalized_f1":           round(_avg("normalized_f1"),           4),
            "n_samples":                    n,
        },
        "per_field_f1": {k: round(sum(v) / len(v), 4) for k, v in field_f1s.items()},
        "per_record":   per,
    }


# ── NER ───────────────────────────────────────────────────────────────────────

def _span_key(s: dict) -> tuple:
    return (s.get("type", ""), s.get("start", -1), s.get("end", -1))


def ner_metrics(predictions: list[list[dict]], records: list[dict]) -> dict:
    """
    Score NER predictions against eval records.

    predictions : list[list[dict]] — one list of {entity, type, start, end} per record
    records     : list[dict]       — loaded from eval_datasets/ner/eval_{split}.jsonl

    Returns:
      {
        "aggregate": { entity_f1_strict, precision, recall, n_samples },
        "per_type":  { entity_type: { precision, recall, f1 } },
        "per_record":[ { id, tp, fp, fn } ]
      }
    """
    assert len(predictions) == len(records), "length mismatch"
    entity_types = records[0].get("entity_types", []) if records else []

    tp_g = 0; fp_g = 0; fn_g = 0
    tp_t: dict = defaultdict(int)
    fp_t: dict = defaultdict(int)
    fn_t: dict = defaultdict(int)
    tag_correct = 0; tag_total = 0
    per: list[dict] = []

    for pred_spans, rec in zip(predictions, records):
        gt_spans = rec.get("ground_truth_spans", [])
        gt_tags  = rec.get("ground_truth_tags", [])

        pred_keys = {_span_key(s) for s in pred_spans}
        gt_keys   = {_span_key(s) for s in gt_spans}

        tp = len(pred_keys & gt_keys)
        fp = len(pred_keys - gt_keys)
        fn = len(gt_keys  - pred_keys)
        tp_g += tp; fp_g += fp; fn_g += fn

        for sk in gt_keys:
            (tp_t if sk in pred_keys else fn_t)[sk[0]] += 1
        for sk in pred_keys:
            if sk not in gt_keys:
                fp_t[sk[0]] += 1

        per.append({"id": rec["id"], "tp": tp, "fp": fp, "fn": fn})

    def _prf(tp, fp, fn):
        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return round(p, 4), round(r, 4), round(f1, 4)

    gp, gr, gf1 = _prf(tp_g, fp_g, fn_g)
    per_type = {}
    for et in entity_types:
        p, r, f1 = _prf(tp_t[et], fp_t[et], fn_t[et])
        per_type[et] = {"precision": p, "recall": r, "f1": f1}

    return {
        "aggregate": {
            "entity_f1_strict": gf1,
            "precision":        gp,
            "recall":           gr,
            "n_samples":        len(records),
        },
        "per_type":   per_type,
        "per_record": per,
    }


def span_accuracy(pred_tags: list[str], gt_tags: list[str]) -> float:
    """Token-level BIO tag accuracy for a single record."""
    if not gt_tags: return 1.0
    correct = sum(p == g for p, g in zip(pred_tags, gt_tags))
    return correct / len(gt_tags)


# ── Log helpers ───────────────────────────────────────────────────────────────

def _load_log() -> dict:
    if BUILD_LOG.exists():
        with open(BUILD_LOG, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_log(log: dict):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(BUILD_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build evaluation datasets from formatted JSONL")
    parser.add_argument("--tasks",  nargs="*", default=None,
                        choices=list(TASK_METRICS.keys()),
                        help="Tasks to build (default: all four)")
    parser.add_argument("--split",  default="test",
                        help="Source split to build from (default: test)")
    parser.add_argument("--max",    type=int, default=None,
                        help="Max records per source dataset (useful for quick checks)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tasks already marked complete in build_log.json")
    args = parser.parse_args()

    tasks = args.tasks or list(TASK_METRICS.keys())
    log   = _load_log() if args.resume else {}
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nBuilding evaluation datasets")
    print(f"  Tasks  : {tasks}")
    print(f"  Split  : {args.split}")
    print(f"  Source : {DATA_DIR}")
    print(f"  Output : {EVAL_DIR}")
    if args.max:    print(f"  Max    : {args.max} per source")
    if args.resume: print(f"  Resume : ON")
    print()

    t0 = time.time()
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"  {task.upper()}")
        print(f"{'='*60}")
        entry = build_task_eval(task, args.split, args.max, args.resume, log)
        log[f"{task}/{args.split}"] = entry
        _save_log(log)

    print(f"\n{'='*60}")
    print(f"  Done in {time.time() - t0:.1f}s")
    print()
    print(f"  Output files:")
    for task in tasks:
        lk = f"{task}/{args.split}"
        n  = log.get(lk, {}).get("count", "?")
        srcs = ", ".join(log.get(lk, {}).get("sources", []))
        print(f"    eval_datasets/{task}/eval_{args.split}.jsonl  ({n} records)  [{srcs}]")
    print(f"\n  Metric specs:  eval_datasets/{{task}}/metadata.json")
    print(f"  Resume log  :  eval_datasets/build_log.json")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

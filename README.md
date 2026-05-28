# IDP Research — H100 Training Pipeline

End-to-end pipeline for fine-tuning task-specific LoRA adapters on a document processing model (Qwen2.5-VL-3B-Instruct) across OCR and Classification tasks.

## Overview

This repo trains two expert LoRA adapters on a frozen base model:

| Adapter | Task | Training Data | Size |
|---------|------|--------------|------|
| `lora_ocr` | OCR / Text Extraction | IAM Handwriting (6,482) + SROIE receipts (626) | ~7K samples |
| `lora_cls` | Document Classification | RVL-CDIP (320,000) + Tobacco3482 (2,437) | ~322K samples |

Base model: `Qwen/Qwen2.5-VL-3B-Instruct` (frozen, 4-bit QLoRA)

## Quick Start

```bash
git clone https://github.com/profaipraj35/h100train.git
cd h100train
bash run_pipeline.sh
```

That single command will:
1. Install all requirements
2. Download and format datasets from HuggingFace
3. Train the OCR adapter (~10 min on H100)
4. Train the Classification adapter (~90 min on H100)
5. Evaluate both adapters and write results to `eval_results/summary.csv`

> Safe to re-run — every step is checkpointed and resumes where it left off.

## File Structure

```
h100train/
├── run_pipeline.sh                  # Master pipeline — run this
├── download_and_format_data.py      # Downloads datasets from HuggingFace
├── finetune_qlora.py                # QLoRA training script
├── evaluate_all_models.py           # Evaluation across all models and tasks
├── build_eval_datasets.py           # Builds evaluation JSONL files
├── unified_schema.py                # Shared data schema
├── finetune_config_lora_ocr.yaml    # OCR adapter config (H100 optimised)
├── finetune_config_lora_cls.yaml    # CLS adapter config (H100 optimised)
└── requirements_h100.txt            # Python dependencies
```

## Pipeline Steps

### 1. Data Download (`download_and_format_data.py`)

Downloads the following datasets directly from HuggingFace:

| Dataset | HF ID | Task | Samples |
|---------|-------|------|---------|
| RVL-CDIP | `aharley/rvl_cdip` | Classification | 320,000 |
| IAM Handwriting | `Teklia/IAM-line` | OCR | 6,482 |
| SROIE v2 | `rth/sroie-2019-v2` | OCR | 626 |
| CORD v2 | `naver-clova-ix/cord-v2` | KIE | 799 |
| FUNSD | `nielsr/funsd` | NER | 149 |
| Tobacco3482 | `maveriq/tobacco3482` | Classification | 2,437 |

Resume support: saves checkpoint every 1,000 records. Re-running skips completed splits.

```bash
python download_and_format_data.py --resume
python download_and_format_data.py --datasets rvlcdip iam   # specific datasets
python download_and_format_data.py --max 1000               # debug mode
```

### 2. Fine-tuning (`finetune_qlora.py`)

QLoRA fine-tuning with 4-bit NF4 quantization + LoRA adapters (rank=32).

```bash
# OCR adapter (~10 min on H100)
python finetune_qlora.py --config finetune_config_lora_ocr.yaml --resume

# Classification adapter (~90 min on H100)
python finetune_qlora.py --config finetune_config_lora_cls.yaml --resume
```

Outputs saved to:
- `finetune_outputs/lora_ocr/qwen25vl_3b/final_adapter/`
- `finetune_outputs/lora_cls/qwen25vl_3b/final_adapter/`

### 3. Evaluation (`evaluate_all_models.py`)

```bash
python evaluate_all_models.py \
  --models qwen25vl_3b_lora_ocr qwen25vl_3b_lora_cls \
  --tasks ocr classification \
  --resume
```

Results written to `eval_results/summary.csv`.

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 16 GB VRAM (RTX 4090) | H100 80 GB |
| RAM | 32 GB | 64 GB |
| Storage | 100 GB | 200 GB |
| Python | 3.10+ | 3.11 |

## Pipeline Flags

```bash
bash run_pipeline.sh                  # full run (recommended)
bash run_pipeline.sh --skip-data      # skip download (data already present)
bash run_pipeline.sh --skip-train     # skip training (adapters already exist)
bash run_pipeline.sh --eval-only      # only run evaluation
```

## Monitoring

**TensorBoard:**
```bash
tensorboard --logdir finetune_outputs/
```

**Training logs:**
```bash
tail -f logs/pipeline_*.log
```

**Resume after interruption:**  
Just re-run `bash run_pipeline.sh` — state is saved in `logs/pipeline_state.json`.

## Why Separate Adapters?

A single mixed-task model suffers from data imbalance — the dominant task (e.g., RVL-CDIP at 320K) crowds out smaller tasks (e.g., IAM at 6K). Separate LoRA adapters solve this:

- Base model weights are **frozen** — no catastrophic forgetting
- Each adapter is trained on **all available data** for its task
- Adapters are swapped at inference time based on the task
- Results are directly comparable to specialist models (DiT, TrOCR) in the paper

## Tasks Roadmap

- [x] OCR / Text Extraction
- [x] Document Classification
- [ ] Key Information Extraction (KIE)
- [ ] Named Entity Recognition (NER)
- [ ] Document VQA
- [ ] Layout Segmentation
- [ ] Document Splitting
- [ ] Table Understanding

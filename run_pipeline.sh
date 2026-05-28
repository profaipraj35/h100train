#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — IDP Research: full pipeline on H100
#
# Runs in order:
#   1. Environment setup (requirements install)
#   2. Data download + format  (resumable)
#   3. OCR adapter fine-tuning (resumable)
#   4. CLS adapter fine-tuning (resumable)
#   5. Evaluation of both adapters
#
# SAFE TO RE-RUN — every step checks if it was already completed.
# Progress is saved to logs/pipeline_state.json.
#
# Usage:
#   bash run_pipeline.sh                  # full run
#   bash run_pipeline.sh --skip-data      # skip download (data already present)
#   bash run_pipeline.sh --skip-train     # skip training (adapters already exist)
#   bash run_pipeline.sh --eval-only      # only run evaluation
# =============================================================================
set -euo pipefail

# ── Parse flags ───────────────────────────────────────────────────────────────
SKIP_DATA=false
SKIP_TRAIN=false
EVAL_ONLY=false
for arg in "$@"; do
  case $arg in
    --skip-data)  SKIP_DATA=true ;;
    --skip-train) SKIP_TRAIN=true ;;
    --eval-only)  EVAL_ONLY=true; SKIP_DATA=true; SKIP_TRAIN=true ;;
  esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$WORKDIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/pipeline_${TIMESTAMP}.log"
STATE_FILE="$LOG_DIR/pipeline_state.json"

mkdir -p "$LOG_DIR"

# ── Tee all output to log file ────────────────────────────────────────────────
exec > >(tee -a "$LOG_FILE") 2>&1

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die()  { log "ERROR: $*"; exit 1; }
hr()   { log "$(printf '─%.0s' {1..70})"; }

step_done() {
  local key="$1"
  python3 -c "
import json, sys
try:
    s = json.load(open('$STATE_FILE'))
    sys.exit(0 if s.get('$key') == 'done' else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

mark_done() {
  local key="$1"
  python3 -c "
import json, os
f = '$STATE_FILE'
s = json.load(open(f)) if os.path.exists(f) else {}
s['$key'] = 'done'
json.dump(s, open(f,'w'), indent=2)
"
}

# ── Start banner ──────────────────────────────────────────────────────────────
hr
log "IDP Research Pipeline"
log "Workdir  : $WORKDIR"
log "Log file : $LOG_FILE"
log "State    : $STATE_FILE"
log "Skip data: $SKIP_DATA  |  Skip train: $SKIP_TRAIN  |  Eval only: $EVAL_ONLY"
hr

# ── GPU info ──────────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    log "GPU info:"
    nvidia-smi --query-gpu=name,memory.total,driver_version \
               --format=csv,noheader | sed 's/^/  /'
else
    log "WARNING: nvidia-smi not found"
fi

# ── Python check ──────────────────────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python || die "python not found")
log "Python: $($PYTHON --version)"
cd "$WORKDIR"

# =============================================================================
# STEP 1 — Install requirements
# =============================================================================
hr
log "STEP 1: Installing requirements"

if step_done "requirements"; then
    log "  Requirements already installed — skipping"
else
    $PYTHON -m pip install --upgrade pip --quiet
    $PYTHON -m pip install -r requirements_h100.txt \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        2>&1 | grep -v "^Requirement already"

    # Optional: Flash Attention for extra speedup on H100
    if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        log "  Installing Flash Attention (may take 5-10 min to compile)..."
        $PYTHON -m pip install flash-attn --no-build-isolation --quiet 2>&1 \
            || log "  WARNING: flash-attn install failed — continuing without it"
    fi

    mark_done "requirements"
    log "  Requirements installed"
fi

# =============================================================================
# STEP 2 — Download and format datasets
# =============================================================================
hr
log "STEP 2: Download and format datasets"

if [ "$SKIP_DATA" = true ]; then
    log "  --skip-data set — skipping"
elif step_done "data_download"; then
    log "  Data already downloaded and formatted — skipping"
    log "  (delete '$STATE_FILE' key 'data_download' to re-run)"
else
    log "  Starting download (this will take 30-90 min for RVL-CDIP 320K)..."
    $PYTHON download_and_format_data.py --resume
    mark_done "data_download"
    log "  Data download + format complete"
fi

# ── Verify key files exist ────────────────────────────────────────────────────
if [ "$SKIP_TRAIN" = false ] && [ "$EVAL_ONLY" = false ]; then
    for f in formatted_data/rvlcdip/train.jsonl \
              formatted_data/iam/train.jsonl \
              formatted_data/sroie/train.jsonl; do
        [ -f "$f" ] || die "Expected $f not found. Run without --skip-data first."
    done
fi

# =============================================================================
# STEP 3 — Train OCR adapter
# =============================================================================
hr
log "STEP 3: Train OCR expert adapter (IAM + SROIE, ~7K samples, 5 epochs)"

if [ "$SKIP_TRAIN" = true ]; then
    log "  --skip-train set — skipping"
elif step_done "train_ocr"; then
    log "  OCR adapter already trained — skipping"
    log "  (delete '$STATE_FILE' key 'train_ocr' to re-run)"
else
    log "  Training OCR adapter..."
    $PYTHON finetune_qlora.py --config finetune_config_lora_ocr.yaml --resume

    # Verify adapter was saved
    ADAPTER_PATH="finetune_outputs/lora_ocr/qwen25vl_3b/final_adapter"
    [ -d "$ADAPTER_PATH" ] || die "OCR adapter not found at $ADAPTER_PATH after training"

    # Find and log the best checkpoint
    log "  Checking best checkpoint..."
    $PYTHON - <<'EOF'
import json
from pathlib import Path
state = Path("finetune_outputs/lora_ocr/qwen25vl_3b/trainer_state.json")
if state.exists():
    history = json.loads(state.read_text())["log_history"]
    best = min((x for x in history if "eval_loss" in x), key=lambda x: x["eval_loss"], default=None)
    if best:
        print(f"  Best checkpoint: step={best['step']}  eval_loss={best['eval_loss']:.4f}")
EOF

    mark_done "train_ocr"
    log "  OCR adapter training complete"
fi

# =============================================================================
# STEP 4 — Train CLS adapter
# =============================================================================
hr
log "STEP 4: Train CLS expert adapter (RVL-CDIP 320K + Tobacco, 2 epochs)"

if [ "$SKIP_TRAIN" = true ]; then
    log "  --skip-train set — skipping"
elif step_done "train_cls"; then
    log "  CLS adapter already trained — skipping"
    log "  (delete '$STATE_FILE' key 'train_cls' to re-run)"
else
    log "  Training CLS adapter (estimated ~60-90 min on H100)..."
    $PYTHON finetune_qlora.py --config finetune_config_lora_cls.yaml --resume

    ADAPTER_PATH="finetune_outputs/lora_cls/qwen25vl_3b/final_adapter"
    [ -d "$ADAPTER_PATH" ] || die "CLS adapter not found at $ADAPTER_PATH after training"

    log "  Checking best checkpoint..."
    $PYTHON - <<'EOF'
import json
from pathlib import Path
state = Path("finetune_outputs/lora_cls/qwen25vl_3b/trainer_state.json")
if state.exists():
    history = json.loads(state.read_text())["log_history"]
    best = min((x for x in history if "eval_loss" in x), key=lambda x: x["eval_loss"], default=None)
    if best:
        print(f"  Best checkpoint: step={best['step']}  eval_loss={best['eval_loss']:.4f}")
EOF

    mark_done "train_cls"
    log "  CLS adapter training complete"
fi

# =============================================================================
# STEP 5 — Evaluate
# =============================================================================
hr
log "STEP 5: Evaluate both adapters"

if step_done "evaluation"; then
    log "  Evaluation already done — skipping"
    log "  (delete '$STATE_FILE' key 'evaluation' to re-run)"
else
    log "  Building eval datasets..."
    $PYTHON build_eval_datasets.py 2>/dev/null || log "  WARNING: build_eval_datasets.py not found — skipping build step"

    log "  Running evaluation..."
    $PYTHON evaluate_all_models.py \
        --models qwen25vl_3b_lora_ocr qwen25vl_3b_lora_cls \
        --tasks ocr classification \
        --resume

    mark_done "evaluation"
    log "  Evaluation complete — results in eval_results/summary.csv"
fi

# ── Final summary ─────────────────────────────────────────────────────────────
hr
log "Pipeline complete!"
log ""
log "Outputs:"
log "  OCR adapter     : finetune_outputs/lora_ocr/qwen25vl_3b/final_adapter/"
log "  CLS adapter     : finetune_outputs/lora_cls/qwen25vl_3b/final_adapter/"
log "  Eval results    : eval_results/summary.csv"
log "  TensorBoard     : tensorboard --logdir finetune_outputs/"
log "  Full log        : $LOG_FILE"
hr

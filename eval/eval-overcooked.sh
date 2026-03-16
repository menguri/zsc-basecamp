#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="$ROOT_DIR/eval"
EVAL_CODE_DIR="$EVAL_DIR/eval_code"

EVAL_MODE="${EVAL_MODE:-xp}"
LAYOUT="${LAYOUT:-cramped_room}"
ALGO0="${ALGO0:-ph2}"
ALGO1="${ALGO1:-bc}"

MODELS_ROOT="${MODELS_ROOT:-$EVAL_DIR/models/Overcooked}"
RESULTS_ROOT="${RESULTS_ROOT:-$EVAL_DIR/results}"

# Expected model layout:
#   <MODELS_ROOT>/<layout>/<algorithm>/run_x/...
# Result layout:
#   <RESULTS_ROOT>/<layout>/xp_<algo0>_<algo1>/{csv,gif}/...
# Visualization:
#   VIZ=1 enables gif dump per evaluated pair into the gif directory.
mkdir -p "$MODELS_ROOT" "$RESULTS_ROOT"

EVAL_STEPS="${EVAL_STEPS:-400}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
N_EVAL_THREADS="${N_EVAL_THREADS:-5}"
EVAL_SEEDS="${EVAL_SEEDS:-5}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
OVERWRITE="${OVERWRITE:-0}"
GENERATE_SUMMARY="${GENERATE_SUMMARY:-1}"
VIZ="${VIZ:-0}"

CMD=(
  python3 "$EVAL_CODE_DIR/xp_eval.py"
  --repo_root "$ROOT_DIR"
  --models_root "$MODELS_ROOT"
  --results_root "$RESULTS_ROOT"
  --layout "$LAYOUT"
  --algo0 "$ALGO0"
  --algo1 "$ALGO1"
  --eval_mode "$EVAL_MODE"
  --eval_steps "$EVAL_STEPS"
  --eval_episodes "$EVAL_EPISODES"
  --n_eval_threads "$N_EVAL_THREADS"
  --eval_seeds "$EVAL_SEEDS"
)

if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
  CMD+=(--cuda_visible_devices "$CUDA_VISIBLE_DEVICES")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi
if [[ "$VIZ" == "1" ]]; then
  CMD+=(--viz)
fi

"${CMD[@]}"

if [[ "$GENERATE_SUMMARY" == "1" ]]; then
  python3 "$EVAL_DIR/generate_summary.py" --results_root "$RESULTS_ROOT"
fi

#!/bin/bash

#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=4:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/G2VLM/logs/run_bench_eval_%j.out

# Usage:
#   sbatch run_bench_eval.sh vsi            # full 200-sample VSI run
#   sbatch run_bench_eval.sh vsti           # full 200-sample VSTI run
#   sbatch run_bench_eval.sh vsi 2          # smoke test: first 2 samples
# Optional 3rd arg overrides num-frames (default 32).

set -euo pipefail

BENCH="${1:?usage: run_bench_eval.sh <vsi|vsti> [limit] [num_frames]}"
LIMIT="${2:-0}"
NUM_FRAMES="${3:-32}"

REPO_DIR="/cluster/work/igp_psr/spanwar/G2VLM"
EVAL_DIR="$REPO_DIR/eval_benchmarks"
SUBSET="$EVAL_DIR/${BENCH}_random200.json"
TAG="$(date +%d%m-%H%M%S)"
OUT="$EVAL_DIR/preds_${BENCH}_${NUM_FRAMES}f_${TAG}.json"

mv "$REPO_DIR/logs/run_bench_eval_${SLURM_JOB_ID}.out" \
   "$REPO_DIR/logs/run_bench_eval_${BENCH}_${TAG}_${SLURM_JOB_ID}.out" 2>/dev/null || true

source /cluster/work/igp_psr/spanwar/envs/g2vlm_env/bin/activate

# VSI video paths live under this HF cache root; harmless for VSTI.
export HF_HOME=/cluster/scratch/spanwar/datasets/eval_cache/hf-home

nvidia-smi
echo "benchmark=$BENCH limit=$LIMIT num_frames=$NUM_FRAMES"
echo "subset=$SUBSET"
echo "output=$OUT"

START=$(date +%s)
python -u "$EVAL_DIR/eval_g2vlm_bench.py" \
    --subset "$SUBSET" \
    --benchmark "$BENCH" \
    --out "$OUT" \
    --num-frames "$NUM_FRAMES" \
    --limit "$LIMIT" \
    --model-path "$REPO_DIR/models/G2VLM-2B-MoT"
END=$(date +%s)
echo "Eval ($BENCH) took $((END - START)) seconds -> $OUT"

# Auto-score (CPU-only, fast) once predictions are written.
python -u "$EVAL_DIR/score_results.py" --predictions "$OUT" --benchmark "$BENCH" || true

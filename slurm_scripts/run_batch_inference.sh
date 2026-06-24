#!/bin/bash

#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=2:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/G2VLM/logs/run_batch_inference_%j.out

REPO_DIR="/cluster/work/igp_psr/spanwar/G2VLM"
SCENES_DIR="/cluster/work/igp_psr/spanwar/inference_tests"
PROMPTS_FILE="$SCENES_DIR/scene_prompts.json"
OUTPUT_FILE="$SCENES_DIR/results_$(date +%d%m-%H%M%S).json"

mv "$REPO_DIR/logs/run_batch_inference_${SLURM_JOB_ID}.out" \
   "$REPO_DIR/logs/run_batch_inference_$(date +%d%m-%H%M%S)_${SLURM_JOB_ID}.out"

source /cluster/work/igp_psr/spanwar/envs/g2vlm_env/bin/activate

nvidia-smi

echo "Output will be saved to: $OUTPUT_FILE"

START=$(date +%s)

python -u "$REPO_DIR/inference_batch.py" \
    --model-path "$REPO_DIR/models/G2VLM-2B-MoT" \
    --prompts-file "$PROMPTS_FILE" \
    --scenes-dir "$SCENES_DIR" \
    --output-file "$OUTPUT_FILE"

END=$(date +%s)
echo "Batch inference took $((END - START)) seconds"

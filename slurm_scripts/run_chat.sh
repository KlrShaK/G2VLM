#!/bin/bash

#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=1:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/G2VLM/logs/run_chat_%j.out

REPO_DIR="/cluster/work/igp_psr/spanwar/G2VLM"
mv "$REPO_DIR/logs/run_chat_${SLURM_JOB_ID}.out" "$REPO_DIR/logs/run_chat_$(date +%d%m-%H%M%S)_${SLURM_JOB_ID}.out"

source /cluster/work/igp_psr/spanwar/envs/g2vlm_env/bin/activate

nvidia-smi

START=$(date +%s)
# python -u "$REPO_DIR/inference_chat.py" \
#     --model-path "$REPO_DIR/models/G2VLM-2B-MoT" \
#     --image-path "$REPO_DIR/examples/25_0.jpg" \
#     --question "Your question here"

if [ $# -eq 0 ]; then
    python -u "$REPO_DIR/inference_chat.py" \
        --model-path "$REPO_DIR/models/G2VLM-2B-MoT" \
        --image-path "$REPO_DIR/examples/prs_view1.jpg" "$REPO_DIR/examples/prs_view2.jpg" \
        --question "What movement would produce the second image from the perspective of the first?"
else
    python -u "$REPO_DIR/inference_chat.py" \
        --model-path "$REPO_DIR/models/G2VLM-2B-MoT" \
        "$@"
fi

END=$(date +%s)
echo "Chat inference took $((END - START)) seconds"


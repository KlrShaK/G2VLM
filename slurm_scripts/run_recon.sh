#!/bin/bash
#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=1:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/G2VLM/logs/run_recon_%j.out

REPO_DIR="/cluster/work/igp_psr/spanwar/G2VLM"
mv "$REPO_DIR/logs/run_recon_${SLURM_JOB_ID}.out" "$REPO_DIR/logs/run_recon_$(date +%d%m-%H%M%S)_${SLURM_JOB_ID}.out"

source /cluster/work/igp_psr/spanwar/envs/g2vlm_env/bin/activate

nvidia-smi

START=$(date +%s)
python -u "$REPO_DIR/inference_recon.py" --model_path "$REPO_DIR/models/G2VLM-2B-MoT"
END=$(date +%s)
echo "Reconstruction took $((END - START)) seconds"

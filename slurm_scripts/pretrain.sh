#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_a100_80gb_pcie:8
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=72:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/G2VLM/logs/pretrain_%j.out

REPO_DIR="/cluster/work/igp_psr/spanwar/G2VLM"
mv "$REPO_DIR/logs/pretrain_${SLURM_JOB_ID}.out" "$REPO_DIR/logs/pretrain_$(date +%d%m-%H%M%S)_${SLURM_JOB_ID}.out"

source /cluster/work/igp_psr/spanwar/envs/g2vlm_env/bin/activate

nvidia-smi

srun bash "$REPO_DIR/scripts/pretrain.sh"

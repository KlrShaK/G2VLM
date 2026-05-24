#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_a100_80gb_pcie:8
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=72:00:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/G2VLM/logs/%j.out

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

source /cluster/work/igp_psr/spanwar/envs/g2vlm_env/bin/activate

nvidia-smi

srun bash "$REPO_DIR/scripts/joint_train.sh"

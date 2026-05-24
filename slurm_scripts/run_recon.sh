#!/bin/bash
#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%j.out

source /cluster/work/igp_psr/spanwar/envs/g2vlm_env/bin/activate

nvidia-smi

START=$(date +%s)
python -u inference_recon.py --model_path models/G2VLM-2B-MoT
END=$(date +%s)
echo "Reconstruction took $((END - START)) seconds"

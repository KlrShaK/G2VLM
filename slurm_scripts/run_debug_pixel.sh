#!/bin/bash
#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=0:30:00
#SBATCH --output=/cluster/work/igp_psr/spanwar/G2VLM/logs/debug_pixel_%j.out
source /cluster/work/igp_psr/spanwar/envs/g2vlm_env/bin/activate
export HF_HOME=/cluster/scratch/spanwar/datasets/eval_cache/hf-home
python -u /cluster/work/igp_psr/spanwar/G2VLM/eval_benchmarks/debug_pixel.py

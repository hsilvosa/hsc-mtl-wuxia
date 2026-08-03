#!/bin/bash

#SBATCH --job-name=wuxia_llama

#SBATCH --output=logs/llama_%j.o
#SBATCH --error=logs/llama_%j.e

#SBATCH -t 5:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH -c 64
#SBATCH --mem-per-cpu=3G 

#SBATCH --mail-type=ALL


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
cd "$PROJECT_ROOT"

module load miniconda3
conda activate "${WUXIA_CONDA_ENV:-wuxia}"



export CUDA_LAUNCH_BLOCKING=1
export WANDB_MODE="disabled"
# Set HUGGING_FACE_HUB_TOKEN in the submission environment when required.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"


TQDM_DISABLE=1 python src/LLM/hpc/scripts/llama3_train.py --prompts 0

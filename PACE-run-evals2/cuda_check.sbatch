#!/bin/bash
#SBATCH -J fedastro-cuda
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:l40s:1
#SBATCH -t 00:30:00
#SBATCH -o logs/cuda-%j.out
export SCR=/storage/ice1/4/7/aagarwal710
export PATH="$SCR/uv/bin:$PATH" UV_CACHE_DIR=$SCR/uv/cache UV_PYTHON_INSTALL_DIR=$SCR/uv/python
export PYTHONUNBUFFERED=1
cd $SCR/FLASH
nvidia-smi
uv run --frozen python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
uv run --frozen fedastro run configs/experiments/cuda_check.yaml
uv run --frozen pytest tests/test_fl.py -k cuda -rs -q

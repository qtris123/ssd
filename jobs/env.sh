#!/usr/bin/env bash
# Shared environment configuration for all SSD SLURM jobs.
# Source this file at the top of every job script:
#   source "$(dirname "$0")/env.sh"

# ── Paths ────────────────────────────────────────────────────────────────────
export SSD_ROOT="$HOME/ssd"
export SSD_HF_CACHE="/scratch/gilbreth/$USER/huggingface/hub"
export SSD_DATASET_DIR="/scratch/gilbreth/$USER/datasets/processed_datasets"
export HF_DATASETS_CACHE="/scratch/gilbreth/$USER/datasets"

# 8.0 = A100, 8.9 = L40/4090, 9.0 = H100
export SSD_CUDA_ARCH="8.0"

# ── Cache directories on scratch (avoid NFS issues & quota) ─────────────────
export TRITON_CACHE_DIR="/scratch/gilbreth/$USER/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="/scratch/gilbreth/$USER/torchinductor_cache"
export HF_HOME="/scratch/gilbreth/$USER/huggingface"
export UV_CACHE_DIR="/scratch/gilbreth/$USER/.cache/uv"
export UV_LINK_MODE=copy

# ── Activate the SSD virtualenv ─────────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"
eval "$(conda shell.bash hook)"
source "$SSD_ROOT/.venv/bin/activate"

# ── Pre-create cache dirs ───────────────────────────────────────────────────
mkdir -p "$SSD_HF_CACHE" "$SSD_DATASET_DIR" \
         "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

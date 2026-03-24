#!/bin/bash
#SBATCH --account=ai-forge
#SBATCH --partition=a100-40gb
#SBATCH --nodes=1
#SBATCH --qos=standby
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=ssd-download
#SBATCH --output=ssd-download.out
#SBATCH --error=ssd-download.err

# Downloads only Llama-8B + Llama-1B (draft) to scratch.
# Total ≈ 18.5 GB. Datasets already downloaded.
#
# BEFORE SUBMITTING: you need a HuggingFace token with Meta Llama access.
#   1. Go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
#      and accept Meta's license agreement
#   2. Run on the login node:
#        pip install -q huggingface_hub
#        huggingface-cli login
#
# Submit:  cd ~/ssd && sbatch jobs/01_download.slurm

echo "============================================"
echo "  SSD — Download Llama-8B + 1B draft"
echo "============================================"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURM_NODELIST"
echo "Time    : $(date)"
echo "============================================"

set -euo pipefail
cd "$HOME/ssd"

source jobs/env.sh

uv sync --extra scripts

python -c "
from huggingface_hub import snapshot_download
import os
cache = os.environ['SSD_HF_CACHE']
for model in ['meta-llama/Llama-3.1-8B-Instruct', 'meta-llama/Llama-3.2-1B-Instruct']:
    print(f'Downloading {model}...')
    snapshot_download(repo_id=model, cache_dir=cache, resume_download=True)
    print(f'  done')
"

echo ""
echo "Verifying downloads..."
ls -dsh "$SSD_HF_CACHE"/models--meta-llama--Llama-3.1-8B-Instruct 2>/dev/null
ls -dsh "$SSD_HF_CACHE"/models--meta-llama--Llama-3.2-1B-Instruct 2>/dev/null

echo ""
echo "============================================"
echo "  Done. Time: $(date)"
echo "============================================"

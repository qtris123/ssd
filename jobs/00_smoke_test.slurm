#!/bin/bash
#SBATCH --account=ai-forge
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --job-name=ssd-smoke
#SBATCH --output=ssd-smoke-%j.out
#SBATCH --error=ssd-smoke-%j.err

# Quick validation: loads Llama-8B on 1× A30, generates 4 prompts.
#
# Submit:  cd ~/ssd && sbatch jobs/00_smoke_test.slurm

echo "============================================"
echo "  SSD Smoke Test — Llama-8B on A30"
echo "============================================"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURM_NODELIST"
echo "GPU     : $CUDA_VISIBLE_DEVICES"
echo "Time    : $(date)"
echo "============================================"

set -euo pipefail
cd "$HOME/ssd"

source jobs/env.sh

nvidia-smi

echo ""
echo "── Python import test ──"
python -c "from ssd import LLM; print('SSD import: ok')"

echo ""
echo "── Quick AR generation (Llama-8B, 4 prompts, 64 tokens) ──"
python -O bench/bench.py \
    --llama --size 8 \
    --gpus 1 \
    --b 1 \
    --temp 0 \
    --numseqs 4 \
    --output_len 64 \
    --example

echo ""
echo "============================================"
echo "  Smoke test passed. Time: $(date)"
echo "============================================"

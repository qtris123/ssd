#!/bin/bash
#SBATCH --account=ai-forge
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --job-name=ssd-llama8b
#SBATCH --output=ssd-llama8b-%j.out
#SBATCH --error=ssd-llama8b-%j.err

# Llama-3.1-8B benchmarks on 2× A30 (48 GB VRAM total).
#
# Runs three modes back-to-back:
#   1. Autoregressive (AR)         — 1 GPU
#   2. Synchronous Spec Decode     — 1 GPU, k=6
#   3. Asynchronous SSD            — 2 GPUs (1 target + 1 draft), k=7, f=3
#
# Model memory: 8B BF16 ≈ 16 GB + 1B draft ≈ 2 GB → fits in A30 (24 GB each)
#
# Submit:  cd ~/ssd && sbatch jobs/03_bench_llama8b.slurm

echo "============================================"
echo "  SSD Benchmark — Llama-3.1-8B on A30"
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

BENCH_ARGS="--llama --size 8 --b 1 --temp 0 --numseqs 128 --output_len 512 --all"

# ── 1. Autoregressive baseline ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "  [1/3] Llama-8B — Autoregressive (1 GPU)"
echo "============================================"
python -O bench/bench.py $BENCH_ARGS --gpus 1

pkill -9 -f "bench.py" 2>/dev/null || true
sleep 5

# ── 2. Synchronous speculative decoding ─────────────────────────────────────
echo ""
echo "============================================"
echo "  [2/3] Llama-8B — Sync Spec Decode (1 GPU, k=6)"
echo "============================================"
python -O bench/bench.py $BENCH_ARGS --gpus 1 --spec --k 6

pkill -9 -f "bench.py" 2>/dev/null || true
sleep 5

# ── 3. Async speculative decoding (SSD) ─────────────────────────────────────
echo ""
echo "============================================"
echo "  [3/3] Llama-8B — Async SSD (2 GPUs, k=7, f=3)"
echo "============================================"
python -O bench/bench.py $BENCH_ARGS --gpus 2 --spec --async --k 7 --f 3

echo ""
echo "============================================"
echo "  All Llama-8B benchmarks complete. Time: $(date)"
echo "============================================"

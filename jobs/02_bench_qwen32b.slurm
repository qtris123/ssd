#!/bin/bash
#SBATCH --account=ai-forge
#SBATCH --partition=a100-40gb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=240G
#SBATCH --time=6:00:00
#SBATCH --job-name=ssd-qwen
#SBATCH --output=ssd-qwen.out
#SBATCH --error=ssd-qwen.err

# Qwen3-32B benchmarks on 4× A100-40GB (160 GB VRAM total).
#
# Runs three modes back-to-back:
#   1. Autoregressive (AR)         — 4 GPUs, TP=4
#   2. Synchronous Spec Decode     — 4 GPUs, TP=4, k=6
#   3. Asynchronous SSD            — 4 GPUs (3 target + 1 draft), k=7, f=3
#
# Model memory: 32B BF16 ≈ 64 GB → 16 GB/GPU with TP=4 (comfortable)
#
# Submit:  cd ~/ssd && sbatch jobs/02_bench_qwen32b.slurm

echo "============================================"
echo "  SSD Benchmark — Qwen3-32B"
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

BENCH_ARGS="--qwen --size 32 --b 1 --temp 0 --numseqs 128 --output_len 512 --all"

# ── 1. Autoregressive baseline ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "  [1/3] Qwen3-32B — Autoregressive (4 GPUs)"
echo "============================================"
python -O bench/bench.py $BENCH_ARGS --gpus 4

pkill -9 -f "bench.py" 2>/dev/null || true
sleep 5

# ── 2. Synchronous speculative decoding ─────────────────────────────────────
echo ""
echo "============================================"
echo "  [2/3] Qwen3-32B — Sync Spec Decode (4 GPUs, k=6)"
echo "============================================"
python -O bench/bench.py $BENCH_ARGS --gpus 4 --spec --k 6

pkill -9 -f "bench.py" 2>/dev/null || true
sleep 5

# ── 3. Async speculative decoding (SSD) ─────────────────────────────────────
echo ""
echo "============================================"
echo "  [3/3] Qwen3-32B — Async SSD (4 GPUs, k=7, f=3)"
echo "============================================"
python -O bench/bench.py $BENCH_ARGS --gpus 4 --spec --async --k 7 --f 3

echo ""
echo "============================================"
echo "  All Qwen3-32B benchmarks complete. Time: $(date)"
echo "============================================"
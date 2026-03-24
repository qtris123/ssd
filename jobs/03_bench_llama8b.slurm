#!/bin/bash
#SBATCH --account=ai-forge
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH --job-name=ssd-llama
#SBATCH --output=ssd-llama_alpaca.out
#SBATCH --error=ssd-llama_alpaca.err

# Llama-3.1-8B benchmarks on 2× A100-40GB (80 GB VRAM total).
#
# Runs three modes back-to-back:
#   1. Autoregressive (AR)         — 1 GPU
#   2. Synchronous Spec Decode     — 1 GPU, k=6
#   3. Asynchronous SSD            — 2 GPUs (1 target + 1 draft), k=7, f=3
#
# Model memory: 8B BF16 ≈ 16 GB + 1B draft ≈ 2 GB → fits in A100-40GB (80 GB each)
#
# Submit:  cd ~/ssd && sbatch jobs/03_bench_llama8b.slurm

echo "============================================"
echo "  SSD Benchmark — Llama-3.1-8B on A100-40GB"
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

# --- [FIXED] GPU MONITORING START ---
GPU_LOG="gpu_usage_${SLURM_JOB_ID}.csv"

# 1. Initialize the header
echo "timestamp, gpu_idx, gpu_util, mem_used, mem_total" > "$GPU_LOG"

# 2. Start monitoring every 3 seconds
# Added 'stdbuf -oL' to force line-buffering so you can see logs in real-time
# Added 'index' to query so you can distinguish between the 2 GPUs
stdbuf -oL nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits -l 3 >> "$GPU_LOG" &

MONITOR_PID=$!
echo "GPU Monitoring started (PID: $MONITOR_PID) every 3s, logging to $GPU_LOG"
# --- GPU MONITORING END ---

BENCH_ARGS="--llama --size 8 --b 1 --temp 0 --numseqs 128 --output_len 512 --ultrafeedback --wandb --name SSD_ultrafeedback --group SSD" #--name

# # ── 1. Autoregressive baseline ──────────────────────────────────────────────
# echo ""
# echo "============================================"
# echo "  [1/3] Llama-8B — Autoregressive (1 GPU)"
# echo "============================================"
# python -O bench/bench.py $BENCH_ARGS --gpus 1

# pkill -9 -f "bench.py" 2>/dev/null || true
# sleep 5

# # ── 2. Synchronous speculative decoding ─────────────────────────────────────
# echo ""
# echo "============================================"
# echo "  [2/3] Llama-8B — Sync Spec Decode (1 GPU, k=6)"
# echo "============================================"
# python -O bench/bench.py $BENCH_ARGS --gpus 2 --spec --k 6

# pkill -9 -f "bench.py" 2>/dev/null || true
# sleep 5

# ── 3. Async speculative decoding (SSD) ─────────────────────────────────────
echo ""
echo "============================================"
echo "  [3/3] Llama-8B — Async SSD (2 GPUs, k=7, f=3)"
echo "============================================"
python -O bench/bench.py $BENCH_ARGS --gpus 2 --spec --async --k 7 --f 3

# echo ""
# echo "============================================"
# echo "  All Llama-8B benchmarks complete. Time: $(date)"
# echo "============================================"

# --- [NEW] CLEANUP MONITORING ---
# --- [FIXED] CLEANUP MONITORING ---
# Use a trap or explicit kill to ensure the background process dies
if ps -p $MONITOR_PID > /dev/null; then
    kill $MONITOR_PID
fi
echo "============================================"
echo "  Benchmark complete. GPU log saved to $GPU_LOG"
echo "  Time: $(date)"
echo "============================================"
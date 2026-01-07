#!/bin/bash

# cd /home/ubuntu/workspace/jeff/sglang/python
source .venv/bin/activate 
# lsof -ti:30000 | xargs kill -9
ls /dev/shm | grep -E '^(draft_group|target_group)$' >/dev/null 2>&1 && sudo rm -f /dev/shm/draft_group /dev/shm/target_group
CUDA_VISIBLE_DEVICES=0,1 \
SGLANG_NANO_PEARL_ALLOW_OVERLAP=0 \
NANO_PEARL_GAMMA=5 \
NANO_PEARL_ACCEPT_RATE_LOG_INTERVAL=200 \
NANO_PEARL_ADAPT_GAMMA=1 \
NANO_PEARL_ACCEPT_RATE_LOW=0.25 \
NANO_PEARL_ACCEPT_RATE_HIGH=0.55 \
NANO_PEARL_GAMMA_MIN=5 \
NANO_PEARL_GAMMA_MAX=8 \
NANO_PEARL_SGLANG_WAIT_TIMEOUT_LIMIT=0 \
NANO_PEARL_SGLANG_KICK_ON_TIMEOUT=1 \
NANO_PEARL_TEMPERATURE_SCALE=0.5 \
NANO_PEARL_SGLANG_PREFETCH_STEPS=8 \
NANO_PEARL_STREAM_BARRIER_INTERVAL=1 \
NANO_PEARL_SGLANG_PREFETCH_FLUSH_STEPS=2 \
NANO_PEARL_SGLANG_STREAM_WAIT_TIMEOUT_S=60 \
NANO_PEARL_SGLANG_WAIT_TIMEOUT_S=10 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m sglang.launch_server \
  --model-path ~/models/Qwen/Qwen2.5-14B  \
  --served-model-name test \
  --enable-nano-pearl \
  --draft-model-path  ~/models/Qwen/Qwen2.5-0.5B \
  --draft-model-tp-size 1 \
  --tensor-parallel-size 1 \
  --max-running-requests 8 \
  --stream-interval 128 \
  --schedule-conservativeness 0.3 \
  --port 12470 \
  --mem-fraction-static 0.85


# python -m sglang.launch_server \
#   --model-path ~/models/Qwen/Qwen3-1.7B \
#   --served-model-name test \
#   --tensor-parallel-size 2 \
#   --port 12470 \
#   --mem-fraction-static 0.95

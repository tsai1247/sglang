#!/bin/bash

# cd /home/ubuntu/workspace/jeff/sglang/python
source .venv/bin/activate 
# lsof -ti:30000 | xargs kill -9
ls /dev/shm | grep -E '^(draft_group|target_group)$' >/dev/null 2>&1 && sudo rm -f /dev/shm/draft_group /dev/shm/target_group
CUDA_VISIBLE_DEVICES=0,1 \
python -m sglang.launch_server \
  --model-path ~/models/Qwen/Qwen3-32B \
  --served-model-name test \
  --enable-nano-pearl \
  --draft-model-path ~/models/Qwen/Qwen3-1.7B \
  --nano-pearl-share-gpus \
  --nano-pearl-target-tp-size 2 \
  --draft-model-tp-size 2 \
  --tensor-parallel-size 1 \
  --nano-pearl-max-num-batched-tokens 8192 \
  --nano-pearl-max-num-seqs 64 \
  --nano-pearl-gpu-memory-utilization 0.95 \
  --mem-fraction-static 0.95 \
  --stream-interval 4 \
  --schedule-conservativeness 0.7 \
  --port 12470

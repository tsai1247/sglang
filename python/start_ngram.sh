#!/bin/bash

# cd /home/ubuntu/workspace/jeff/sglang/python
source .venv/bin/activate 
# lsof -ti:30000 | xargs kill -9
ls /dev/shm | grep -E '^(draft_group|target_group)$' >/dev/null 2>&1 && sudo rm -f /dev/shm/draft_group /dev/shm/target_group
python -m sglang.launch_server \
  --model-path ~/models/Qwen/Qwen2.5-14B \
  --served-model-name test \
  --tensor-parallel-size 1 \
  --port 12470 \
  --mem-fraction-static 0.85 \
  --speculative-algorithm NGRAM \
  --speculative-num-steps 5 \
  --max-model-len 32768
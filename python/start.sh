#!/bin/bash

# cd /home/ubuntu/workspace/jeff/sglang/python
source .venv/bin/activate 
# lsof -ti:30000 | xargs kill -9
ls /dev/shm | grep -E '^(draft_group|target_group)$' >/dev/null 2>&1 && sudo rm -f /dev/shm/draft_group /dev/shm/target_group
python -m sglang.launch_server --model-path ~/models/Qwen/Qwen3-32B --served-model-name test --enable-nano-pearl --draft-model-path ~/models/Qwen/Qwen3-1.7B \
    --mem-fraction-static 0.95 \
    --port 12470 \
    --tensor-parallel-size 1 --draft-model-tp-size 1


#!/bin/bash

source .venv/bin/activate
python -m sglang.launch_server \
  --model-path ~/models/Qwen/Qwen3-32B \
  --served-model-name test \
  --speculative-algorithm SLRS \
  --speculative-draft-model-path ~/models/meta-llama/Llama-3.2-1B-Instruct

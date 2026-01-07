#!/bin/bash

# cd /home/ubuntu/workspace/jeff/sglang/python
source .venv/bin/activate 
python -m sglang.launch_server --model-path ~/models/Qwen/Qwen3-32B --served-model-name test

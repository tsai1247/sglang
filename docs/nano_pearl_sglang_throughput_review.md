# nano-pearl + sglang throughput review

Checked paths:
- python/sglang/srt/server_args.py
- python/sglang/srt/managers/tp_worker.py
- python/sglang/srt/managers/scheduler_output_processor_mixin.py
- python/sglang/srt/managers/scheduler.py
- python/sglang/srt/entrypoints/http_server.py
- python/start_nano.sh
- python/nano-PEARL/nano_pearl/pearl_engine/pearl_engine.py
- python/nano-PEARL/nano_pearl/pearl_engine/pearl_model_runner.py
- python/nano-PEARL/nano_pearl/pearl_engine/scheduler.py

Fixes applied:
1) Reduce IPC overhead between sglang and nano-pearl when not streaming.
   - The nano-pearl worker now batches PEARL steps via stream_generate_steps when there are no streaming requests, which cuts per-step shared-memory/event round-trips while keeping streaming latency unchanged.
   - Path: python/sglang/srt/managers/tp_worker.py

2) Clean up per-request nano-pearl bookkeeping on completion.
   - Finished requests now clear _nano_pearl_active/_nano_pearl_pending_tokens/_nano_pearl_generated and related warning sets to prevent unbounded growth and extra lookups during long-running high-QPS sessions.
   - Path: python/sglang/srt/managers/tp_worker.py
   - Hook: python/sglang/srt/managers/scheduler_output_processor_mixin.py

3) Cancel nano-pearl sequences when requests are aborted in sglang.
   - Aborted requests now notify nano-pearl to drop the matching seq_id, preventing wasted GPU work on requests that the scheduler has already terminated.
   - Paths: python/sglang/srt/managers/scheduler.py, python/sglang/srt/managers/tp_worker.py
   - nano-PEARL hooks: python/nano-PEARL/nano_pearl/pearl_engine/pearl_engine.py, python/nano-PEARL/nano_pearl/pearl_engine/pearl_model_runner.py, python/nano-PEARL/nano_pearl/pearl_engine/scheduler.py

4) Avoid emitting fallback tokens when nano-pearl is still running.
   - If a request has no tokens ready but the nano-pearl engine is still active, we now return a sentinel to skip output rather than injecting an EOS fallback token, preventing truncated responses under load.
   - Path: python/sglang/srt/managers/tp_worker.py

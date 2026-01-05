# nano-PEARL Integration Notes

## 目的與需求
- 目標：在 sglang 啟動時能同時載入草稿與驗證模型，並使用 nano-PEARL 的平行解碼流程，不依賴原本的 speculative decoding。
- 需求：以 `--enable-nano-pearl` 啟動後，tp_worker / model_runner 應啟用 PEARLEngine，並實際走 nano-PEARL 的 generate 流程。

## 目前使用的啟動方式
- bash start.sh
  - 裡面包含以下關鍵程式碼：
  
```bash
python -m sglang.launch_server \
  --model-path ~/models/Qwen/Qwen3-1.7B \
  --served-model-name test \
  --enable-nano-pearl \
  --draft-model-path ~/models/Qwen/Qwen3-0.6B
```

## 推薦啟動參數（目前實測）
```bash
CUDA_VISIBLE_DEVICES=0,1 \
SGLANG_NANO_PEARL_ALLOW_OVERLAP=1 \
NANO_PEARL_GAMMA=6 \
NANO_PEARL_SGLANG_PREFETCH_STEPS=8 \
NANO_PEARL_SGLANG_PREFETCH_FLUSH_STEPS=2 \
NANO_PEARL_SGLANG_STREAM_WAIT_TIMEOUT_S=0.5 \
NANO_PEARL_SGLANG_WAIT_TIMEOUT_S=10 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m sglang.launch_server \
  --model-path ~/models/Qwen/Qwen3-32B \
  --served-model-name test \
  --enable-nano-pearl \
  --draft-model-path ~/models/Qwen/Qwen3-1.7B \
  --draft-model-tp-size 1 \
  --tensor-parallel-size 1 \
  --max-running-requests 8 \
  --stream-interval 32 \
  --schedule-conservativeness 0.5 \
  --port 12470 \
  --mem-fraction-static 0.95
```

## 主要修改與完成內容
### Server args / CLI
- `python/sglang/srt/server_args.py`
  - 新增 nano-pearl 初始化處理（draft path / revision 設定、忽略 speculative_algorithm）。
  - `--enable-nano-pearl` 改為 `store_true`。
  - 支援 `--speculative-algorithm nano_pearl` 作為等效開關。

### nano-PEARL API 補強
- `python/nano-PEARL/nano_pearl/pearl_engine/pearl_engine.py`
  - `add_request()` 回傳 `seq_id` 以便外部對應 req。
  - 新增 `stream_generate_step()` 回傳單步 token chunk，供 sglang 逐步整合。

### TpModelWorker 對接 PEARLEngine
- `python/sglang/srt/managers/tp_worker.py`
  - 初始化 `PEARLEngine`（含配置轉換、路徑補齊）。
  - `forward_batch_generation()` 在 nano-pearl 模式下走 PEARLEngine 路徑。
  - 將 SGLang 的 SamplingParams 轉成 nano-PEARL SamplingParams。
  - token 由 PEARL 產出後，回填給 Scheduler 的 `next_token_ids`。

### NanoPearlRunner / 初始化修復
- `python/sglang/srt/model_executor/model_runner.py`
  - 修正 world group 初始化順序（target 先建）。
  - 移除錯誤的 `attention_backend` 屬性存取。

### nano-PEARL Streaming 支援
- `python/nano-PEARL/nano_pearl/pearl_engine/pearl_model_runner.py`
  - 新增 `pearl_stream_generate`，逐步回傳新生成 token chunk。
  - 新增 `pearl_stream_steps`，一次跑多步 decode 後再回填，降低 IPC 開銷。
- `python/nano-PEARL/nano_pearl/pearl_engine/pearl_engine.py`
  - 新增 `stream_generate()` 介面，搭配 shared memory 回傳 chunk。
  - 新增 `stream_generate_steps()` 介面，支援單次跨多步的 stream 呼叫。
- `python/sglang/srt/managers/tp_worker.py`
  - worker thread 呼叫 `stream_generate_step()` 單步推進，逐步提供 token 給 Scheduler。
  - 目前改為呼叫 `stream_generate_steps()`，可用環境變數控制每次跨步數。

## 效能優化（與 sglang 排程整合）
- `python/sglang/srt/managers/tp_worker.py`
  - 常駐 worker thread 以 `stream_generate_step()` 單步推進，並在每步之間插入新請求，降低批次鎖死造成的等待。
  - streaming / non-streaming 共用同一套 token 佇列機制；non-stream decode 每步會一次回填當前 chunk，減少 scheduler 循環次數。
  - overlap 啟用時 `next_token_ids` 會建立在 GPU，避免 future map 跨裝置錯誤。
  - 新增 `NANO_PEARL_SGLANG_WAIT_TIMEOUT_S` 等待上限，避免卡死時無限阻塞。
  - 新增 `NANO_PEARL_SGLANG_PREFETCH_STEPS` 與 `NANO_PEARL_SGLANG_PREFETCH_FLUSH_STEPS`，平衡 IPC 次數與 token 回填時機。
 - `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
   - nano-pearl decode 支援一次回填多 token 時，額外分配 KV 索引並同步更新 seq_len 與 KV 統計，避免記憶體帳務錯亂。
 - `python/sglang/srt/managers/scheduler_runtime_checker_mixin.py`
   - 針對 nano-pearl 的 KV allocator 增加去重與回收修正，減少誤判與記憶體洩漏。

## 已修正的錯誤
- `world group is not initialized`
  - 原因：draft runner 初始化時觸發 `get_world_group()`，但分散式環境尚未完成。
  - 解法：調整 nano-pearl runner 初始化順序，先建立 target runner。

- `AttributeError: ModelRunner has no attribute attention_backend`
  - 原因：ModelRunner 實際屬性名稱為 `attn_backend`。
  - 解法：移除錯誤存取。
 - `indices should be either on cpu or on the same device`
   - 原因：CPU tensor 使用 GPU indices 取值。
   - 解法：依 tensor 裝置選擇對應 indices，避免跨裝置索引。
 - `token_to_kv_pool_allocator memory leak detected`
   - 原因：KV free/release 記帳不一致或缺漏。
   - 解法：在 runtime checker 補齊去重與回收流程，避免誤判與漏釋放。
 - `StopIteration` in CUDA graph selection
   - 原因：batch size 超出已捕獲的 graph size 範圍。
   - 解法：找不到合適 graph 時回退 eager，避免 crash。

## 目前限制（已知）
- 僅支援 `tp_size=1`、`pp_size=1`。
- 需要至少 2 張 GPU（draft + target 各一張，或符合 `draft_model_tp_size + tp_size`）。
- 不支援 `return_logprob` 與 prefill-only batches。
- Sampling 目前只支援：
  - `temperature`
  - `max_new_tokens`
  - `ignore_eos`
- `top_p` / `top_k` 會被忽略（會提示 warning）。
- nano-PEARL streaming 目前以單步推進，仍無法在單次步驟內插入新請求，但可在步與步之間加入。

## 行為與流程摘要
1. `--enable-nano-pearl` 會觸發 PEARLEngine 初始化。
2. Scheduler 進入 decode 後，TpModelWorker 將 req 送入 nano-pearl worker 佇列。
3. worker 以 `stream_generate_steps()` 跨步產生 token chunk，回填到 token 佇列。
4. Scheduler 從佇列取出 `next_token_ids`，回傳給後續處理；`stream=true` 逐步回傳 chunk。

## 後續可能需要處理的方向
- 支援 `tp_size > 1` 與 `pp_size > 1` 的 PEARLEngine 配置與分工。
- 讓 nano-PEARL 回傳 logprob 或其他 metadata（若有需求）。
- 對齊 sglang 原本的 streaming / overlap 調度行為（目前為同步式 token feed）。

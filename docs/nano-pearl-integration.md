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

## 主要修改與完成內容
### Server args / CLI
- `python/sglang/srt/server_args.py`
  - 新增 nano-pearl 初始化處理（draft path / revision 設定、忽略 speculative_algorithm）。
  - `--enable-nano-pearl` 改為 `store_true`。
  - 支援 `--speculative-algorithm nano_pearl` 作為等效開關。

### nano-PEARL API 補強
- `python/nano-PEARL/nano_pearl/pearl_engine/pearl_engine.py`
  - `add_request()` 回傳 `seq_id` 以便外部對應 req。
  - 新增 `generate_tokens()` 回傳 raw token ids，供 sglang 對接。

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
- `python/nano-PEARL/nano_pearl/pearl_engine/pearl_engine.py`
  - 新增 `stream_generate()` 介面，搭配 shared memory 回傳 chunk。
- `python/sglang/srt/managers/tp_worker.py`
  - 在 `stream=true` 時走 `stream_generate` 串流流程，逐步提供 token 給 Scheduler。

## 已修正的錯誤
- `world group is not initialized`
  - 原因：draft runner 初始化時觸發 `get_world_group()`，但分散式環境尚未完成。
  - 解法：調整 nano-pearl runner 初始化順序，先建立 target runner。

- `AttributeError: ModelRunner has no attribute attention_backend`
  - 原因：ModelRunner 實際屬性名稱為 `attn_backend`。
  - 解法：移除錯誤存取。

## 目前限制（已知）
- 僅支援 `tp_size=1`、`pp_size=1`。
- 需要至少 2 張 GPU（draft + target 各一張，或符合 `draft_model_tp_size + tp_size`）。
- 不支援 `return_logprob` 與 prefill-only batches。
- Sampling 目前只支援：
  - `temperature`
  - `max_new_tokens`
  - `ignore_eos`
- `top_p` / `top_k` 會被忽略（會提示 warning）。
- nano-PEARL streaming 目前是以整個 batch 為單位同步推進。

## 行為與流程摘要
1. `--enable-nano-pearl` 會觸發 PEARLEngine 初始化。
2. Scheduler 進入 decode 後，TpModelWorker 使用 PEARLEngine 產生 tokens。
3. 產生的 tokens 轉成 `next_token_ids`，回傳給 Scheduler 進行後續處理。
4. 若 `/v1/completions` 帶 `stream=true`，nano-PEARL 會改走 `stream_generate`，逐步產生 chunk 並回填 token。

## 後續可能需要處理的方向
- 支援 `tp_size > 1` 與 `pp_size > 1` 的 PEARLEngine 配置與分工。
- 讓 nano-PEARL 回傳 logprob 或其他 metadata（若有需求）。
- 對齊 sglang 原本的 streaming / overlap 調度行為（目前為同步式 token feed）。

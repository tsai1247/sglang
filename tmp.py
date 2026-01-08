import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

# ----------------設定區----------------
# 注意：Qwen3 目前可能尚未發布，此處以 Qwen2.5-32B 為例 (邏輯通用)
# 如果您確實有權限存取 Qwen3，請直接替換 MODEL_ID
# MODEL_ID = "Qwen/Qwen2.5-32B-Instruct" 
MODEL_ID = "/home/ubuntu/models/Qwen/Qwen3-0.6B" 

# 為了節省記憶體，建議使用 4-bit 量化載入 (需要安裝 bitsandbytes)
LOAD_IN_4BIT = False 
# 生成步數 (每次輸入預設生成 50 個 token)
NUM_STEPS = 10
# 資料來源
DATA_PATH = "./data.json"
# -------------------------------------

print(f"正在載入模型: {MODEL_ID} ... (這可能需要一點時間)")

# 1. 載入 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# 2. 載入模型
# 32B 模型很大，建議使用 device_map="auto" 自動分配 GPU/CPU
model_kwargs = {"device_map": "auto", "trust_remote_code": True}
if LOAD_IN_4BIT:
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model_kwargs["quantization_config"] = quantization_config
else:
    model_kwargs["torch_dtype"] = "auto"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)

def get_next_token_candidates(input_ids):
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(f"輸入 Token IDs: {input_ids}")
    print(f"輸入 Tokens: {input_tokens}")

    input_tensor = torch.tensor([input_ids], device=model.device)
    attention_mask = torch.ones_like(input_tensor)

    # 4. 前向傳播 (Forward Pass) 獲取 Logits
    # 我們不需要計算梯度，使用 torch.no_grad() 節省記憶體
    with torch.no_grad():
        outputs = model(input_ids=input_tensor, attention_mask=attention_mask)
        # 獲取最後一個 token 的 logits (預測下一個字)
        next_token_logits = outputs.logits[0, -1, :]

    # 5. 計算機率 (Softmax)
    probs = torch.softmax(next_token_logits, dim=-1)

    # 6. 取得前 50 個候選 (Top-K)
    top_k = 50
    top_probs, top_indices = torch.topk(probs, top_k)

    # 7. 整理數據並解碼
    candidates = []
    candidates_json = []
    for i in range(top_k):
        token_id = top_indices[i].item()
        probability = top_probs[i].item()
        token_str = tokenizer.decode([token_id])
        
        candidates.append({
            "Rank": i + 1,
            "Token ID": token_id,
            "Token String": f"'{token_str}'", # 加上引號以便觀察空格
            "Probability": f"{probability:.4%}"
        })
        candidates_json.append({
            "id": token_id,
            "value": token_str,
            "p": f"{probability:.4%}"
        })

    # 8. 決定最終選擇 (Greedy Strategy - 選擇機率最高的)
    # 注意：實際生成時若 temperature > 0，可能會隨機選取前幾名之一
    chosen_token_id = top_indices[0].item()
    chosen_token_str = tokenizer.decode([chosen_token_id])
    
    return candidates, candidates_json, chosen_token_id, chosen_token_str, input_tokens

# --- 執行程式 ---
if __name__ == "__main__":
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    for i, item in enumerate(all_data):
        for j, convo in enumerate(item.get("conversations", [])):
            if convo.get("from") != "human":
                continue
            my_prompt = convo.get("value", "")
            if not my_prompt:
                continue
            try:
                input_ids = tokenizer(my_prompt, return_tensors="pt")["input_ids"][0].tolist()
                with open("tmp.jsonl", "a", encoding="utf-8") as f:
                    for step in range(NUM_STEPS):
                        current_prompt = tokenizer.decode(input_ids)
                        print(f"\n=== Item {i} Convo {j} Step {step + 1}/{NUM_STEPS} ===")
                        print(f"輸入 Prompt: '{current_prompt}'")

                        candidates_list, candidates_json, output_token, output_value, input_tokens = get_next_token_candidates(input_ids)

                        # 顯示結果
                        df = pd.DataFrame(candidates_list)
                        print("\n=== 下一個 Token 的前 50 個候選 ===")
                        print(df.to_string(index=False))

                        print("\n" + "="*40)
                        print(f"模型決定的下一個 Token 是: '{output_value}'")
                        print("="*40)

                        record = {
                            "prompt": current_prompt,
                            "input_tokens": input_tokens,
                            "candidates": candidates_json,
                            "output_token": output_token,
                            "output_value": output_value,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                        input_ids.append(output_token)

            except Exception as e:
                print(f"發生錯誤: {e}")
                print("提示: 請確認您的 VRAM 足夠，或已安裝 bitsandbytes, accelerate 等套件。")

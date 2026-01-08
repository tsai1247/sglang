"""
使用 Cross-Encoder 微調分類器 - 讓 Input 與 Candidate 交互
環境: pip install sentence-transformers
"""
from sentence_transformers import CrossEncoder, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
import math

# --- 1. 初始化模型 ---
# cross-encoder/mmarco-mMiniLM-v2-L12-H384-v1 是一個強大的多語言重排序模型基底
# num_labels=1 表示輸出一個分數 (Regression)
model = CrossEncoder('/home/ubuntu/models/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', num_labels=1)

# --- 2. 準備數據 ---
# CrossEncoder 需要明確的 正樣本 (分數=1.0) 和 負樣本 (分數=0.0)
# 你需要構建數據： [Input, Correct_Class] -> 1.0, [Input, Wrong_Class] -> 0.0
train_examples = []

# 假設你有一筆數據: input="專利A", correct="半導體", wrongs=["面板", "生技"...]
# 為了效率，通常比例是 1個正樣本 : N個負樣本 (例如 1:4)
dataset_samples = [
    {"q": "專利摘要：一種高效能的FinFET...", "pos": "半導體製程", "negs": ["生物科技", "面板顯示", "化工製程"]},
    # ... 更多數據
]

for item in dataset_samples:
    # 正樣本
    train_examples.append(InputExample(texts=[item['q'], item['pos']], label=1.0))
    # 負樣本 (這就是讓模型學習 "在這些候選中，這個是錯的")
    for neg in item['negs']:
        train_examples.append(InputExample(texts=[item['q'], neg], label=0.0))

# DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

# --- 3. 訓練 ---
# 這裡不需要特殊的 RankingLoss，因為我們把它轉化成了二元分類/回歸問題
# 但模型學會的是：給定 (Input, Cand)，判斷它們的匹配程度
num_epochs = 3
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

model.fit(
    train_dataloader=train_dataloader,
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path='./cross_encoder_finetuned',
    show_progress_bar=True
)

# --- 4. 推理 (Inference) ---
# 當你拿到 100 個候選時：
def predict_best_candidate(input_text, candidates_list):
    # 準備 100 對 (input, cand)
    pairs = [[input_text, cand] for cand in candidates_list]
    
    # 讓模型一次評分所有對 (GPU 上可以 batch 處理，很快)
    scores = model.predict(pairs) # 返回 array([0.9, 0.1, 0.05 ...])
    
    # 找出最高分的 index
    best_idx = scores.argmax()
    return candidates_list[best_idx], scores[best_idx]

# 測試
input_text = "專利摘要：一種高效能的FinFET..."
candidates = ["生物科技", "半導體製程", "面板顯示"] # 實際會有100個
result, score = predict_best_candidate(input_text, candidates)
print(f"最佳匹配: {result} (Score: {score:.4f})")

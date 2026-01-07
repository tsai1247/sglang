import asyncio
import json
import time
from typing import List, Dict
import aiohttp

# 配置參數
PORT = 12470  # 修改為你的 port
CONCURRENCY = 8
JSON_FILE = "train-00000-of-00001_system-human-gpt.json"

async def send_completion_request(session: aiohttp.ClientSession, prompt: str, request_id: int) -> Dict:
    """發送單個 completion 請求並記錄指標"""
    url = f"http://localhost:{PORT}/v1/completions"
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": 256,
        "temperature": 0.7,
        "stream": False
    }
    
    metrics = {
        "request_id": request_id,
        "ttft": None,
        "total_time": None,
        "tokens_generated": 0,
        "itl_list": []
    }
    
    start_time = time.perf_counter()
    
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
            if response.status != 200:
                print(f"Request {request_id} failed with status {response.status}")
                return metrics
            
            result = await response.json()
            end_time = time.perf_counter()
            
            # 計算指標
            metrics["total_time"] = end_time - start_time
            metrics["ttft"] = metrics["total_time"]  # 非串流模式，TTFT 等於總時間
            
            # 從回應中提取 token 數量
            if "usage" in result:
                metrics["tokens_generated"] = result["usage"].get("completion_tokens", 0)
            
            # 計算平均 ITL（總時間除以 token 數）
            if metrics["tokens_generated"] > 1:
                avg_itl = metrics["total_time"] / metrics["tokens_generated"]
                metrics["itl_list"] = [avg_itl] * metrics["tokens_generated"]
            
    except Exception as e:
        print(f"Request {request_id} error: {e}")
    
    return metrics

async def run_benchmark():
    """執行並發測試"""
    # 讀取測試資料
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取 prompt（從 conversations 中取 human 的內容）
    prompts = []
    for one_prompt in data:
        convs = one_prompt["conversations"]
        for conv in convs:
            if conv.get("from") == "human":
                prompts.append(conv.get("value", ""))
    
    # 如果沒有足夠的 prompt，重複使用
    if len(prompts) < CONCURRENCY:
        prompts = (prompts * ((CONCURRENCY // len(prompts)) + 1))[:CONCURRENCY]
    else:
        prompts = prompts[:CONCURRENCY]
    
    print(f"開始測試：{CONCURRENCY} 個並發請求")
    print(f"目標 URL: http://localhost:{PORT}/v1/completions\n")
    
    # 建立 session 並發送請求
    connector = aiohttp.TCPConnector(limit=100)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            send_completion_request(session, prompt, i) 
            for i, prompt in enumerate(prompts)
        ]
        
        benchmark_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        benchmark_end = time.perf_counter()
    
    # 計算統計指標
    successful_requests = [r for r in results if r["total_time"] is not None]
    
    if not successful_requests:
        print("所有請求都失敗了")
        return
    
    total_tokens = sum(r["tokens_generated"] for r in successful_requests)
    total_time = benchmark_end - benchmark_start
    
    ttft_values = [r["ttft"] for r in successful_requests]
    avg_ttft = sum(ttft_values) / len(ttft_values)
    
    # 計算平均 ITL
    all_itls = []
    for r in successful_requests:
        all_itls.extend(r["itl_list"])
    avg_itl = sum(all_itls) / len(all_itls) if all_itls else 0
    
    # 輸出結果
    print("=" * 50)
    print("測試結果")
    print("=" * 50)
    print(f"並發數量: {CONCURRENCY}")
    print(f"成功請求: {len(successful_requests)}/{CONCURRENCY}")
    print(f"總耗時: {total_time:.2f} 秒")
    print(f"\n--- 吞吐量 (Throughput) ---")
    print(f"總 token 數: {total_tokens}")
    print(f"吞吐量: {total_tokens / total_time:.2f} tokens/sec")
    print(f"\n--- TTFT (Time To First Token) ---")
    print(f"平均 TTFT: {avg_ttft:.3f} 秒")
    print(f"最小 TTFT: {min(ttft_values):.3f} 秒")
    print(f"最大 TTFT: {max(ttft_values):.3f} 秒")
    print(f"\n--- ITL (Inter-Token Latency) ---")
    print(f"平均 ITL: {avg_itl * 1000:.2f} ms")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(run_benchmark())

import asyncio
import aiohttp
import json
import time
import sys

# Configuration
BASE_URL = "http://localhost:12470/v1/chat/completions"
MODEL_NAME = "test"  # As defined in start.sh
CONCURRENT_REQUESTS = 10
PROMPT = "Explain the theory of relativity in one sentence."

async def send_request(session, request_id):
    url = BASE_URL
    headers = {"Content-Type": "application/json"}
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 50,
        "temperature": 0.0,
        "stream": True
    }
    
    start_time = time.time()
    first_token_time = None
    full_response = ""
    error = None
    
    try:
        async with session.post(url, headers=headers, json=data, timeout=60) as response:
            if response.status != 200:
                text = await response.text()
                return {
                    "id": request_id,
                    "status": "error",
                    "code": response.status,
                    "error": text,
                    "duration": time.time() - start_time
                }
            
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    if first_token_time is None:
                        first_token_time = time.time()
                    
                    json_str = line[6:]
                    try:
                        chunk = json.loads(json_str)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content is None:
                                print(f"Request {request_id}: Received NULL content chunk. Raw: {chunk}")
                                error = "NULL content"
                            else:
                                full_response += content
                    except json.JSONDecodeError:
                        pass
    except asyncio.TimeoutError:
        error = "Timeout"
    except Exception as e:
        error = str(e)
        
    end_time = time.time()
    duration = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time else None
    
    return {
        "id": request_id,
        "status": "ok" if not error else "error",
        "error": error,
        "full_response": full_response,
        "duration": duration,
        "ttft": ttft
    }

async def main():
    print(f"Starting reproduction test with {CONCURRENT_REQUESTS} concurrent requests...")
    
    # Wait for server to be ready
    async with aiohttp.ClientSession() as check_session:
        for i in range(60):
            try:
                async with check_session.get("http://localhost:12470/health") as resp:
                    if resp.status == 200:
                        print("Server is ready.")
                        break
            except:
                pass
            print("Waiting for server...")
            await asyncio.sleep(2)
        else:
             print("Server not ready after 30s. Exiting.")
             # Continue anyway to see if we can trigger something, or maybe the health check endpoint is different.
    
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i) for i in range(CONCURRENT_REQUESTS)]
        results = await asyncio.gather(*tasks)
        
    print("\nResults:")
    failure_count = 0
    null_token_count = 0
    long_hang_count = 0 # Define your threshold, e.g., > 10s
    
    for res in results:
        print(f"Req {res['id']}: Status={res['status']}, Duration={res['duration']:.2f}s, TTFT={res['ttft'] if res['ttft'] else 'N/A'}")
        if res['status'] == 'error':
            failure_count += 1
            print(f"  Error: {res['error']}")
            if res['error'] == "NULL content":
                null_token_count += 1
        
        if res['duration'] > 10.0: # Arbitrary threshold for "hanging"
             long_hang_count += 1

    print(f"\nSummary:")
    print(f"Total Requests: {CONCURRENT_REQUESTS}")
    print(f"Failures: {failure_count}")
    print(f"Null Tokens: {null_token_count}")
    print(f"Hangs (>10s): {long_hang_count}")

if __name__ == "__main__":
    asyncio.run(main())

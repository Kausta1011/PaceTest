import requests,json

response = requests.post(
    "http://localhost:11434/api/generate",
    json = {"model": "qwen3:8b", "prompt": "Which model are you out of the 3 Qwen series? Out of the three Qwen-7B (7 billion parameters) Qwen-14B** (14 billion parameters) Qwen-32B (32 billion parameters), which one are you?" , "stream": False}
)
print(response.json()["response"])
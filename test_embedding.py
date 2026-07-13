import os
import requests
from dotenv import load_dotenv

# 加载密钥
load_dotenv()

# 向量网关配置
EMBED_BASE = "https://public.cnki.net/llm"
EMBED_V1 = f"{EMBED_BASE}/v1"
PRIVATE_API_KEY = os.getenv("PRIVATE_API_KEY")
EMBED_MODEL = "qwen3-embedding-8b"

def test_embedding(text_list):
    headers = {
        "Authorization": f"Bearer {PRIVATE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": EMBED_MODEL,
        "input": text_list
    }
    print(f"请求地址：{EMBED_V1}/embeddings")
    print(f"请求文本：{text_list}")
    resp = requests.post(f"{EMBED_V1}/embeddings", headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    print(f"✅ 向量接口调用成功！")
    print(f"返回向量数量：{len(data['data'])}")
    print(f"单条向量长度：{len(data['data'][0]['embedding'])}")
    return data

if __name__ == "__main__":
    # 测试文本，可自行修改
    test_texts = ["婚假多少天", "员工休假管理规定"]
    test_embedding(test_texts)

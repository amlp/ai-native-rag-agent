import json
import os
import requests
from pathlib import Path
import chromadb
from dotenv import load_dotenv

load_dotenv()

# ===================== 网关配置 =====================
# 向量网关（测试可用 /v1/embeddings）
EMBED_BASE = "https://public.cnki.net/llm/v1"
PRIVATE_API_KEY = os.getenv("PRIVATE_API_KEY")
EMBED_MODEL = "qwen3-embedding-8b"

# 对话网关：完全使用你指定的完整地址
CHAT_FULL_URL = "https://coder.cnki.net/KCoder-Claude/chat/completions"
KCODER_API_KEY = os.getenv("KCODER_API_KEY")
CHAT_MODEL = "KCoder-Claude"
# ====================================================

# 持久化路径
CHROMA_PATH = "./db"
MEMORY_PATH = "./memory/chat_history.json"
DOC_FOLDER = "./docs"

# 对话记忆持久化
class ChatMemory:
    def __init__(self):
        self.path = Path(MEMORY_PATH)
        self.path.parent.mkdir(exist_ok=True)
        if not self.path.exists():
            self.save_history([])
    def load_history(self):
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)
    def save_history(self, history):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    def add_msg(self, role, content):
        h = self.load_history()
        h.append({"role": role, "content": content})
        self.save_history(h)

# 向量RAG加载器
class EmbeddingDocLoader:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection("local_docs")
        self.doc_dir = Path(DOC_FOLDER)
        self.embed_key = PRIVATE_API_KEY
        self.embed_url = EMBED_BASE

    def get_embedding(self, text_list):
        headers = {
            "Authorization": f"Bearer {self.embed_key}",
            "Content-Type": "application/json"
        }
        payload = {"model": EMBED_MODEL, "input": text_list}
        resp = requests.post(f"{self.embed_url}/embeddings", headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]

    def split_text(self, text, chunk_size=300, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start+chunk_size])
            start += chunk_size - overlap
        return chunks

    def load_all_docs(self):
        self.doc_dir.mkdir(exist_ok=True)
        file_list = list(self.doc_dir.glob("*.md")) + list(self.doc_dir.glob("*.txt"))
        if not file_list:
            print("⚠️ docs文件夹无文档")
            return
        for file in file_list:
            try:
                txt = file.read_text(encoding="utf-8")
                chunks = self.split_text(txt)
                embeds = self.get_embedding(chunks)
                ids = [f"{file.name}_{i}" for i in range(len(chunks))]
                self.collection.add(embeddings=embeds, documents=chunks, ids=ids, metadatas=[{"source": file.name}] * len(chunks))
                print(f"✅ 文档向量化完成：{file.name}")
            except Exception as e:
                print(f"❌ {file.name} 向量调用失败：{str(e)}")
        print(f"\n文档加载完毕，向量持久化至 ./db")

    def search_relevant(self, query, top_k=3):
        q_emb = self.get_embedding([query])[0]
        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        return res["documents"][0], res["metadatas"][0]

# 对话请求函数：直接使用你提供的完整对话URL
def chat_request(context_text, user_question, chat_history):
    headers = {
        "Authorization": f"Bearer {KCODER_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"""严格依据下方参考文档回答，无相关内容直接说明，禁止编造信息：
【参考文档片段】
{context_text}
用户问题：{user_question}
"""
    messages = chat_history + [{"role": "user", "content": prompt}]
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": 0.6
    }
    # 直接请求你指定的地址，不再拼接/v1
    resp = requests.post(CHAT_FULL_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def main():
    try:
        if not PRIVATE_API_KEY or not KCODER_API_KEY:
            print("❌ .env缺少PRIVATE_API_KEY或KCODER_API_KEY")
            input("按回车关闭窗口...")
            return

        memory = ChatMemory()
        rag_loader = EmbeddingDocLoader()
        rag_loader.load_all_docs()

        print("\n===== 第16天完整双网关RAG =====")
        print(f"向量服务：{EMBED_BASE}/embeddings | {EMBED_MODEL}")
        print(f"对话服务：{CHAT_FULL_URL} | {CHAT_MODEL}")
        print("输入 quit 退出程序\n")

        while True:
            user_input = input("用户：").strip()
            if user_input.lower() == "quit":
                print("💾 向量库、对话记录已保存，程序正常退出")
                break
            if not user_input:
                print("请输入有效问题！")
                continue
            doc_list, meta_list = rag_loader.search_relevant(user_input)
            context = "\n\n".join([f"【文档来源：{meta_list[i]['source']}】\n{doc_list[i]}" for i in range(len(doc_list))])
            memory.add_msg("user", user_input)
            agent_answer = chat_request(context, user_input, memory.load_history())
            memory.add_msg("assistant", agent_answer)
            print(f"Agent：{agent_answer}\n")
    except Exception as err:
        print(f"\n运行异常：{str(err)}")
        import traceback
        traceback.print_exc()
    finally:
        input("\n执行完毕，按回车键关闭窗口...")

if __name__ == "__main__":
    main()

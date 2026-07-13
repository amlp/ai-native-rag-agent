import json
import os
import time
import logging
import requests
from pathlib import Path
import chromadb
from dotenv import load_dotenv

# 日志初始化
log_dir = Path("./logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./logs/rag_run.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# 网关配置
BASE_V1 = "https://public.cnki.net/llm/v1"
PRIVATE_API_KEY = os.getenv("PRIVATE_API_KEY")
EMBED_MODEL = "qwen3-embedding-8b"

CHAT_FULL_URL = "https://coder.cnki.net/KCoder-Claude/chat/completions"
KCODER_API_KEY = os.getenv("KCODER_API_KEY")
CHAT_MODEL = "KCoder-Claude"

# 路径常量
CHROMA_PATH = "./db"
MEMORY_PATH = "./memory/chat_history.json"
DOC_FOLDER = "./docs"
MAX_RETRY = 2

# 对话记忆
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
        hist = self.load_history()
        hist.append({"role": role, "content": content})
        self.save_history(hist)

# 向量工具
class EmbeddingTool:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection("local_docs")
        self.doc_dir = Path(DOC_FOLDER)
        self.api_key = PRIVATE_API_KEY

    def get_embedding(self, text_list):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": EMBED_MODEL, "input": text_list}
        retry = 0
        while retry <= MAX_RETRY:
            try:
                start = time.time()
                resp = requests.post(f"{BASE_V1}/embeddings", headers=headers, json=payload, timeout=90)
                resp.raise_for_status()
                cost = round(time.time() - start, 2)
                logger.info(f"向量接口调用成功，耗时{cost}s，文本数量：{len(text_list)}")
                return [item["embedding"] for item in resp.json()["data"]]
            except Exception as e:
                retry += 1
                logger.error(f"向量接口失败，第{retry}次重试：{str(e)}")
                time.sleep(1)
        raise Exception("向量接口多次调用失败")

    def split_text(self, text, chunk_size=300, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size - overlap
        return chunks

    def load_all_docs(self):
        self.doc_dir.mkdir(exist_ok=True)
        file_list = list(self.doc_dir.glob("*.md")) + list(self.doc_dir.glob("*.txt"))
        if not file_list:
            logger.warning("docs文件夹无文档")
            return
        for file in file_list:
            try:
                txt = file.read_text(encoding="utf-8")
                chunks = self.split_text(txt)
                embeds = self.get_embedding(chunks)
                ids = [f"{file.name}_{i}" for i in range(len(chunks))]
                self.collection.add(embeddings=embeds, documents=chunks, ids=ids, metadatas=[{"source": file.name}] * len(chunks))
                logger.info(f"文档向量化完成：{file.name}")
            except Exception as e:
                logger.error(f"{file.name} 向量化失败：{str(e)}")
        logger.info(f"全部文档加载完毕，共{len(file_list)}个")

    def vector_search(self, query, top_k=3):
        q_emb = self.get_embedding([query])[0]
        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        return res["documents"][0]

# 流式对话（增加空数据判断，修复list index out of range）
def stream_chat(context_text, user_question, chat_history):
    headers = {
        "Authorization": f"Bearer {KCODER_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    prompt = f"""严格依据下方参考文档回答，文档无相关内容直接说明，禁止编造信息：
【参考文档片段】
{context_text}
用户问题：{user_question}
"""
    messages = chat_history + [{"role": "user", "content": prompt}]
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": 0.6,
        "stream": True
    }
    retry = 0
    while retry <= MAX_RETRY:
        try:
            resp = requests.post(CHAT_FULL_URL, headers=headers, json=payload, timeout=120, stream=True)
            resp.raise_for_status()
            logger.info(f"流式对话接口连接成功，用户问题：{user_question}")
            full_content = ""
            print("Agent：", end="", flush=True)
            for line in resp.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    # 修复：判断choices存在且不为空，防止下标越界
                    if "choices" not in chunk or len(chunk["choices"]) == 0:
                        continue
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        print(delta, end="", flush=True)
                        full_content += delta
            print("\n")
            return full_content
        except Exception as e:
            retry += 1
            logger.error(f"流式对话失败，第{retry}次重试：{str(e)}")
            time.sleep(1)
    raise Exception("对话接口多次调用失败")

def main():
    try:
        if not PRIVATE_API_KEY or not KCODER_API_KEY:
            logger.error("缺少PRIVATE_API_KEY或KCODER_API_KEY")
            input("回车关闭...")
            return

        memory = ChatMemory()
        embed_tool = EmbeddingTool()
        embed_tool.load_all_docs()

        print("\n==================== 第18天 工程化RAG（无Rerank） ====================")
        print(f"向量接口：{BASE_V1}/embeddings | {EMBED_MODEL}")
        print(f"对话流式接口：{CHAT_FULL_URL} | {CHAT_MODEL}")
        print("日志路径：./logs/rag_run.log | 输入 quit 退出\n")

        while True:
            user_q = input("用户：").strip()
            if user_q.lower() == "quit":
                logger.info("用户退出程序")
                print("程序正常退出")
                break
            if not user_q:
                print("请输入有效问题！")
                continue
            logger.info(f"用户提问：{user_q}")

            raw_docs = embed_tool.vector_search(user_q, top_k=3)
            context_final = "\n\n".join(raw_docs)
            answer = stream_chat(context_final, user_q, memory.load_history())
            memory.add_msg("user", user_q)
            memory.add_msg("assistant", answer)
            logger.info(f"本轮回答完成，回复长度：{len(answer)}")

    except Exception as err:
        logger.critical(f"全局异常：{str(err)}")
        import traceback
        traceback.print_exc()
    finally:
        input("\n执行完毕，按回车键关闭窗口...")

if __name__ == "__main__":
    main()

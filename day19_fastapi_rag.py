import json
import os
import time
import logging
import requests
from pathlib import Path
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -------------------------- 日志模块 --------------------------
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

# 加载密钥
load_dotenv()

# ===================== 网关配置 =====================
BASE_V1 = "https://public.cnki.net/llm/v1"
PRIVATE_API_KEY = os.getenv("PRIVATE_API_KEY")
EMBED_MODEL = "qwen3-embedding-8b"

CHAT_FULL_URL = "https://coder.cnki.net/KCoder-Claude/chat/completions"
KCODER_API_KEY = os.getenv("KCODER_API_KEY")
CHAT_MODEL = "KCoder-Claude"
# ====================================================

# 路径常量
CHROMA_PATH = "./db"
MEMORY_PATH = "./memory/chat_history.json"
DOC_FOLDER = "./docs"
MAX_RETRY = 2

# FastAPI 实例
app = FastAPI(title="第19天 RAG Web服务接口", version="1.0")

# 请求体数据模型
class QueryRequest(BaseModel):
    question: str

# 1. 对话记忆模块
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

# 2. 向量检索工具
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
            return False
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
                return False
        logger.info(f"全部文档加载完毕，共{len(file_list)}个")
        return True

    def vector_search(self, query, top_k=3):
        q_emb = self.get_embedding([query])[0]
        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        return res["documents"][0]

# 全局实例
memory = ChatMemory()
embed_tool = EmbeddingTool()

# 3. 流式对话封装
def chat_full_response(context_text, user_question, chat_history):
    headers = {
        "Authorization": f"Bearer {KCODER_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
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
        "temperature": 0.6,
        "stream": False
    }
    retry = 0
    while retry <= MAX_RETRY:
        try:
            resp = requests.post(CHAT_FULL_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            logger.info(f"对话接口连接成功，用户问题：{user_question}")
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            retry += 1
            logger.error(f"对话失败，第{retry}次重试：{str(e)}")
            time.sleep(1)
    raise Exception("对话接口多次调用失败")

# ===================== API接口定义 =====================
# 问答接口
@app.post("/rag/chat", summary="RAG问答接口")
def rag_chat(req: QueryRequest):
    try:
        user_q = req.question.strip()
        if not user_q:
            raise HTTPException(status_code=400, detail="问题不能为空")
        logger.info(f"API提问：{user_q}")
        # 向量检索
        raw_docs = embed_tool.vector_search(user_q, top_k=3)
        context = "\n\n".join(raw_docs)
        # 获取回答
        ans = chat_full_response(context, user_q, memory.load_history())
        # 保存对话
        memory.add_msg("user", user_q)
        memory.add_msg("assistant", ans)
        return {
            "code": 200,
            "question": user_q,
            "reference_docs": raw_docs,
            "answer": ans
        }
    except Exception as e:
        logger.error(f"问答接口异常：{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 重载文档向量库接口
@app.post("/rag/reload_docs", summary="重新加载文档并向量化入库")
def reload_docs():
    try:
        ok = embed_tool.load_all_docs()
        if ok:
            return {"code":200, "msg":"文档重载向量化完成"}
        else:
            return {"code":500, "msg":"文档加载失败，请查看日志"}
    except Exception as e:
        logger.error(f"重载文档异常：{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 健康检查接口
@app.get("/health", summary="服务健康检测")
def health():
    return {"code":200, "msg":"RAG服务运行正常"}

# 启动服务
if __name__ == "__main__":
    # 启动时自动加载文档
    embed_tool.load_all_docs()
    import uvicorn
    uvicorn.run(app="day19_fastapi_rag:app", host="0.0.0.0", port=8000, reload=True)

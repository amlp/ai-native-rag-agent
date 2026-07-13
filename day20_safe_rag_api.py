import json
import os
import time
import logging
import requests
from pathlib import Path
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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

load_dotenv()

# ===================== 全局安全配置 =====================
SERVER_API_KEY = "rag-server-2026-secure-key-123456"
limiter = Limiter(key_func=get_remote_address)
MAX_QUESTION_LENGTH = 200

# 向量网关
BASE_V1 = "https://public.cnki.net/llm/v1"
PRIVATE_API_KEY = os.getenv("PRIVATE_API_KEY")
EMBED_MODEL = "qwen3-embedding-8b"

# 对话网关
CHAT_FULL_URL = "https://coder.cnki.net/KCoder-Claude/chat/completions"
KCODER_API_KEY = os.getenv("KCODER_API_KEY")
CHAT_MODEL = "KCoder-Claude"
# ====================================================

CHROMA_PATH = "./db"
MEMORY_ROOT = Path("./sessions")
MEMORY_ROOT.mkdir(exist_ok=True)
DOC_FOLDER = "./docs"
MAX_RETRY = 2

# FastAPI实例
app = FastAPI(title="第20天 安全加固RAG Web服务", version="1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 跨域中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求体模型
class QueryRequest(BaseModel):
    session_id: str
    question: str

    @field_validator("question")
    def check_question_len(cls, v):
        v = v.strip()
        if len(v) == 0:
            raise ValueError("问题不能为空")
        if len(v) > MAX_QUESTION_LENGTH:
            raise ValueError(f"问题长度不能超过{MAX_QUESTION_LENGTH}字符")
        return v

# 统一返回格式
class RespModel(BaseModel):
    code: int
    msg: str
    data: dict | None = None

# 会话记忆
class SessionMemory:
    def __init__(self, session_id: str):
        self.path = MEMORY_ROOT / f"{session_id}.json"
        self.path.parent.mkdir(exist_ok=True)
        if not self.path.exists():
            self.save([])

    def load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, history):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def add_msg(self, role, content):
        hist = self.load()
        hist.append({"role": role, "content": content})
        self.save(hist)

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
        logger.info(f"全部文档加载完毕，共{len(file_list)}个")
        return True

    def vector_search(self, query, top_k=3):
        q_emb = self.get_embedding([query])[0]
        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        return res["documents"][0]

# 对话接口
def chat_full_response(context_text, user_question, chat_history):
    headers = {
        "Authorization": f"Bearer {KCODER_API_KEY}",
        "Content-Type": "application/json"
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

# 鉴权函数（必须传入request对象）
def verify_api_key(req: Request):
    client_key = req.headers.get("X-API-Key")
    if client_key != SERVER_API_KEY:
        raise HTTPException(status_code=401, detail="非法访问，API Key校验失败")

# 全局异常捕获
@app.exception_handler(Exception)
def global_err_handler(request: Request, exc: Exception):
    logger.critical(f"全局异常：{str(exc)}", exc_info=True)
    return RespModel(code=500, msg=f"服务内部异常：{str(exc)}", data=None).model_dump()

@app.exception_handler(HTTPException)
def http_err_handler(request: Request, exc: HTTPException):
    return RespModel(code=exc.status_code, msg=exc.detail, data=None).model_dump()

# ---------------- 接口（全部补全request参数给limiter和鉴权） ----------------
@app.get("/health", response_model=RespModel, summary="健康检测")
@limiter.limit("10/minute")
def health_check(request: Request):
    verify_api_key(request)
    return RespModel(code=200, msg="RAG服务运行正常", data=None)

@app.post("/rag/reload_docs", response_model=RespModel, summary="重载文档向量库")
@limiter.limit("5/minute")
def reload_docs(request: Request):
    verify_api_key(request)
    ok = embed_tool.load_all_docs()
    if ok:
        return RespModel(code=200, msg="文档重载向量化完成", data=None)
    return RespModel(code=500, msg="文档加载失败，请查看日志", data=None)

@app.post("/rag/chat", response_model=RespModel, summary="RAG问答接口")
@limiter.limit("10/minute")
def rag_chat(request: Request, req: QueryRequest):
    verify_api_key(request)
    user_q = req.question
    sid = req.session_id
    logger.info(f"会话{sid} 用户提问：{user_q}")

    raw_docs = embed_tool.vector_search(user_q, top_k=3)
    context = "\n\n".join(raw_docs)
    # 加载会话记忆
    memory = SessionMemory(sid)
    hist = memory.load()
    answer = chat_full_response(context, user_q, hist)
    # 保存对话
    memory.add_msg("user", user_q)
    memory.add_msg("assistant", answer)
    logger.info(f"会话{sid} 回答完成，长度{len(answer)}")
    return RespModel(
        code=200,
        msg="问答成功",
        data={
            "session_id": sid,
            "question": user_q,
            "reference_docs": raw_docs,
            "answer": answer
        }
    )

# 全局向量工具实例
embed_tool = EmbeddingTool()

if __name__ == "__main__":
    logger.info("===== 第20天 安全加固RAG服务启动 =====")
    embed_tool.load_all_docs()
    import uvicorn
    uvicorn.run("day20_safe_rag_api:app", host="0.0.0.0", port=8000, reload=True)

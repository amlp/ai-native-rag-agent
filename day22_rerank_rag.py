import json
import os
import time
import logging
import requests
from pathlib import Path
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, BackgroundTasks
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

# ===================== 全局安全&业务配置 =====================
SERVER_API_KEY = "rag-server-2026-secure-key-123456"
limiter = Limiter(key_func=get_remote_address)
MAX_QUESTION_LENGTH = 200
# 文件上传配置
UPLOAD_DIR = Path("./upload_docs")
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOW_EXT = {".txt", ".md"}
MAX_FILE_SIZE = 5 * 1024 * 1024
task_cache = {}
# 分块配置（保留重叠滑动分块优化，这个不需要模型）
CHUNK_SIZE = 350
CHUNK_OVERLAP = 80
# 向量网关
BASE_V1 = "https://public.cnki.net/llm/v1"
PRIVATE_API_KEY = os.getenv("PRIVATE_API_KEY")
EMBED_MODEL = "qwen3-embedding-8b"
# 对话网关
CHAT_FULL_URL = "https://coder.cnki.net/KCoder-Claude/chat/completions"
KCODER_API_KEY = os.getenv("KCODER_API_KEY")
CHAT_MODEL = "KCoder-Claude"
# 持久化路径
CHROMA_PATH = "./db"
MEMORY_ROOT = Path("./sessions")
MEMORY_ROOT.mkdir(exist_ok=True)
DOC_FOLDER = "./docs"
MAX_RETRY = 2
# ====================================================

app = FastAPI(title="第22天 RAG 检索优化服务（无本地重排模型）", version="1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求参数模型
class QueryRequest(BaseModel):
    session_id: str
    question: str
    @field_validator("question")
    def check_q_len(cls, v):
        v = v.strip()
        if len(v) == 0:
            raise ValueError("问题不能为空")
        if len(v) > MAX_QUESTION_LENGTH:
            raise ValueError(f"问题不能超过{MAX_QUESTION_LENGTH}字符")
        return v

# 统一返回体
class RespModel(BaseModel):
    code: int
    msg: str
    data: dict | None = None

# 多用户会话隔离
class SessionMemory:
    def __init__(self, session_id: str):
        self.file = MEMORY_ROOT / f"{session_id}.json"
        if not self.file.exists():
            self.save([])
    def load(self):
        with open(self.file, "r", encoding="utf-8") as f:
            return json.load(f)
    def save(self, hist):
        with open(self.file, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
    def add_msg(self, role, content):
        h = self.load()
        h.append({"role": role, "content": content})
        self.save(h)

# 向量&检索工具（移除本地rerank，仅向量召回）
class EmbeddingRetrievalTool:
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
                logger.info(f"向量调用成功，耗时{cost}s")
                return [i["embedding"] for i in resp.json()["data"]]
            except Exception as e:
                retry += 1
                logger.error(f"向量失败 重试{retry}: {str(e)}")
                time.sleep(1)
        raise Exception("向量接口多次失败")

    # 优化滑动窗口分块，带重叠（无模型依赖，保留）
    def split_text_slide(self, text):
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + CHUNK_SIZE, text_len)
            chunk = text[start:end]
            chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    # 加载本地文档入库
    def load_all_local_docs(self):
        self.doc_dir.mkdir(exist_ok=True)
        file_list = list(self.doc_dir.glob("*.txt")) + list(self.doc_dir.glob("*.md"))
        if not file_list:
            logger.warning("docs无文档")
            return False
        for file in file_list:
            try:
                txt = file.read_text(encoding="utf-8")
                chunks = self.split_text_slide(txt)
                embeds = self.get_embedding(chunks)
                ids = [f"{file.name}_{i}" for i in range(len(chunks))]
                self.collection.add(embeddings=embeds, documents=chunks, ids=ids, metadatas=[{"source": file.name}]*len(chunks))
                logger.info(f"文档入库：{file.name}")
            except Exception as e:
                logger.error(f"{file.name} 入库失败：{str(e)}")
        logger.info(f"本地文档加载完成，共{len(file_list)}个")
        return True

    # 仅向量粗召回，无本地重排模型
    def search_rerank_docs(self, query, top_k=3):
        q_emb = self.get_embedding([query])[0]
        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        raw_docs = res["documents"][0]
        return raw_docs

embed_tool = EmbeddingRetrievalTool()

# 后台异步处理上传文件
def background_parse_upload(task_id: str, file_path: Path, coll):
    task_cache[task_id] = {"status": "processing", "msg": "解析、分块、向量化中"}
    try:
        txt = file_path.read_text(encoding="utf-8")
        chunks = embed_tool.split_text_slide(txt)
        embeds = embed_tool.get_embedding(chunks)
        ids = [f"{file_path.name}_{idx}" for idx in range(len(chunks))]
        coll.add(embeddings=embeds, documents=chunks, ids=ids, metadatas=[{"source": file_path.name}]*len(chunks))
        task_cache[task_id] = {"status": "success", "msg": f"{file_path.name} 入库完成"}
        logger.info(f"异步任务{task_id}处理成功")
    except Exception as err:
        task_cache[task_id] = {"status": "fail", "msg": str(err)}
        logger.error(f"异步任务{task_id}失败：{str(err)}")

# LLM对话请求
def chat_request(context_text, user_q, chat_history):
    headers = {"Authorization": f"Bearer {KCODER_API_KEY}", "Content-Type": "application/json"}
    prompt = f"""严格根据下面参考文档回答问题，无相关内容直接说明，禁止编造：
【参考检索内容】
{context_text}
用户问题：{user_q}
"""
    messages = chat_history + [{"role": "user", "content": prompt}]
    payload = {"model": CHAT_MODEL, "messages": messages, "temperature": 0.6, "stream": False}
    retry = 0
    while retry <= MAX_RETRY:
        try:
            resp = requests.post(CHAT_FULL_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            retry += 1
            logger.error(f"对话重试{retry}: {str(e)}")
            time.sleep(1)
    raise Exception("对话接口多次调用失败")

# 鉴权校验
def verify_api_key(request: Request):
    key = request.headers.get("X-API-Key")
    if key != SERVER_API_KEY:
        raise HTTPException(status_code=401, detail="非法访问，API密钥错误")

# 全局统一异常捕获
@app.exception_handler(Exception)
def global_err_handler(request: Request, exc: Exception):
    logger.critical(f"全局异常：{str(exc)}", exc_info=True)
    return RespModel(code=500, msg=f"服务异常：{str(exc)}", data=None).model_dump(mode="json")

@app.exception_handler(HTTPException)
def http_err_handler(request: Request, exc: HTTPException):
    return RespModel(code=exc.status_code, msg=exc.detail, data=None).model_dump(mode="json")

# ===================== 接口路由 =====================
@app.get("/health", response_model=RespModel, summary="健康检测")
@limiter.limit("10/minute")
def health(request: Request):
    verify_api_key(request)
    return RespModel(code=200, msg="RAG服务正常运行", data=None)

@app.post("/rag/reload_docs", response_model=RespModel, summary="重载本地文档库")
@limiter.limit("5/minute")
def reload_docs(request: Request):
    verify_api_key(request)
    ok = embed_tool.load_all_local_docs()
    return RespModel(code=200 if ok else 500, msg="重载完成" if ok else "加载失败", data=None)

@app.post("/rag/upload_file", response_model=RespModel, summary="上传文档异步入库")
@limiter.limit("3/minute")
def upload_file(request: Request, bg_task: BackgroundTasks, file: UploadFile):
    verify_api_key(request)
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOW_EXT:
        raise HTTPException(400, f"仅支持 {ALLOW_EXT} 文件")
    content = file.file.read()
    if len(content) == 0:
        raise HTTPException(400, "文件不能为空")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "文件最大5MB")
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(content)
    task_id = f"task_{int(time.time()*1000)}"
    bg_task.add_task(background_parse_upload, task_id, save_path, embed_tool.collection)
    logger.info(f"文件{file.filename}上传，任务ID：{task_id}")
    return RespModel(code=200, msg="文件接收，后台处理中", data={"task_id": task_id})

@app.get("/rag/task/{task_id}", response_model=RespModel, summary="查询上传任务状态")
@limiter.limit("20/minute")
def get_task(request: Request, task_id: str):
    verify_api_key(request)
    if task_id not in task_cache:
        return RespModel(code=404, msg="任务不存在", data=None)
    return RespModel(code=200, msg="查询成功", data=task_cache[task_id])

# RAG问答（仅向量召回，无本地重排模型）
@app.post("/rag/chat", response_model=RespModel, summary="RAG问答（仅向量召回，无本地重排）")
@limiter.limit("10/minute")
def rag_chat(request: Request, req: QueryRequest):
    verify_api_key(request)
    sid = req.session_id
    question = req.question
    logger.info(f"会话{sid}提问：{question}")
    ref_docs = embed_tool.search_rerank_docs(question, top_k=3)
    context = "\n\n=====\n\n".join(ref_docs)
    memory = SessionMemory(sid)
    history = memory.load()
    answer = chat_request(context, question, history)
    memory.add_msg("user", question)
    memory.add_msg("assistant", answer)
    return RespModel(
        code=200,
        msg="问答成功（滑动分块优化，无本地重排模型）",
        data={
            "session_id": sid,
            "question": question,
            "reference_docs": ref_docs,
            "answer": answer
        }
    )

# 启动入口
if __name__ == "__main__":
    logger.info("===== 第22天 RAG 服务启动（无本地重排模型） =====")
    embed_tool.load_all_local_docs()
    import uvicorn
    uvicorn.run("day22_rerank_rag:app", host="0.0.0.0", port=8000, reload=True)

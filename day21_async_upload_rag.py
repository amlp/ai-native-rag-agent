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

# 日志
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

# 全局配置
SERVER_API_KEY = "rag-server-2026-secure-key-123456"
limiter = Limiter(key_func=get_remote_address)
MAX_QUESTION_LENGTH = 200
UPLOAD_DIR = Path("./upload_docs")
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOW_EXT = {".txt", ".md"}
MAX_FILE_SIZE = 5 * 1024 * 1024
task_cache = {}

# 向量、对话地址
BASE_V1 = "https://public.cnki.net/llm/v1"
PRIVATE_API_KEY = os.getenv("PRIVATE_API_KEY")
EMBED_MODEL = "qwen3-embedding-8b"
CHAT_FULL_URL = "https://coder.cnki.net/KCoder-Claude/chat/completions"
KCODER_API_KEY = os.getenv("KCODER_API_KEY")
CHAT_MODEL = "KCoder-Claude"

CHROMA_PATH = "./db"
MEMORY_ROOT = Path("./sessions")
MEMORY_ROOT.mkdir(exist_ok=True)
DOC_FOLDER = "./docs"
MAX_RETRY = 2

app = FastAPI(title="第21天 RAG 异步文件上传", version="1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class QueryRequest(BaseModel):
    session_id: str
    question: str
    @field_validator("question")
    def check_q(cls, v):
        v = v.strip()
        if len(v) == 0:
            raise ValueError("问题不能为空")
        if len(v) > MAX_QUESTION_LENGTH:
            raise ValueError(f"问题不能超过{MAX_QUESTION_LENGTH}字符")
        return v

# 统一返回
class RespModel(BaseModel):
    code: int
    msg: str
    data: dict | None = None

# 会话记忆
class SessionMemory:
    def __init__(self, session_id: str):
        self.path = MEMORY_ROOT / f"{session_id}.json"
        if not self.path.exists():
            self.save([])
    def load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)
    def save(self, hist):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
    def add_msg(self, role, content):
        h = self.load()
        h.append({"role": role, "content": content})
        self.save(h)

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
                logger.info(f"向量调用成功，耗时{cost}s")
                return [i["embedding"] for i in resp.json()["data"]]
            except Exception as e:
                retry += 1
                logger.error(f"向量失败 重试{retry}: {str(e)}")
                time.sleep(1)
        raise Exception("向量接口多次失败")

    def split_text(self, text, chunk_size=300, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start+chunk_size])
            start += chunk_size - overlap
        return chunks

    def load_all_docs(self):
        self.doc_dir.mkdir(exist_ok=True)
        files = list(self.doc_dir.glob("*.md")) + list(self.doc_dir.glob("*.txt"))
        if not files:
            logger.warning("docs无文档")
            return False
        for f in files:
            try:
                txt = f.read_text(encoding="utf-8")
                chunks = self.split_text(txt)
                embeds = self.get_embedding(chunks)
                ids = [f"{f.name}_{i}" for i in range(len(chunks))]
                self.collection.add(embeddings=embeds, documents=chunks, ids=ids, metadatas=[{"source":f.name}]*len(chunks))
                logger.info(f"文档入库: {f.name}")
            except Exception as e:
                logger.error(f"{f.name} 失败: {str(e)}")
        logger.info(f"加载完成，共{len(files)}个文档")
        return True

    def vector_search(self, query, top_k=3):
        emb = self.get_embedding([query])[0]
        res = self.collection.query(query_embeddings=[emb], n_results=top_k)
        return res["documents"][0]

embed_tool = EmbeddingTool()

# 后台处理文件（传入collection，无self报错）
def background_process_file(task_id: str, file_path: Path, coll):
    task_cache[task_id] = {"status": "processing", "msg": "正在解析向量化"}
    try:
        txt = file_path.read_text(encoding="utf-8")
        chunks = embed_tool.split_text(txt)
        embeds = embed_tool.get_embedding(chunks)
        ids = [f"{file_path.name}_{i}" for i in range(len(chunks))]
        coll.add(embeddings=embeds, documents=chunks, ids=ids, metadatas=[{"source":file_path.name}]*len(chunks))
        task_cache[task_id] = {"status": "success", "msg": f"{file_path.name} 入库完成"}
        logger.info(f"任务{task_id}成功")
    except Exception as e:
        task_cache[task_id] = {"status": "fail", "msg": str(e)}
        logger.error(f"任务{task_id}失败: {str(e)}")

# 对话请求
def chat_query(context, q, hist):
    headers = {"Authorization": f"Bearer {KCODER_API_KEY}", "Content-Type": "application/json"}
    prompt = f"""严格依据文档回答，无相关内容直接说明，禁止编造：
【参考文档】
{context}
问题：{q}
"""
    messages = hist + [{"role":"user", "content":prompt}]
    payload = {"model": CHAT_MODEL, "messages": messages, "temperature":0.6, "stream":False}
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
    raise Exception("对话接口多次失败")

# 鉴权
def verify_key(req: Request):
    key = req.headers.get("X-API-Key")
    if key != SERVER_API_KEY:
        raise HTTPException(status_code=401, detail="API Key非法，拒绝访问")

# 全局异常统一返回JSON（修复纯文本500）
@app.exception_handler(Exception)
def global_err_handler(request: Request, exc: Exception):
    logger.critical(f"全局异常: {str(exc)}", exc_info=True)
    return RespModel(code=500, msg=f"服务异常: {str(exc)}", data=None).model_dump(mode="json")

@app.exception_handler(HTTPException)
def http_err_handler(request: Request, exc: HTTPException):
    return RespModel(code=exc.status_code, msg=exc.detail, data=None).model_dump(mode="json")

# 健康检测
@app.get("/health", response_model=RespModel)
@limiter.limit("10/minute")
def health(request: Request):
    verify_key(request)
    return RespModel(code=200, msg="服务正常", data=None)

# 重载文档
@app.post("/rag/reload_docs", response_model=RespModel)
@limiter.limit("5/minute")
def reload_docs(request: Request):
    verify_key(request)
    ok = embed_tool.load_all_docs()
    return RespModel(code=200 if ok else 500, msg="重载完成" if ok else "加载失败", data=None)

# 问答接口
@app.post("/rag/chat", response_model=RespModel)
@limiter.limit("10/minute")
def rag_chat(request: Request, req: QueryRequest):
    verify_key(request)
    q = req.question
    sid = req.session_id
    logger.info(f"会话{sid}提问: {q}")
    ref_docs = embed_tool.vector_search(q, 3)
    ctx = "\n\n".join(ref_docs)
    memory = SessionMemory(sid)
    ans = chat_query(ctx, q, memory.load())
    memory.add_msg("user", q)
    memory.add_msg("assistant", ans)
    return RespModel(code=200, msg="问答成功", data={"question":q, "reference":ref_docs, "answer":ans})

# 文件上传接口
@app.post("/rag/upload_file", response_model=RespModel)
@limiter.limit("3/minute")
def upload_file(request: Request, bg_task: BackgroundTasks, file: UploadFile):
    verify_key(request)
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOW_EXT:
        raise HTTPException(status_code=400, detail=f"仅支持{ALLOW_EXT}")
    content = file.file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="文件最大5MB")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="文件不能为空")
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(content)
    task_id = f"task_{int(time.time()*1000)}"
    bg_task.add_task(background_process_file, task_id, save_path, embed_tool.collection)
    logger.info(f"文件{file.filename}上传成功，任务{task_id}")
    return RespModel(code=200, msg="文件已接收，后台处理中", data={"task_id": task_id})

# 查询任务状态
@app.get("/rag/task/{task_id}", response_model=RespModel)
@limiter.limit("20/minute")
def get_task(request: Request, task_id: str):
    verify_key(request)
    if task_id not in task_cache:
        return RespModel(code=404, msg="任务不存在", data=None)
    return RespModel(code=200, msg="查询成功", data=task_cache[task_id])

# 启动入口（文件名匹配，无热加载导入报错）
if __name__ == "__main__":
    logger.info("===== 第21天 RAG异步文件上传服务启动 =====")
    embed_tool.load_all_docs()
    import uvicorn
    uvicorn.run("day21_async_upload_rag:app", host="0.0.0.0", port=8000, reload=True)

import json
import os
import time
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
import chromadb
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from PyPDF2 import PdfReader

# -------------------------- 日志模块 --------------------------
log_dir = Path("./logs")
log_dir.mkdir(parents=True, exist_ok=True)
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

# ===================== 全局配置 =====================
SERVER_API_KEY = "rag-server-2026-secure-key-123456"
limiter = Limiter(key_func=get_remote_address)
MAX_QUESTION_LENGTH = 200
# 文件上传
UPLOAD_DIR = Path("./upload_docs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOW_EXT = {".txt", ".md", ".pdf"}
MAX_FILE_SIZE = 5 * 1024 * 1024
task_cache = {}
# 分块
CHUNK_SIZE = 350
CHUNK_OVERLAP = 80
# LLM上下文最大字符限制
MAX_CONTEXT_LEN = 1800
# 云端API（彻底移除Rerank地址）
BASE_V1 = "https://public.cnki.net/llm/v1"
PRIVATE_API_KEY = os.getenv("PRIVATE_API_KEY")
EMBED_MODEL = "qwen3-embedding-8b"
CHAT_FULL_URL = "https://coder.cnki.net/KCoder-Claude/chat/completions"
KCODER_API_KEY = os.getenv("KCODER_API_KEY")
CHAT_MODEL = "KCoder-Claude"
# 向量库分片目录
CHROMA_ROOT = "./db_v3"
MEMORY_ROOT = Path("./sessions")
MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
DOC_FOLDER = Path("./docs")
DOC_FOLDER.mkdir(parents=True, exist_ok=True)
MAX_RETRY = 2
# 调用日志sqlite数据库
LOG_DB_PATH = "./api_call_log.db"
# ====================================================

# 初始化日志SQLite库
def init_log_db():
    conn = sqlite3.connect(LOG_DB_PATH)
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS api_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_time TEXT,
        api_path TEXT,
        session_id TEXT,
        question TEXT,
        source_file TEXT,
        cost_ms INTEGER,
        status_code INTEGER,
        msg TEXT
    )
    ''')
    conn.commit()
    conn.close()

# 写入调用日志
def write_api_log(api_path: str, session_id: str, question: str, source_file: str, cost_ms: int, status_code: int, msg: str):
    conn = sqlite3.connect(LOG_DB_PATH)
    cur = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(
        "INSERT INTO api_log(request_time,api_path,session_id,question,source_file,cost_ms,status_code,msg) VALUES (?,?,?,?,?,?,?,?)",
        (now, api_path, session_id, question, source_file, cost_ms, status_code, msg)
    )
    conn.commit()
    conn.close()

init_log_db()

app = FastAPI(title="第24天 RAG 纯净修复版（无Rerank、无对话压缩）", version="1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
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
    filter_source: str | None = None
    @field_validator("question")
    def check_q_len(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("问题不能为空")
        if len(v) > MAX_QUESTION_LENGTH:
            raise ValueError(f"问题长度不能超过{MAX_QUESTION_LENGTH}字符")
        return v

# 统一返回体
class RespModel(BaseModel):
    code: int
    msg: str
    data: dict | None = None

# 会话记忆（彻底关闭对话压缩）
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
    # 压缩函数直接返回原始对话，不调用LLM
    def compress_history(self):
        return self.load()

# 文档解析工具（PDF/TXT/MD）
def parse_file_to_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            page_txt = page.extract_text()
            if page_txt:
                full_text += page_txt + "\n"
        return full_text
    elif suffix in (".txt", ".md"):
        return file_path.read_text(encoding="utf-8")
    else:
        raise Exception(f"不支持的文件格式 {suffix}")

# 向量检索工具（完全移除Rerank相关代码）
class EmbeddingRetrievalTool:
    def __init__(self, slice_name: str = "default"):
        self.slice_path = Path(CHROMA_ROOT) / slice_name
        self.slice_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.slice_path))
        self.collection = self.client.get_or_create_collection("local_docs")
        self.doc_dir = DOC_FOLDER
        self.api_key = PRIVATE_API_KEY
        self.slice_name = slice_name

    def get_embedding(self, text_list):
        headers = {"Authorization": f"Bearer {PRIVATE_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": EMBED_MODEL, "input": text_list}
        retry = 0
        while retry <= MAX_RETRY:
            try:
                start = time.time()
                resp = requests.post(f"{BASE_V1}/embeddings", headers=headers, json=payload, timeout=90)
                resp.raise_for_status()
                cost = round(time.time() - start, 2)
                logger.info(f"向量接口调用成功，耗时{cost}s")
                return [i["embedding"] for i in resp.json()["data"]]
            except Exception as e:
                retry += 1
                logger.error(f"向量接口失败，重试{retry}: {str(e)}")
                time.sleep(1)
        raise Exception("向量接口多次调用失败")

    # 滑动重叠分块
    def split_slide_chunk(self, text):
        chunks = []
        start = 0
        total = len(text)
        while start < total:
            end = min(start + CHUNK_SIZE, total)
            chunk = text[start:end]
            chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    # 带元数据过滤检索，无Rerank
    def search_with_filter(self, query: str, filter_source: str | None = None, top_k=6):
        q_emb = self.get_embedding([query])[0]
        where_filter = {}
        if filter_source:
            where_filter = {"source": filter_source}
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where=where_filter if where_filter else None
        )
        raw_docs = res["documents"][0]
        if not raw_docs:
            return ["暂无相关文档内容"]
        # 直接返回前3条，不调用Rerank接口
        return raw_docs[:3]

    # 加载本地全部文档入库
    def load_all_local(self):
        file_list = list(self.doc_dir.glob("*.txt")) + list(self.doc_dir.glob("*.md")) + list(self.doc_dir.glob("*.pdf"))
        if not file_list:
            logger.warning("本地docs文件夹无文档")
            return False
        for f in file_list:
            try:
                txt = parse_file_to_text(f)
                chunks = self.split_slide_chunk(txt)
                embeds = self.get_embedding(chunks)
                ids = [f"{f.name}_{idx}" for idx in range(len(chunks))]
                meta = [{"source": f.name}] * len(chunks)
                self.collection.add(embeddings=embeds, documents=chunks, ids=ids, metadatas=meta)
                logger.info(f"分片{self.slice_name} 文档入库完成：{f.name}")
            except Exception as e:
                logger.error(f"{f.name} 入库失败：{str(e)}")
        logger.info(f"分片{self.slice_name} 本地文档加载完毕，共{len(file_list)}个")
        return True

    # 按源文档名批量删除向量
    def delete_by_source(self, source_name: str):
        all_data = self.collection.get()
        del_ids = []
        for idx, meta in enumerate(all_data["metadatas"]):
            if meta and meta.get("source") == source_name:
                del_ids.append(all_data["ids"][idx])
        if del_ids:
            self.collection.delete(ids=del_ids)
        logger.info(f"分片{self.slice_name} 已删除文档{source_name} 全部向量片段")

    # 清空当前分片向量库
    def clear_slice_vector(self):
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)
        logger.info(f"分片{self.slice_name} 向量库全部清空")

embed_tool = EmbeddingRetrievalTool(slice_name="default")

# 后台任务：解析文件+分块+入库
def background_parse_task(task_id: str, file_path: Path, coll):
    task_cache[task_id] = {"status": "processing", "msg": "解析文档、分块、向量化中"}
    try:
        full_text = parse_file_to_text(file_path)
        chunks = embed_tool.split_slide_chunk(full_text)
        embeds = embed_tool.get_embedding(chunks)
        ids = [f"{file_path.name}_{idx}" for idx in range(len(chunks))]
        meta = [{"source": file_path.name}] * len(chunks)
        coll.add(embeddings=embeds, documents=chunks, ids=ids, metadatas=meta)
        task_cache[task_id] = {"status": "success", "msg": f"{file_path.name} 解析入库成功"}
        logger.info(f"异步任务{task_id}处理完成")
    except Exception as err:
        task_cache[task_id] = {"status": "fail", "msg": str(err)}
        logger.error(f"异步任务{task_id}失败：{str(err)}")

# LLM 对话调用
def llm_chat_call(context_text, user_q, chat_history):
    if len(context_text) > MAX_CONTEXT_LEN:
        context_text = context_text[:MAX_CONTEXT_LEN]
    prompt = f"""严格根据下面参考文档回答问题，无相关内容直接说明，禁止编造信息。
【参考检索内容】
{context_text}
用户问题：{user_q}
"""
    messages = chat_history + [{"role": "user", "content": prompt}]
    headers = {"Authorization": f"Bearer {KCODER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": CHAT_MODEL, "messages": messages, "temperature": 0.6, "stream": False}
    retry = 0
    while retry <= MAX_RETRY:
        try:
            resp = requests.post(CHAT_FULL_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            retry += 1
            logger.error(f"对话接口重试{retry}: {str(e)}")
            time.sleep(1)
    raise Exception("对话接口多次调用失败")

# 鉴权校验
def verify_api_key(request: Request):
    key = request.headers.get("X-API-Key")
    if key != SERVER_API_KEY:
        raise HTTPException(status_code=401, detail="非法访问，API密钥错误")

# 【修复】全局异常处理，正确返回Pydantic模型
@app.exception_handler(Exception)
def global_err_handler(request: Request, exc: Exception):
    logger.critical(f"全局异常：{str(exc)}", exc_info=True)
    return RespModel(code=500, msg=f"服务异常：{str(exc)}")

@app.exception_handler(HTTPException)
def http_err_handler(request: Request, exc: HTTPException):
    return RespModel(code=exc.status_code, msg=exc.detail)

# ===================== 接口路由 =====================
@app.get("/health", response_model=RespModel, summary="健康检测")
@limiter.limit("10/minute")
def health(request: Request):
    start_ms = int(time.time() * 1000)
    verify_api_key(request)
    cost = int(time.time() * 1000) - start_ms
    write_api_log("/health", "", "", "", cost, 200, "服务正常")
    return RespModel(code=200, msg="RAG服务正常运行")

@app.post("/rag/reload_docs", response_model=RespModel, summary="重载本地全部文档入库默认分片")
@limiter.limit("5/minute")
def reload_docs(request: Request):
    start_ms = int(time.time() * 1000)
    verify_api_key(request)
    ok = embed_tool.load_all_local()
    cost = int(time.time() * 1000) - start_ms
    write_api_log("/rag/reload_docs", "", "", "", cost, 200 if ok else 500, "重载完成")
    return RespModel(code=200 if ok else 500, msg="本地文档重载完成" if ok else "无文档可加载")

# 多格式文件上传异步入库
@app.post("/rag/upload_file", response_model=RespModel, summary="PDF/TXT/MD文件上传，异步入库默认分片")
@limiter.limit("3/minute")
def upload_file(request: Request, bg_task: BackgroundTasks, file: UploadFile):
    start_ms = int(time.time() * 1000)
    verify_api_key(request)
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOW_EXT:
        raise HTTPException(400, f"仅支持 {ALLOW_EXT} 文件")
    content = file.file.read()
    if len(content) == 0:
        raise HTTPException(400, "文件不能为空")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "文件最大限制5MB")
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(content)
    task_id = f"task_{int(time.time()*1000)}"
    bg_task.add_task(background_parse_task, task_id, save_path, embed_tool.collection)
    cost = int(time.time() * 1000) - start_ms
    write_api_log("/rag/upload_file", "", "", file.filename, cost, 200, "文件接收，后台解析中")
    logger.info(f"文件{file.filename}已上传，任务ID：{task_id}")
    return RespModel(code=200, msg="文件接收成功，后台解析中", data={"task_id": task_id})

@app.get("/rag/task/{task_id}", response_model=RespModel, summary="查询上传异步任务状态")
@limiter.limit("20/minute")
def get_task_status(request: Request, task_id: str):
    start_ms = int(time.time() * 1000)
    verify_api_key(request)
    if task_id not in task_cache:
        write_api_log("/rag/task", "", "", "", int(time.time()*1000)-start_ms, 404, "任务不存在")
        return RespModel(code=404, msg="任务不存在")
    cost = int(time.time() * 1000) - start_ms
    write_api_log("/rag/task", "", "", "", cost, 200, "查询成功")
    return RespModel(code=200, msg="查询成功", data=task_cache[task_id])

# 带元数据过滤RAG问答（无Rerank、无对话压缩）
@app.post("/rag/chat", response_model=RespModel, summary="支持文档过滤RAG问答（移除云端Rerank）")
@limiter.limit("10/minute")
def rag_chat(request: Request, req: QueryRequest):
    start_ms = int(time.time() * 1000)
    verify_api_key(request)
    sid = req.session_id
    question = req.question
    filter_src = req.filter_source
    logger.info(f"会话{sid} 提问：{question} 过滤文档：{filter_src if filter_src else '无'}")
    # 向量召回，无Rerank
    ref_docs = embed_tool.search_with_filter(question, filter_source=filter_src, top_k=6)
    context = "\n\n=====\n\n".join(ref_docs)
    # 会话，不压缩
    memory = SessionMemory(sid)
    history = memory.load()
    # LLM生成回答
    answer = llm_chat_call(context, question, history)
    memory.add_msg("user", question)
    memory.add_msg("assistant", answer)
    cost = int(time.time() * 1000) - start_ms
    write_api_log("/rag/chat", sid, question, filter_src if filter_src else "all", cost, 200, "问答成功（移除云端Rerank）")
    return RespModel(
        code=200,
        msg="问答成功（已移除云端Rerank，仅向量召回）",
        data={
            "session_id": sid,
            "question": question,
            "filter_source": filter_src,
            "reference_docs": ref_docs,
            "answer": answer
        }
    )

# 运维接口：按文档名批量删除向量
@app.post("/admin/delete_source", response_model=RespModel, summary="按文档名批量删除该文档全部向量片段")
@limiter.limit("2/minute")
def delete_source_vector(request: Request, source_name: str = Query(..., description="待删除的文档文件名")):
    start_ms = int(time.time() * 1000)
    verify_api_key(request)
    embed_tool.delete_by_source(source_name)
    cost = int(time.time() * 1000) - start_ms
    write_api_log("/admin/delete_source", "", "", source_name, cost, 200, f"文档{source_name}向量已删除")
    return RespModel(code=200, msg=f"文档{source_name}对应的向量片段全部删除")

# 运维接口：清空当前默认分片向量库
@app.post("/admin/clear_slice", response_model=RespModel, summary="清空默认分片全部向量知识库")
@limiter.limit("2/minute")
def clear_slice_lib(request: Request):
    start_ms = int(time.time() * 1000)
    verify_api_key(request)
    embed_tool.clear_slice_vector()
    cost = int(time.time() * 1000) - start_ms
    write_api_log("/admin/clear_slice", "", "", "", cost, 200, "分片向量库已清空")
    return RespModel(code=200, msg="默认分片向量库已全部清空")

# 运维接口：查询全部接口调用日志
@app.get("/admin/list_log", response_model=RespModel, summary="查询全部API调用日志")
@limiter.limit("5/minute")
def list_api_log(request: Request, limit: int = Query(20, ge=1, le=200)):
    start_ms = int(time.time() * 1000)
    verify_api_key(request)
    conn = sqlite3.connect(LOG_DB_PATH)
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM api_log ORDER BY id DESC LIMIT {limit}")
    rows = cur.fetchall()
    conn.close()
    log_list = []
    for row in rows:
        log_list.append({
            "id": row[0],
            "request_time": row[1],
            "api_path": row[2],
            "session_id": row[3],
            "question": row[4],
            "source_file": row[5],
            "cost_ms": row[6],
            "status_code": row[7],
            "msg": row[8]
        })
    cost = int(time.time() * 1000) - start_ms
    write_api_log("/admin/list_log", "", "", "", cost, 200, "日志查询成功")
    return RespModel(code=200, msg="日志查询成功", data={"logs": log_list})

# 启动入口（文件名匹配无热加载报错）
if __name__ == "__main__":
    logger.info("===== 第24天 RAG 纯净修复版启动：彻底移除Rerank、关闭对话压缩、修复异常返回类型 =====")
    embed_tool.load_all_local()
    import uvicorn
    uvicorn.run("day24_rerank_filter_log_rag:app", host="0.0.0.0", port=8000, reload=True)

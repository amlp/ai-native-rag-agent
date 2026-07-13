import requests
import json
import os
import time
import logging
from pathlib import Path
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from PyPDF2 import PdfReader

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

# ===================== 全局配置 =====================
SERVER_API_KEY = "rag-server-2026-secure-key-123456"
limiter = Limiter(key_func=get_remote_address)
MAX_QUESTION_LENGTH = 200
# 文件上传
UPLOAD_DIR = Path("./upload_docs")
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOW_EXT = {".txt", ".md", ".pdf"}
MAX_FILE_SIZE = 5 * 1024 * 1024
task_cache = {}
# 分块
CHUNK_SIZE = 350
CHUNK_OVERLAP = 80
# LLM 上下文最大字符限制
MAX_CONTEXT_LEN = 1800
# 向量/对话云端API
BASE_V1 = "https://public.cnki.net/llm/v1"
PRIVATE_API_KEY = os.getenv("PRIVATE_API_KEY")
EMBED_MODEL = "qwen3-embedding-8b"
CHAT_FULL_URL = "https://coder.cnki.net/KCoder-Claude/chat/completions"
KCODER_API_KEY = os.getenv("KCODER_API_KEY")
CHAT_MODEL = "KCoder-Claude"
# 向量库持久化
CHROMA_PATH = "./db_v2"
MEMORY_ROOT = Path("./sessions")
MEMORY_ROOT.mkdir(exist_ok=True)
DOC_FOLDER = Path("./docs")
DOC_FOLDER.mkdir(exist_ok=True)
MAX_RETRY = 2
# ====================================================

app = FastAPI(title="第23天 多格式文档RAG服务", version="1.0")
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

# 会话记忆（新增长对话压缩）
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
    # 超长对话压缩
    def compress_history(self):
        history = self.load()
        if len(history) <= 4:
            return history
        compress_prompt = f"""请把下面多轮对话压缩成简短摘要，保留关键业务信息：
{json.dumps(history, ensure_ascii=False)}
输出一段精简摘要，不要多余内容。"""
        headers = {"Authorization": f"Bearer {KCODER_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": CHAT_MODEL, "messages": [{"role": "user", "content": compress_prompt}], "temperature":0.3}
        resp = requests.post(CHAT_FULL_URL, headers=headers, json=payload, timeout=60)
        summary = resp.json()["choices"][0]["message"]["content"]
        # 压缩后只保留摘要+最新2轮对话
        new_hist = [{"role":"system","content":f"历史对话摘要：{summary}"}] + history[-2:]
        self.save(new_hist)
        return new_hist

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

# 向量检索工具
class EmbeddingRetrievalTool:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection("local_docs")
        self.doc_dir = DOC_FOLDER
        self.api_key = PRIVATE_API_KEY

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
                self.collection.add(embeddings=embeds, documents=chunks, ids=ids, metadatas=[{"source": f.name}]*len(chunks))
                logger.info(f"文档入库完成：{f.name}")
            except Exception as e:
                logger.error(f"{f.name} 入库失败：{str(e)}")
        logger.info(f"本地文档加载完毕，共{len(file_list)}个")
        return True

    # 向量召回
    def search_top3(self, query, top_k=3):
        q_emb = self.get_embedding([query])[0]
        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        return res["documents"][0]

    # 清空整个向量库（后台管理接口）
    def clear_all_vector(self):
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)
        logger.info("向量库全部清空")

embed_tool = EmbeddingRetrievalTool()

# 后台任务：解析文件+分块+入库
def background_parse_task(task_id: str, file_path: Path, coll):
    task_cache[task_id] = {"status": "processing", "msg": "解析文档、分块、向量化中"}
    try:
        full_text = parse_file_to_text(file_path)
        chunks = embed_tool.split_slide_chunk(full_text)
        embeds = embed_tool.get_embedding(chunks)
        ids = [f"{file_path.name}_{idx}" for idx in range(len(chunks))]
        coll.add(embeddings=embeds, documents=chunks, ids=ids, metadatas=[{"source": file_path.name}]*len(chunks))
        task_cache[task_id] = {"status": "success", "msg": f"{file_path.name} 解析入库成功"}
        logger.info(f"异步任务{task_id}处理完成")
    except Exception as err:
        task_cache[task_id] = {"status": "fail", "msg": str(err)}
        logger.error(f"异步任务{task_id}失败：{str(err)}")

# LLM 对话调用
def llm_chat_call(context_text, user_q, chat_history):
    # 智能裁剪上下文，不超限
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

# 全局异常捕获
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

@app.post("/rag/reload_docs", response_model=RespModel, summary="重载本地全部文档")
@limiter.limit("5/minute")
def reload_docs(request: Request):
    verify_api_key(request)
    ok = embed_tool.load_all_local()
    return RespModel(code=200 if ok else 500, msg="本地文档重载完成" if ok else "无文档可加载", data=None)

# 上传PDF/TXT/MD 异步解析入库
@app.post("/rag/upload_file", response_model=RespModel, summary="多格式文档上传异步入库")
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
        raise HTTPException(400, "文件最大限制5MB")
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(content)
    task_id = f"task_{int(time.time()*1000)}"
    bg_task.add_task(background_parse_task, task_id, save_path, embed_tool.collection)
    logger.info(f"文件{file.filename}已上传，任务ID：{task_id}")
    return RespModel(code=200, msg="文件接收成功，后台解析中", data={"task_id": task_id})

@app.get("/rag/task/{task_id}", response_model=RespModel, summary="查询上传任务状态")
@limiter.limit("20/minute")
def get_task_status(request: Request, task_id: str):
    verify_api_key(request)
    if task_id not in task_cache:
        return RespModel(code=404, msg="任务不存在", data=None)
    return RespModel(code=200, msg="查询成功", data=task_cache[task_id])

# RAG问答（新增对话自动压缩、上下文裁剪）
@app.post("/rag/chat", response_model=RespModel, summary="带对话压缩+上下文裁剪RAG问答")
@limiter.limit("10/minute")
def rag_chat(request: Request, req: QueryRequest):
    verify_api_key(request)
    sid = req.session_id
    question = req.question
    logger.info(f"会话{sid} 提问：{question}")
    # 检索文档
    ref_docs = embed_tool.search_top3(question, top_k=3)
    context = "\n\n=====\n\n".join(ref_docs)
    # 会话管理+长对话自动压缩
    memory = SessionMemory(sid)
    history = memory.load()
    # LLM生成回答
    answer = llm_chat_call(context, question, history)
    memory.add_msg("user", question)
    memory.add_msg("assistant", answer)
    return RespModel(
        code=200,
        msg="问答成功（多格式文档支持+对话自动压缩）",
        data={
            "session_id": sid,
            "question": question,
            "reference_docs": ref_docs,
            "answer": answer
        }
    )

# 后台管理：清空向量库
@app.post("/admin/clear_vector", response_model=RespModel, summary="清空全部向量知识库")
@limiter.limit("2/minute")
def clear_vector_lib(request: Request):
    verify_api_key(request)
    embed_tool.clear_all_vector()
    return RespModel(code=200, msg="向量库已全部清空", data=None)

# 启动入口（文件名完全匹配无热加载报错）
if __name__ == "__main__":
    logger.info("===== 第23天 多格式文档RAG服务启动 =====")
    embed_tool.load_all_local()
    import uvicorn
    uvicorn.run("day23_multi_file_rag:app", host="0.0.0.0", port=8000, reload=True)

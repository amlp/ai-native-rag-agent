import json
import os
import numpy as np
import faiss
from datetime import datetime
from openai import OpenAI

# ===================== 全局路径配置（复用Day15定义） =====================
STORE_PATH = "./agent_session"
DOC_ROOT = "./docs"
VECTOR_DB_DIR = "./vector_store"
VECTOR_INDEX_PATH = os.path.join(VECTOR_DB_DIR, "leave_index.faiss")
CHUNK_SIZE = 300
CHUNK_OVERLAP = 40
MAX_CONTEXT_LEN = 1000
TOP_K = 3  # 语义检索返回相似度前3个分片

# 自动创建目录
os.makedirs(STORE_PATH, exist_ok=True)
os.makedirs(DOC_ROOT, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# ===================== 1. 复用Day15会话持久化模块 =====================
class SessionStore:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.file_path = os.path.join(STORE_PATH, f"{session_id}.json")
        if not os.path.exists(self.file_path):
            init_data = {
                "chat_history": [],
                "doc_cache": {},
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.save_data(init_data)

    def load_data(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"读取会话文件异常：{e}")
            return {"chat_history": [], "doc_cache": {}}

    def save_data(self, data):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_chat_record(self, role: str, content: str):
        data = self.load_data()
        data["chat_history"].append({
            "role": role,
            "content": content,
            "record_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        self.save_data(data)

    def cache_doc_info(self, doc_name: str, chunk_list: list):
        data = self.load_data()
        data["doc_cache"][doc_name] = chunk_list
        self.save_data(data)

# ===================== 2. 复用Day15文档分片加载工具 =====================
def split_text_chunk(text: str):
    chunks = []
    text_length = len(text)
    start_index = 0
    while start_index < text_length:
        end_index = min(start_index + CHUNK_SIZE, text_length)
        chunks.append(text[start_index:end_index])
        start_index += (CHUNK_SIZE - CHUNK_OVERLAP)
    return chunks

def load_all_target_docs(folder_path: str = DOC_ROOT):
    document_collection = []
    for root, _, file_names in os.walk(folder_path):
        for file in file_names:
            if file.endswith((".md", ".txt")):
                full_file_path = os.path.join(root, file)
                try:
                    with open(full_file_path, "r", encoding="utf-8") as f:
                        raw_content = f.read()
                    text_chunks = split_text_chunk(raw_content)
                    document_collection.append({
                        "file_name": file,
                        "full_path": full_file_path,
                        "chunk_count": len(text_chunks),
                        "chunks": text_chunks
                    })
                except Exception as err:
                    print(f"文档读取失败 {full_file_path}，错误：{str(err)}")
    return document_collection

# ===================== 3. Day16新增：FAISS向量库+Embedding模块 =====================
class LocalFaissRAG:
    def __init__(self, key_json_path="./key.json"):
        # 初始化LLM/Embedding客户端
        with open(key_json_path, "r", encoding="utf-8") as f:
            key_cfg = json.load(f)
        self.client = OpenAI(api_key=key_cfg["api_key"], base_url=key_cfg["base_url"])
        self.embedding_model = "text-embedding-ada-002"
        self.index = None
        self.chunk_metadata = []  # 存储分片对应文档名、原文，索引和向量一一对应
        self._load_or_build_index()

    def get_text_embedding(self, text: str):
        """文本转向量"""
        resp = self.client.embeddings.create(input=text, model=self.embedding_model)
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        return vec

    def _load_or_build_index(self):
        """加载已有向量库 / 新建向量索引"""
        # 索引文件存在则直接读取
        if os.path.exists(VECTOR_INDEX_PATH):
            self.index = faiss.read_index(VECTOR_INDEX_PATH)
            # 加载分片元数据（文档映射关系）
            meta_path = VECTOR_INDEX_PATH.replace(".faiss", "_meta.json")
            with open(meta_path, "r", encoding="utf-8") as f:
                self.chunk_metadata = json.load(f)
            print(f"✅ 加载已有FAISS向量库，共存储 {len(self.chunk_metadata)} 个文本分片向量")
            return

        # 无索引文件：读取全部文档，批量向量化构建索引
        print("🔨 未检测到向量库，开始构建FAISS索引...")
        docs = load_all_target_docs()
        vec_list = []
        self.chunk_metadata = []
        for doc in docs:
            for chunk_text in doc["chunks"]:
                vec = self.get_text_embedding(chunk_text)
                vec_list.append(vec)
                self.chunk_metadata.append({
                    "doc_name": doc["file_name"],
                    "chunk_text": chunk_text
                })
        # 初始化FAISS索引
        vec_array = np.array(vec_list, dtype=np.float32)
        vec_dim = vec_array.shape[1]
        self.index = faiss.IndexFlatL2(vec_dim)
        self.index.add(vec_array)
        # 持久化索引文件 + 分片元数据
        faiss.write_index(self.index, VECTOR_INDEX_PATH)
        meta_path = VECTOR_INDEX_PATH.replace(".faiss", "_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunk_metadata, f, ensure_ascii=False, indent=2)
        print(f"✅ FAISS向量库构建完成，存储分片总数：{len(self.chunk_metadata)}")

    def semantic_search(self, query: str, top_k=TOP_K):
        """语义检索：输入问题，返回相似度最高的TopK分片"""
        query_vec = self.get_text_embedding(query)
        query_vec = np.expand_dims(query_vec, axis=0)
        dists, ids = self.index.search(query_vec, top_k)
        hit_result = []
        for idx in ids[0]:
            if idx < len(self.chunk_metadata):
                hit_result.append(self.chunk_metadata[idx])
        return hit_result

# ===================== 4. Day16整合RAG问答Agent =====================
class RAGDocAgent:
    def __init__(self, session_id="day16_rag_session"):
        self.session = SessionStore(session_id)
        self.rag = LocalFaissRAG()
        # 缓存文档到会话（复用Day15缓存逻辑）
        all_docs = load_all_target_docs()
        for doc in all_docs:
            self.session.cache_doc_info(doc["file_name"], doc["chunks"])

    def print_doc_stat(self):
        docs = load_all_target_docs()
        if not docs:
            print("docs文件夹为空，请放入leave_rule文档！")
            return
        print("===== 知识库文档统计 =====")
        total_chunk = 0
        for d in docs:
            print(f"文档：{d['file_name']} | 分片数：{d['chunk_count']}")
            total_chunk += d["chunk_count"]
        print(f"文档总数：{len(docs)} | 总分片：{total_chunk}")

    def semantic_qa(self, user_question: str):
        self.session.add_chat_record("user", user_question)
        # 1. 语义向量检索相关分片
        hit_chunks = self.rag.semantic_search(user_question, top_k=TOP_K)
        # 2. 拼接检索上下文
        context = ""
        if hit_chunks:
            last_doc = ""
            for item in hit_chunks:
                if item["doc_name"] != last_doc:
                    context += f"\n【文档来源：{item['doc_name']}】\n"
                    last_doc = item["doc_name"]
                context += f"{item['chunk_text']}\n"
        # 上下文截断
        if len(context) > MAX_CONTEXT_LEN:
            context = context[:MAX_CONTEXT_LEN] + "\n...(超长自动截断)"
        # 3. 生成回答
        if not context:
            ans = f"未在知识库中检索到和「{user_question}」语义相关的内容"
        else:
            ans = f"基于本地知识库语义检索结果回答你的问题：{user_question}\n关联规则上下文：\n{context}"
        self.session.add_chat_record("assistant", ans)
        return ans

# ===================== 程序入口测试 =====================
if __name__ == "__main__":
    agent = RAGDocAgent()
    agent.print_doc_stat()
    print("\n===== 测试语义检索问答 =====")
    # 测试模糊语义提问（Day15关键词匹配会失效，向量检索可正常命中）
    test_questions = [
        "事假最多能请多少天",
        "请假申请需要提前多久提交",
        "紧急事假审批时间要求"
    ]
    for q in test_questions:
        print(f"\n用户提问：{q}")
        res = agent.semantic_qa(q)
        print(f"Agent回答：\n{res}")
    # 打印持久化会话历史
    history = agent.session.load_data()["chat_history"]
    print("\n===== 持久化对话记录 =====")
    print(json.dumps(history, ensure_ascii=False, indent=2))

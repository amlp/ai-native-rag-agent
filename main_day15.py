import json
import os
from datetime import datetime

# ===================== 全局路径配置 =====================
STORE_PATH = "./agent_session"
DOC_ROOT = "./docs"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 40
MAX_CONTEXT_LEN = 1000

os.makedirs(STORE_PATH, exist_ok=True)
os.makedirs(DOC_ROOT, exist_ok=True)

# ===================== 模块1：会话持久化存储类 =====================
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

# ===================== 模块2：文档加载+文本分片工具 =====================
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

# ===================== 模块3：修复后的文档问答Agent =====================
class LocalDocAgent:
    def __init__(self, session_id: str = "day15_test_001"):
        self.session = SessionStore(session_id)
        self.all_docs = load_all_target_docs()
        self._cache_all_docs_to_session()

    def _cache_all_docs_to_session(self):
        for doc in self.all_docs:
            self.session.cache_doc_info(doc["file_name"], doc["chunks"])

    def print_all_docs_info(self):
        if not self.all_docs:
            print("docs文件夹为空，请放入leave_rule.md或txt文档！")
            return
        print("===== 本地文档加载完成统计 =====")
        total_chunk = 0
        for doc in self.all_docs:
            print(f"文档名：{doc['file_name']} | 分片数量：{doc['chunk_count']}")
            total_chunk += doc["chunk_count"]
        print(f"文档总数：{len(self.all_docs)} | 全部文本分片总数：{total_chunk}")

    def simple_doc_qa(self, user_question: str):
        self.session.add_chat_record("user", user_question)
        context_text = ""
        # 修复点1：提取核心业务关键词，不再用整句匹配
        core_words = ["请假", "事假", "年假", "调休"]
        hit_chunks_all = []

        for doc in self.all_docs:
            for chunk in doc["chunks"]:
                # 修复点2：只要包含任意一个业务关键词即命中
                if any(word in chunk for word in core_words):
                    hit_chunks_all.append((doc["file_name"], chunk))

        # 拼接匹配内容
        if hit_chunks_all:
            temp_doc = ""
            last_doc_name = ""
            for doc_name, chunk in hit_chunks_all:
                if doc_name != last_doc_name:
                    temp_doc += f"\n【文档：{doc_name}】\n"
                    last_doc_name = doc_name
                temp_doc += f"{chunk}\n"
            context_text = temp_doc

        # 上下文截断
        if len(context_text) == 0:
            answer = f"未在本地文档中检索到和「{user_question}」相关的内容，请更换提问关键词。"
        else:
            if len(context_text) > MAX_CONTEXT_LEN:
                context_text = context_text[:MAX_CONTEXT_LEN] + "\n...(上下文已自动截断)"
            answer = f"根据本地文档匹配结果回答你的问题：{user_question}\n匹配到的规则上下文：\n{context_text}"

        self.session.add_chat_record("assistant", answer)
        return answer

# ===================== 测试运行入口 =====================
if __name__ == "__main__":
    agent = LocalDocAgent(session_id="day15_demo_session")
    agent.print_all_docs_info()
    # 测试提问不变，依旧使用原问句
    user_q = "读取leave_rule.md里的请假规则"
    reply = agent.simple_doc_qa(user_q)
    print("\n===== Agent回答 =====")
    print(reply)
    session_data = agent.session.load_data()
    print("\n===== 磁盘持久化会话数据（重启程序不会丢失） =====")
    print(json.dumps(session_data["chat_history"], ensure_ascii=False, indent=2))

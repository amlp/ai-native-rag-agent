import asyncio
import re
import json
import os
import glob
import time
import numpy as np
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from llm_client import AsyncLLMClient, LLMRequest, Message
from simple_rag_keyword_day03 import SimpleRAG
import uvicorn

# ===================== 1. 对话持久化记忆 =====================
class ChatMemory:
    def __init__(self, save_path="chat_memory_day13.json"):
        self.save_path = save_path
        self.history = []
        self.load()

    def load(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.history = [Message(role=i["role"], content=i["content"]) for i in data]

    def add(self, role: str, content: str):
        self.history.append(Message(role=role, content=content))
        self.save()

    def save(self):
        dump_data = [{"role": m.role, "content": m.content} for m in self.history]
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, ensure_ascii=False, indent=2)

    def clear(self):
        self.history = []
        self.save()

# ===================== 2. ReAct循环状态持久化 =====================
class ReactStateStore:
    def __init__(self, state_path="react_state_day13.json"):
        self.state_path = state_path
        self.loop_records = []
        self.load()

    def load(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, "r", encoding="utf-8") as f:
                self.loop_records = json.load(f)

    def add_record(self, text: str):
        self.loop_records.append(text)
        self.save()

    def clear(self):
        self.loop_records = []
        self.save()

    def save(self):
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.loop_records, f, ensure_ascii=False, indent=2)

# ===================== 3. 工具调用日志持久化模块 =====================
class ToolLogger:
    def __init__(self, log_dir="./tool_logs_day13"):
        self.log_dir = log_dir
        self.err_log_path = os.path.join(log_dir, "error_log.jsonl")
        self.normal_log_path = os.path.join(log_dir, "tool_record.jsonl")
        os.makedirs(self.log_dir, exist_ok=True)

    def write_log(self, tool_name: str, args: str, result: str, cost_ms: float, success: bool):
        log_item = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "tool": tool_name,
            "params": args,
            "response": result,
            "cost_ms": round(cost_ms, 2),
            "is_success": success
        }
        line = json.dumps(log_item, ensure_ascii=False) + "\n"
        target = self.normal_log_path if success else self.err_log_path
        with open(target, "a", encoding="utf-8") as f:
            f.write(line)

# ===================== 4. 问答缓存模块（统一标点空白，消除key不匹配） =====================
class AnswerCache:
    def __init__(self, cache_path="./answer_cache.json", expire_sec=3600):
        self.cache_path = cache_path
        self.expire_sec = expire_sec
        self.cache_data = {}
        self.load_cache()

    def _normalize_q(self, q: str) -> str:
        q = re.sub(r"\s+", " ", q.strip())
        q = q.replace("？", "?")
        return q

    def load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache_data = json.load(f)
            except Exception:
                self.cache_data = {}
        self.clean_expire()

    def save_cache(self):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache_data, f, ensure_ascii=False, indent=2)

    def clean_expire(self):
        now = time.time()
        del_keys = []
        for k, v in self.cache_data.items():
            if now - v["time"] > self.expire_sec:
                del_keys.append(k)
        for k in del_keys:
            del self.cache_data[k]
        self.save_cache()

    def get(self, raw_question: str):
        q = self._normalize_q(raw_question)
        self.clean_expire()
        item = self.cache_data.get(q)
        if not item:
            return None
        return item["ans"]

    def set(self, raw_question, ans):
        q = self._normalize_q(raw_question)
        self.cache_data[q] = {"ans": ans, "time": time.time()}
        self.save_cache()

# ===================== 5. 文本切片工具 =====================
class TextSplitter:
    @staticmethod
    def split_text(text: str, chunk_size=150, chunk_overlap=30):
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start += (chunk_size - chunk_overlap)
        return [c for c in chunks if c]

# ===================== 6. 可增量更新FAISS向量库 =====================
class IncrementFaissStore:
    def __init__(self, dim=64, index_path="./faiss_index_day13.json", doc_map_path="./doc_map_day13.json"):
        self.dim = dim
        self.index_path = index_path
        self.doc_map_path = doc_map_path
        self.index = faiss.IndexFlatL2(dim)
        self.doc_list = []
        self.load_index()

    def text_to_vec(self, text: str):
        vec = np.zeros(self.dim, dtype=np.float32)
        for idx, c in enumerate(text):
            vec[idx % self.dim] += ord(c) / 100
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def add_doc(self, doc):
        vec = self.text_to_vec(doc)
        self.index.add(np.expand_dims(vec, axis=0))
        self.doc_list.append(doc)

    def batch_add(self, doc_arr):
        for d in doc_arr:
            self.add_doc(d)

    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.doc_map_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_list, f, ensure_ascii=False, indent=2)
        print("FAISS增量索引与文档映射已保存")

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.doc_map_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.doc_map_path, "r", encoding="utf-8") as f:
                self.doc_list = json.load(f)
            print("检测到本地FAISS增量索引，直接加载，支持追加新文档")
        else:
            print("无本地索引，全新构建向量库")

    def add_increment_chunk(self, chunk_list):
        self.batch_add(chunk_list)
        self.save_index()
        print(f"成功增量追加 {len(chunk_list)} 条文本分片，无需全量重建")

    def search_topk(self, query, top_k=2, sim_threshold=0.15):
        q_vec = self.text_to_vec(query)
        q_vec = np.expand_dims(q_vec, axis=0)
        dist, idx_arr = self.index.search(q_vec, top_k)
        res_docs = []
        for idx, dist_val in zip(idx_arr[0], dist[0]):
            sim = 1 - dist_val / 2
            if sim >= sim_threshold and 0 <= idx < len(self.doc_list):
                res_docs.append(self.doc_list[idx])
        return res_docs

# ===================== 7. 批量文档加载 =====================
class BatchDocLoader:
    @staticmethod
    def load_all_docs(folder_path="./data"):
        os.makedirs(folder_path, exist_ok=True)
        file_paths = glob.glob(os.path.join(folder_path, "*.txt")) + glob.glob(os.path.join(folder_path, "*.md"))
        all_raw_text = []
        for fp in file_paths:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        all_raw_text.append(content)
            except Exception as e:
                print(f"读取文件{fp}失败：{str(e)}")
        default_doc = """
年假1-3年5天，3-10年10天，不可跨年；
周末加班调休有效期6个月；
报销单笔超5000元需总监审批；
婚假统一3天，晚婚额外增加7天，当年休完不可跨年顺延；
# 事假管理制度
1. 员工事假需提前3个工作日向直属领导线上提交申请；
2. 每月事假累计不得超过3天，全年事假上限20天；
3. 事假无薪资，按当日基础工资扣除；
4. 事假不可和年假、调休合并连续超过15天；
5. 紧急突发事假需当日上午10点前补提交审批。
        """
        if not all_raw_text:
            all_raw_text.append(default_doc.strip())
        full_text = "\n".join(all_raw_text)
        chunk_list = TextSplitter.split_text(full_text, chunk_size=150, chunk_overlap=30)
        return chunk_list

# ===================== 8. 优化混合检索RAG =====================
class HybridFaissRAGTool:
    def __init__(self, rag: SimpleRAG, vec_store: IncrementFaissStore, logger: ToolLogger):
        self.rag = rag
        self.vec_store = vec_store
        self.logger = logger
        self.name = "doc_search"
        self.desc = "人事检索，增量FAISS+关键词加权Rerank"
        self.retry_times = 2
        self.high_weight_key = ["年假", "婚假", "事假", "调休", "报销", "育儿假"]

    async def run(self, query: str) -> str:
        start_time = time.time()
        err_msg = ""
        final_result = ""
        success_flag = False
        for i in range(self.retry_times):
            try:
                res = await self.rag.ask(query)
                context_data = res["context"]
                raw_docs = context_data if isinstance(context_data, list) else context_data.split("\n")
                raw_docs = [d.strip() for d in raw_docs if d.strip()]
                if not raw_docs:
                    final_result = "【工具失败】无匹配文档"
                    break
                top_vec_docs = self.vec_store.search_topk(query, top_k=2)
                if not top_vec_docs:
                    final_result = "【工具失败】知识库无匹配文档"
                    break
                score_list = []
                for doc in top_vec_docs:
                    weight = 0.1
                    for word in self.high_weight_key:
                        if word in doc and word in query:
                            weight = 0.8
                            break
                    score_list.append((weight, doc))
                score_list.sort(reverse=True, key=lambda x: x[0])
                unique_docs = []
                seen_set = set()
                for _, doc in score_list:
                    if doc not in seen_set:
                        seen_set.add(doc)
                        unique_docs.append(doc)
                top_final = unique_docs[:2]
                final_result = f"【FAISS精选知识库】\n{"\n".join(top_final)}"
                success_flag = True
                break
            except Exception as e:
                err_msg = str(e)
                await asyncio.sleep(0.5)
        if not success_flag:
            final_result = f"【工具失败】重试{self.retry_times}次未获取文档：{err_msg}"
        cost_ms = (time.time() - start_time) * 1000
        self.logger.write_log(self.name, query, final_result, cost_ms, success_flag)
        return final_result

# ===================== 9. 计算器工具 =====================
class CalcTool:
    def __init__(self, logger: ToolLogger):
        self.logger = logger
        self.name = "calculator"
        self.desc = "四则数学运算"
        self.retry_times = 1

    async def run(self, expr: str) -> str:
        start_time = time.time()
        err_msg = ""
        final_result = ""
        success_flag = False
        for i in range(self.retry_times):
            try:
                match = re.search(r"[\d\+\-\*\/\(\)\.]+", expr)
                if not match:
                    final_result = "【计算失败】无合法四则算式"
                    break
                safe_expr = match.group(0)
                result = eval(safe_expr)
                final_result = f"【计算结果】{safe_expr} = {result}"
                success_flag = True
                break
            except Exception as e:
                err_msg = str(e)
        if not success_flag:
            final_result = f"【计算失败】{err_msg}，请规范输出纯数字算式"
        cost_ms = (time.time() - start_time) * 1000
        self.logger.write_log(self.name, expr, cost_ms, success_flag)
        return final_result

# ===================== 10. Agent核心（彻底移除危险正则清洗） =====================
class CacheReflectAgent:
    def __init__(self):
        self.llm = AsyncLLMClient(provider="tongyi")
        self.rag = SimpleRAG()
        self.vec_store = IncrementFaissStore(dim=64)
        self.tool_logger = ToolLogger()
        self.cache = AnswerCache()
        self.tools = {
            "doc_search": HybridFaissRAGTool(self.rag, self.vec_store, self.tool_logger),
            "calculator": CalcTool(self.tool_logger)
        }
        self.chat_memory = ChatMemory()
        self.react_state = ReactStateStore()
        self.chunk_docs = BatchDocLoader.load_all_docs("./data")
        self.doc_text = "\n".join(self.chunk_docs)

    # 无任何正则，直接返回原文，不会清空答案
    @staticmethod
    def clean_think_block(raw_text: str) -> str:
        return raw_text.strip()

    async def load_knowledge(self):
        print("批量读取./data文档，自动切片...")
        await self.rag.load_doc(self.doc_text)
        if not os.path.exists(self.vec_store.index_path):
            self.vec_store.batch_add(self.chunk_docs)
            self.vec_store.save_index()
        print("增量FAISS向量库、问答缓存、反思Agent就绪\n")

    async def add_new_knowledge(self, raw_text: str):
        chunks = TextSplitter.split_text(raw_text)
        self.vec_store.add_increment_chunk(chunks)

    def build_prompt(self, user_question: str) -> str:
        base = f"""用户问题：{user_question}
可用工具：doc_search(人事检索)、calculator(数学计算)
【零容忍强制规则，违反作废回答】
1. 禁止**、#、markdown加粗等格式，仅输出纯文本；
2. 单轮仅输出1条Action，格式严格：Action:工具名|参数，无参数写空；
3. 复合需求分两轮：第一轮doc_search查资料，第二轮calculator计算；
4. 红线约束：
若doc_search返回【工具失败】/内容和问题无关，**绝对禁止引用任何外部法律、行业惯例、外部法条**，只能固定回复：暂无公司人事制度，无法确认该规定，请咨询公司HR；
5. 缺少计算参数（工龄、年限等），禁止调用计算器，直接向用户提问获取；
6. 【自我反思规则】
读完历史Thought/Observation后先自检：
- 若上一轮doc_search已返回完整相关文档，禁止重复执行相同关键词检索，直接使用已有文档作答；
- 现有信息是否足够完整回答用户？
- 工具返回内容是否和问题匹配？
- 有无缺失关键数据、矛盾信息、计算错误？
若信息不足/内容不匹配，必须继续输出Action补充查询/计算；
只有信息完整无缺失，才允许直接输出最终答案；
输出格式：
需要工具：你的思考与自检反思
Action:工具名|参数
无需工具：直接输出简洁完整答案，结束循环
本轮全部推理记录：
"""
        for item in self.react_state.loop_records:
            base += f"{item}\n"
        return base

    async def run_react_loop(self, user_question: str, max_loop=5):
        # 缓存读取逻辑
        cache_hit = self.cache.get(user_question)
        if cache_hit is not None:
            print("\n========== 缓存命中，跳过全部检索推理 ==========")
            print(cache_hit)
            return cache_hit

        self.chat_memory.add("user", user_question)
        self.react_state.clear()
        final_raw = ""
        for step in range(max_loop):
            prompt = self.build_prompt(user_question)
            try:
                req = LLMRequest(messages=[Message(role="user", content=prompt)])
                resp = await self.llm.chat(req)
                content = resp.content.strip()
                print(f"\n====第{step+1}轮推理（含自检反思）：\n{content}\n")
            except Exception as e:
                print(f"大模型瞬时异常，跳过本轮：{e}")
                await asyncio.sleep(1)
                continue

            if "Action:" in content:
                self.react_state.add_record(content)
                action_part = content.split("Action:")[-1].strip()
                parts = action.split("|", 1)
                if len(parts) == 1:
                    tool_raw = parts[0]
                    args = ""
                else:
                    tool_raw, args = parts
                tool_name_clean = re.sub(r"[\*\s]", "", tool_raw.strip())
                if tool_name_clean not in self.tools:
                    obs = "工具不存在，仅支持 doc_search / calculator"
                    print(f"工具异常：{obs}")
                    self.react_state.add_record(f"Observation:{obs}")
                    continue
                tool = self.tools[tool_name_clean]
                print(f"执行工具 {tool_name_clean} 参数：{args.strip()}")
                await asyncio.sleep(0.8)
                obs = await tool.run(args.strip())
                print(f"工具返回：{obs}\n")
                self.react_state.add_record(f"Observation:{obs}")
            else:
                final_raw = content
                break
        # 缓存存入原始完整文本，绝对不会空
        self.cache.set(user_question, final_raw)
        return final_raw

# ===================== FastAPI Web服务 =====================
app = FastAPI(title="人事RAG问答服务 Day13")
agent = CacheReflectAgent()

class QueryReq(BaseModel):
    question: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    await agent.load_knowledge()
    yield
    print("Web服务关闭，资源释放完成")

app.router.lifespan_context = lifespan

@app.post("/api/chat")
async def chat_api(req: QueryReq):
    raw_ans = await agent.run_react_loop(req.question, max_loop=5)
    clean_ans = agent.clean_think_block(raw_ans)
    return {
        "code": 0,
        "msg": "success",
        "data": {
            "question": req.question,
            "answer": clean_ans
        }
    }

@app.post("/api/add_knowledge")
async def add_knowledge(content: str):
    await agent.add_new_knowledge(content)
    return {"code":0, "msg":"增量知识库追加完成，无需重建全量向量库"}

# 命令行交互控制台
async def chat_console():
    await agent.load_knowledge()
    print("====命令行交互模式，输入exit退出====")
    print("新增知识库示例输入：add:这里粘贴新人事制度文本\n")
    while True:
        user_input = input("\n请输入问题：")
        if user_input.strip().lower() == "exit":
            print("对话、缓存、日志已保存，程序退出...")
            await asyncio.sleep(0.3)
            break
        if user_input.startswith("add:"):
            new_text = user_input.replace("add:", "").strip()
            await agent.add_new_knowledge(new_text)
            print("✅ 新制度增量入库完成，无需重启服务")
            continue
        if not user_input.strip():
            continue
        raw_ans = await agent.run_react_loop(user_input, max_loop=5)
        print(f"\n====Agent完整回答====\n{agent.clean_think_block(raw_ans)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        try:
            asyncio.run(chat_console())
        except Exception as e:
            import traceback
            print(f"\n全局异常捕获：{e}")
            print(traceback.format_exc())
            input("按回车关闭窗口")
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)
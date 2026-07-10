import asyncio
import re
import json
import os
import glob
import time
import numpy as np
import faiss
from llm_client import AsyncLLMClient, LLMRequest, Message
from simple_rag_keyword_day03 import SimpleRAG

# ===================== 1. 对话持久化记忆 =====================
class ChatMemory:
    def __init__(self, save_path="chat_memory_day11.json"):
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
    def __init__(self, state_path="react_state_day11.json"):
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

# ===================== 3. 工具调用日志持久化模块（Day11新增） =====================
class ToolLogger:
    def __init__(self, log_dir="./tool_logs"):
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
        if success:
            with open(self.normal_log_path, "a", encoding="utf-8") as f:
                f.write(line)
        else:
            with open(self.err_log_path, "a", encoding="utf-8") as f:
                f.write(line)

# ===================== 4. 持久化FAISS向量库（Day11核心新增） =====================
class PersistFaissStore:
    def __init__(self, dim=64, index_path="./faiss_index.bin", doc_map_path="./doc_map.json"):
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

    def add_doc(self, doc: str):
        vec = self.text_to_vec(doc)
        self.index.add(np.expand_dims(vec, axis=0))
        self.doc_list.append(doc)

    def batch_add(self, doc_arr):
        for d in doc_arr:
            self.add_doc(d)

    def save_index(self):
        # 保存向量索引
        faiss.write_index(self.index, self.index_path)
        # 保存文档映射关系
        with open(self.doc_map_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_list, f, ensure_ascii=False, indent=2)
        print("FAISS索引与文档映射已持久化保存至本地文件")

    def load_index(self):
        # 存在本地文件则直接加载，无需重新向量化
        if os.path.exists(self.index_path) and os.path.exists(self.doc_map_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.doc_map_path, "r", encoding="utf-8") as f:
                self.doc_list = json.load(f)
            print("检测到本地FAISS持久化文件，直接加载完成，跳过文档向量化")
        else:
            print("无本地FAISS索引文件，将重新构建向量库")

    def search_topk(self, query: str, top_k=2, sim_threshold=0.05):
        q_vec = self.text_to_vec(query)
        q_vec = np.expand_dims(q_vec, axis=0)
        dist, idx_arr = self.index.search(q_vec, top_k)
        res_docs = []
        for idx, dist_val in zip(idx_arr[0], dist[0]):
            sim = 1 - dist_val / 2
            if sim >= sim_threshold and 0 <= idx < len(self.doc_list):
                res_docs.append(self.doc_list[idx])
        return res_docs

# ===================== 5. 批量文档加载工具 =====================
class BatchDocLoader:
    @staticmethod
    def load_all_docs(folder_path="./data"):
        os.makedirs(folder_path, exist_ok=True)
        file_paths = glob.glob(os.path.join(folder_path, "*.txt")) + glob.glob(os.path.join(folder_path, "*.md"))
        all_text = []
        for fp in file_paths:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        all_text.append(content)
            except Exception as e:
                print(f"读取文件{fp}失败：{str(e)}")
        default_doc = """
年假1-3年5天，3-10年10天，不可跨年；
周末加班调休有效期6个月；
报销单笔超5000元需总监审批；
婚假统一3天，晚婚额外增加7天，当年休完不可跨年顺延。
        """
        if not all_text:
            all_text.append(default_doc.strip())
        return "\n".join(all_text)

# ===================== 6. 混合检索RAG工具（接入日志计时） =====================
class HybridFaissRAGTool:
    def __init__(self, rag: SimpleRAG, vec_store: PersistFaissStore, logger: ToolLogger):
        self.rag = rag
        self.vec_store = vec_store
        self.logger = logger
        self.name = "doc_search"
        self.desc = "人事制度检索，FAISS向量+关键词混合Rerank"
        self.retry_times = 2

    async def run(self, query: str) -> str:
        start_time = time.time()
        err_msg = ""
        final_result = ""
        success_flag = False
        for i in range(self.retry_times):
            try:
                res = await self.rag.ask(query)
                context_data = res["context"]
                if isinstance(context_data, list):
                    raw_docs = context_data
                else:
                    raw_docs = context_data.split("\n")
                raw_docs = [d.strip() for d in raw_docs if d.strip()]
                if not raw_docs:
                    final_result = "【工具失败】无匹配文档"
                    break
                self.vec_store.index.reset()
                self.vec_store.doc_list.clear()
                self.vec_store.batch_add(raw_docs)
                top_vec_docs = self.vec_store.search_topk(query, top_k=2)
                if not top_vec_docs:
                    final_result = "【工具失败】知识库无匹配文档"
                    break
                score_list = []
                for doc in top_vec_docs:
                    kw = 0.3 if any(k in doc for k in ["年假", "调休", "报销", "婚假", "事假"]) else 0
                    score_list.append((kw, doc))
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

# ===================== 7. 计算器工具（接入日志计时） =====================
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
        self.logger.write_log(self.name, expr, final_result, cost_ms, success_flag)
        return final_result

# ===================== 8. 带反思+持久化向量库Agent =====================
class PersistReflectAgent:
    def __init__(self):
        self.llm = AsyncLLMClient(provider="tongyi")
        self.rag = SimpleRAG()
        self.vec_store = PersistFaissStore(dim=64)
        self.tool_logger = ToolLogger()
        self.tools = {
            "doc_search": HybridFaissRAGTool(self.rag, self.vec_store, self.tool_logger),
            "calculator": CalcTool(self.tool_logger)
        }
        self.chat_memory = ChatMemory()
        self.react_state = ReactStateStore()
        self.doc_text = BatchDocLoader.load_all_docs("./data")

    async def load_knowledge(self):
        print("批量读取./data文件夹txt/md文档...")
        await self.rag.load_doc(self.doc_text)
        # 首次构建向量后保存索引
        if not os.path.exists(self.vec_store.index_path):
            raw_split = self.doc_text.split("\n")
            clean_docs = [d.strip() for d in raw_split if d.strip()]
            self.vec_store.batch_add(clean_docs)
            self.vec_store.save_index()
        print("FAISS持久化向量库就绪，带反思机制ReAct Agent启动\n")

    def build_prompt(self, user_question: str) -> str:
        base = f"""用户问题：{user_question}
可用工具：doc_search(人事检索)、calculator(数学计算)
【零容忍强制规则，违反作废回答】
1. 禁止**、#、markdown加粗等格式，仅输出纯文本；
2. 单轮仅输出1条Action，禁止同时调用多个工具；
3. 复合需求分两轮：第一轮doc_search查资料，第二轮calculator计算；
4. 红线约束：
   若doc_search返回【工具失败】/内容和问题无关，**绝对禁止引用任何外部法律、行业惯例、外部法条**，只能固定回复：暂无公司人事制度，无法确认该规定，请咨询公司HR；
5. 缺少计算参数（工龄、年限等），禁止调用计算器，直接向用户提问获取；
6. 【自我反思规则】
   读完历史Thought/Observation后先自检：
   - 现有信息是否足够完整回答用户？
   - 工具返回内容是否和问题匹配？
   - 有无缺失关键数据、矛盾信息、计算错误？
   若信息不足/内容不匹配，必须继续输出Action补充查询/计算；
   只有信息完整无缺失，才允许直接输出最终答案；
输出格式：
需要工具：Thought:你的思考与自检反思\nAction:工具名|参数
无需工具：直接输出简洁完整答案，结束循环
本轮全部推理记录：
"""
        for item in self.react_state.loop_records:
            base += f"{item}\n"
        return base

    async def run_react_loop(self, user_question: str, max_loop=5):
        self.chat_memory.add("user", user_question)
        self.react_state.clear()
        for step in range(max_loop):
            prompt = self.build_prompt(user_question)
            try:
                req = LLMRequest(messages=[Message(role="user", content=prompt)])
                resp = await self.llm.chat(req)
                content = resp.content.strip()
                print(f"\n====第{step+1}轮推理（含自检反思）：\n{content}\n")
            except Exception as e:
                print(f"大模型瞬时请求异常，跳过本轮：{e}")
                await asyncio.sleep(1)
                continue

            if "Action:" in content:
                self.react_state.add_record(content)
                action_part = content.split("Action:")[-1].strip()
                tool_raw, args = action_part.split("|", 1)
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
                self.chat_memory.add("assistant", content)
                return content
        final_prompt = f"结合全部工具记录，简洁回答用户问题，严格遵守不编造外部法规规则：{user_question}\n{self.react_state.loop_records}"
        final_res = await self.llm.chat(LLMRequest(messages=[Message(role="user", content=final_prompt)]))
        self.chat_memory.add("assistant", final_res.content.strip())
        return final_res.content.strip()

# ===================== 交互式控制台，优雅释放异步连接 =====================
async def chat_console():
    agent = PersistReflectAgent()
    await agent.load_knowledge()
    print("====交互式对话控制台启动，输入exit退出程序====")
    print("工具调用日志自动保存至 ./tool_logs 文件夹")
    print("FAISS向量持久化文件：faiss_index.bin、doc_map.json\n")
    while True:
        user_input = input("\n请输入你的问题：")
        if user_input.strip().lower() == "exit":
            print("对话记忆保存至 chat_memory_day11.json，推理状态保存至 react_state_day11.json，程序正在安全释放连接退出...")
            await asyncio.sleep(0.3)
            break
        if not user_input.strip():
            continue
        answer = await agent.run_react_loop(user_input, max_loop=5)
        print(f"\n====Agent完整回答====\n{answer}")

if __name__ == "__main__":
    try:
        asyncio.run(chat_console())
    except Exception as e:
        import traceback
        print(f"\n全局兜底异常捕获：{e}")
        print(traceback.format_exc())
        input("按回车键关闭窗口")
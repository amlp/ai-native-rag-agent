import asyncio
import re
import json
import os
import glob
import numpy as np
import faiss
from llm_client import AsyncLLMClient, LLMRequest, Message
from simple_rag_keyword_day03 import SimpleRAG

# 1. 对话持久化记忆
class ChatMemory:
    def __init__(self, save_path="chat_memory_day10.json"):
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

# 2. ReAct循环状态持久化
class ReactStateStore:
    def __init__(self, state_path="react_state_day10.json"):
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

# 3. FAISS向量工具 带相似度阈值过滤
class FaissVectorStore:
    def __init__(self, dim=64):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.doc_list = []

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

# 4. 批量文档加载工具：读取data文件夹下全部txt/md
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
        # 内置默认文档，无外部文件时兜底
        default_doc = """
年假1-3年5天，3-10年10天，不可跨年；
周末加班调休有效期6个月；
报销单笔超5000元需总监审批；
婚假统一3天，晚婚额外增加7天，当年休完不可跨年顺延。
        """
        if not all_text:
            all_text.append(default_doc.strip())
        return "\n".join(all_text)

# 5. FAISS混合检索RAG工具 带重试+结果去重
class HybridFaissRAGTool:
    def __init__(self, rag: SimpleRAG, vec_store: FaissVectorStore):
        self.rag = rag
        self.vec_store = vec_store
        self.name = "doc_search"
        self.desc = "人事制度检索，FAISS向量+关键词混合Rerank"
        self.retry_times = 2

    async def run(self, query: str) -> str:
        err_msg = ""
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
                    return "【工具失败】无匹配文档"
                self.vec_store.index.reset()
                self.vec_store.doc_list.clear()
                self.vec_store.batch_add(raw_docs)
                top_vec_docs = self.vec_store.search_topk(query, top_k=2)
                if not top_vec_docs:
                    return "【工具失败】知识库无匹配文档"
                score_list = []
                for doc in top_vec_docs:
                    kw = 0.3 if any(k in doc for k in ["年假", "调休", "报销", "婚假"]) else 0
                    score_list.append((kw, doc))
                score_list.sort(reverse=True, key=lambda x: x[0])
                unique_docs = []
                seen_set = set()
                for _, doc in score_list:
                    if doc not in seen_set:
                        seen_set.add(doc)
                        unique_docs.append(doc)
                top_final = unique_docs[:2]
                return f"【FAISS精选知识库】\n{"\n".join(top_final)}"
            except Exception as e:
                err_msg = str(e)
                await asyncio.sleep(0.5)
        return f"【工具失败】重试{self.retry_times}次未获取文档"

# 6. 计算器工具 过滤脏算式，异常降级
class CalcTool:
    def __init__(self):
        self.name = "calculator"
        self.desc = "四则数学运算"
        self.retry_times = 1

    async def run(self, expr: str) -> str:
        err_msg = ""
        for i in range(self.retry_times):
            try:
                match = re.search(r"[\d\+\-\*\/\(\)\.]+", expr)
                if not match:
                    return "【计算失败】无合法四则算式"
                safe_expr = match.group(0)
                result = eval(safe_expr)
                return f"【计算结果】{safe_expr} = {result}"
            except Exception as e:
                err_msg = str(e)
        return f"【计算失败】{err_msg}，请规范输出纯数字算式"

# 7. 带自我反思的ReAct Agent
class ReflectReActAgent:
    def __init__(self):
        self.llm = AsyncLLMClient(provider="tongyi")
        self.rag = SimpleRAG()
        self.vec_store = FaissVectorStore(dim=64)
        self.tools = {
            "doc_search": HybridFaissRAGTool(self.rag, self.vec_store),
            "calculator": CalcTool()
        }
        self.chat_memory = ChatMemory()
        self.react_state = ReactStateStore()
        # 批量加载本地文档
        self.doc_text = BatchDocLoader.load_all_docs("./data")

    async def load_knowledge(self):
        print("批量读取./data文件夹txt/md文档，初始化FAISS向量库...")
        await self.rag.load_doc(self.doc_text)
        print("批量知识库加载完成，带反思机制ReAct Agent就绪\n")

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
        # 达到最大轮次强制汇总兜底回答
        final_prompt = f"结合全部工具记录，简洁回答用户问题，严格遵守不编造外部法规规则：{user_question}\n{self.react_state.loop_records}"
        final_res = await self.llm.chat(LLMRequest(messages=[Message(role="user", content=final_prompt)]))
        self.chat_memory.add("assistant", final_res.content.strip())
        return final_res.content.strip()

# 交互式控制台，优雅关闭释放异步连接
async def chat_console():
    agent = ReflectReActAgent()
    await agent.load_knowledge()
    print("====交互式对话控制台启动，输入exit退出程序====")
    while True:
        user_input = input("\n请输入你的问题：")
        if user_input.strip().lower() == "exit":
            print("对话记忆保存至 chat_memory_day10.json，推理状态保存至 react_state_day10.json，程序正在安全释放连接退出...")
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
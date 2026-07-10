import asyncio
import re
import json
import os
import numpy as np
import faiss
from llm_client import AsyncLLMClient, LLMRequest, Message
from simple_rag_keyword_day03 import SimpleRAG

# 1. 对话持久化记忆
class ChatMemory:
    def __init__(self, save_path="chat_memory_day9.json"):
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

# 2. ReAct循环状态持久化（保存思考/行动/观察，重启续跑）
class ReactStateStore:
    def __init__(self, state_path="react_state.json"):
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

# 3. FAISS向量工具
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

    def search_topk(self, query: str, top_k=2):
        q_vec = self.text_to_vec(query)
        q_vec = np.expand_dims(q_vec, axis=0)
        dist, idx_arr = self.index.search(q_vec, top_k)
        res_docs = []
        for idx in idx_arr[0]:
            if 0 <= idx < len(self.doc_list):
                res_docs.append(self.doc_list[idx])
        return res_docs

# 4. FAISS混合检索RAG工具，修复list split，带重试
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
                    return "【工具失败】无匹配文档，禁止编造外部法规"
                # 批量写入FAISS
                self.vec_store.batch_add(raw_docs)
                top_vec_docs = self.vec_store.search_topk(query, top_k=2)
                # 关键词加分Rerank
                score_list = []
                for doc in top_vec_docs:
                    kw = 0.3 if any(k in doc for k in ["年假", "调休", "报销"]) else 0
                    score_list.append((kw, doc))
                score_list.sort(reverse=True, key=lambda x: x[0])
                top_final = [item[1] for item in score_list[:2]]
                return f"【FAISS精选知识库】\n{"\n".join(top_final)}"
            except Exception as e:
                err_msg = str(e)
                await asyncio.sleep(0.5)
        return f"【工具失败】重试{self.retry_times}次未获取文档，禁止编造外部法规"

# 5. 计算器：仅截取第一段合法算式，过滤脏数字
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
                    return "【计算失败】无合法算式"
                safe_expr = match.group(0)
                result = eval(safe_expr)
                return f"【计算结果】{safe_expr} = {result}"
            except Exception as e:
                err_msg = str(e)
        return f"【计算失败】{err_msg}"

# 6. ReAct Agent 支持多需求自动分轮调度工具
class MultiTaskReActAgent:
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
        self.doc_text = "年假1-3年5天，3-10年10天，不可跨年；周末加班调休有效期6个月；报销单笔超5000元需总监审批。"

    async def load_knowledge(self):
        print("加载知识库，初始化FAISS向量库...")
        await self.rag.load_doc(self.doc_text)
        print("FAISS混合检索就绪，多任务ReAct Agent启动\n")

    def build_prompt(self, user_question: str) -> str:
        base = f"""用户问题：{user_question}
可用工具：doc_search(人事检索)、calculator(数学计算)
强制规则：
1. 禁止**、#、markdown加粗等格式，只输出纯文本；
2. 单轮仅输出1条Action，不能同时调用多个工具；
3. 若同时包含【查资料】+【数学计算】两类需求：第一轮调用doc_search获取制度，第二轮调用calculator完成计算；
4. 工具返回【工具失败】时，绝对不能编造外部法律/通用规则，仅回复：暂无公司人事制度，无法确认；
输出格式：
需要工具：Thought:思考内容\nAction:工具名|参数
无需工具：直接输出简洁完整答案，结束循环
本轮历史推理记录：
"""
        for item in self.react_state.loop_records:
            base += f"{item}\n"
        return base

    async def run_react_loop(self, user_question: str, max_loop=4):
        self.chat_memory.add("user", user_question)
        self.react_state.clear()
        for step in range(max_loop):
            prompt = self.build_prompt(user_question)
            try:
                req = LLMRequest(messages=[Message(role="user", content=prompt)])
                resp = await self.llm.chat(req)
                content = resp.content.strip()
                print(f"\n====第{step+1}轮推理：\n{content}\n")
            except Exception as e:
                print(f"大模型瞬时请求异常，跳过本轮：{e}")
                await asyncio.sleep(1)
                continue

            if "Action:" in content:
                self.react_state.add_record(content)
                action_part = content.split("Action:")[-1].strip()
                tool_raw, args = action_part.split("|", 1)
                # 清洗工具名特殊符号防KeyError
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
        # 达到最大轮次强制汇总回答
        final_prompt = f"结合全部工具记录完整回答用户问题：{user_question}\n{self.react_state.loop_records}"
        final_res = await self.llm.chat(LLMRequest(messages=[Message(role="user", content=final_prompt)]))
        self.chat_memory.add("assistant", final_res.content.strip())
        return final_res.content.strip()

# 交互式持续对话控制台
async def chat_console():
    agent = MultiTaskReActAgent()
    await agent.load_knowledge()
    print("====交互式对话控制台启动，输入exit退出程序====")
    while True:
        user_input = input("\n请输入你的问题：")
        if user_input.strip().lower() == "exit":
            print("对话记忆保存至 chat_memory_day9.json，推理状态保存至 react_state.json，程序退出")
            break
        if not user_input.strip():
            continue
        answer = await agent.run_react_loop(user_input, max_loop=4)
        print(f"\n====Agent完整回答====\n{answer}")

if __name__ == "__main__":
    try:
        asyncio.run(chat_console())
    except Exception as e:
        import traceback
        print(f"\n全局兜底异常捕获：{e}")
        print(traceback.format_exc())
        input("按回车关闭窗口")
import asyncio
import re
import json
import os
import numpy as np
from llm_client import AsyncLLMClient, LLMRequest, Message
from simple_rag_keyword_day03 import SimpleRAG

# 持久化对话记忆
class ChatMemory:
    def __init__(self, save_path="chat_memory_day8.json"):
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

# 简易向量工具
class SimpleVector:
    @staticmethod
    def text_to_vec(text: str, dim=64):
        vec = np.zeros(dim)
        for idx, c in enumerate(text):
            vec[idx % dim] += ord(c) / 100
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    @staticmethod
    def cos_sim(vec1, vec2):
        return np.dot(vec1, vec2)

# 混合检索RAG工具 修复list split报错
class HybridRAGTool:
    def __init__(self, rag: SimpleRAG):
        self.rag = rag
        self.name = "doc_search"
        self.desc = "查询年假、加班调休、报销人事制度，混合关键词+向量检索"
        self.retry_times = 2
        self.doc_vec_cache = {}

    def build_doc_vec_cache(self, doc_list):
        self.doc_vec_cache.clear()
        for idx, doc in enumerate(doc_list):
            self.doc_vec_cache[idx] = SimpleVector.text_to_vec(doc)

    async def run(self, query: str) -> str:
        err_msg = ""
        for i in range(self.retry_times):
            try:
                res = await self.rag.ask(query)
                context_data = res["context"]
                # 修复：区分list/str类型
                if isinstance(context_data, list):
                    raw_docs = context_data
                else:
                    raw_docs = context_data.split("\n")
                # 过滤空行
                raw_docs = [doc.strip() for doc in raw_docs if doc.strip()]
                if not raw_docs:
                    return "【工具失败】知识库无匹配文档，不可用于推理作答"
                self.build_doc_vec_cache(raw_docs)
                q_vec = SimpleVector.text_to_vec(query)
                score_list = []
                for idx, doc in enumerate(raw_docs):
                    d_vec = self.doc_vec_cache[idx]
                    sim = SimpleVector.cos_sim(q_vec, d_vec)
                    kw_score = 0.3 if any(k in doc for k in ["年假", "调休", "报销"]) else 0
                    total_score = sim + kw_score
                    score_list.append((total_score, doc))
                # Rerank取top2
                score_list.sort(reverse=True, key=lambda x: x[0])
                top_docs = [item[1] for item in score_list[:2]]
                return f"【精选知识库】\n{"\n".join(top_docs)}"
            except Exception as e:
                err_msg = str(e)
                await asyncio.sleep(0.5)
        return f"【工具失败】知识库查询重试{self.retry_times}次未获取公司人事文档，不可用于推理作答"

# 计算器：正则只截取第一段合法算式，过滤后缀乱数字
class CalcTool:
    def __init__(self):
        self.name = "calculator"
        self.desc = "四则数学运算"
        self.retry_times = 1

    async def run(self, expr: str) -> str:
        err_msg = ""
        for i in range(self.retry_times):
            try:
                # 提取第一段连续合法运算字符，丢弃后面脏数字
                match = re.search(r"[\d\+\-\*\/\(\)\.]+", expr)
                if not match:
                    return "【计算失败】无合法四则算式"
                safe_expr = match.group(0)
                result = eval(safe_expr)
                return f"【计算结果】{safe_expr} = {result}"
            except Exception as e:
                err_msg = str(e)
        return f"【计算失败】{err_msg}，请规范输出纯算式"

# ReAct 智能体
class VectorReActAgent:
    def __init__(self):
        self.llm = AsyncLLMClient(provider="tongyi")
        self.rag = SimpleRAG()
        self.tools = {
            "doc_search": HybridRAGTool(self.rag),
            "calculator": CalcTool()
        }
        self.memory = ChatMemory()
        self.loop_memory = []
        self.doc_text = "年假1-3年5天，3-10年10天，不可跨年；周末加班调休有效期6个月；报销单笔超5000元需总监审批。"

    async def load_knowledge(self):
        print("加载企业知识库，构建向量缓存...")
        await self.rag.load_doc(self.doc_text)
        print("知识库+混合检索就绪，交互式Agent启动\n")

    def build_prompt(self, user_question: str) -> str:
        base = f"""用户问题：{user_question}
可用工具：doc_search(人事制度混合检索)、calculator(数学计算)
硬性强制规则：
1. 禁止输出**、#、markdown、加粗、特殊格式符号，仅纯文本；
2. 每一轮仅允许输出1条Action，禁止同时调用多个工具；
3. 若doc_search返回【工具失败】，禁止编造外部法律、通用规则，只能回复：暂无公司人事制度，无法确认该规定；
输出格式：
需要工具：Thought:思考内容\nAction:工具名|参数
无需工具：直接输出简洁最终答案，结束循环
本轮临时循环记录：
"""
        for item in self.loop_memory:
            base += f"{item}\n"
        return base

    async def run_react_loop(self, user_question: str, max_loop=3):
        self.memory.add("user", user_question)
        self.loop_memory.clear()
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
                self.loop_memory.append(content)
                action_part = content.split("Action:")[-1].strip()
                tool_raw, args = action_part.split("|", 1)
                # 清洗*、空格，防止KeyError
                tool_name_clean = re.sub(r"[\*\s]", "", tool_raw.strip())
                if tool_name_clean not in self.tools:
                    obs = f"工具不存在，仅支持 doc_search / calculator"
                    print(f"工具异常：{obs}")
                    self.loop_memory.append(f"Observation:{obs}")
                    continue
                tool = self.tools[tool_name_clean]
                print(f"执行工具 {tool_name_clean} 参数：{args.strip()}")
                await asyncio.sleep(0.8)
                obs = await tool.run(args.strip())
                print(f"工具返回：{obs}\n")
                self.loop_memory.append(f"Observation:{obs}")
            else:
                self.memory.add("assistant", content)
                return content
        # 达到最大轮次强制收尾
        final_prompt = f"结合全部工具记录简洁回答用户问题：{user_question}\n{self.loop_memory}"
        final_res = await self.llm.chat(LLMRequest(messages=[Message(role="user", content=final_prompt)]))
        self.memory.add("assistant", final_res.content.strip())
        return final_res.content.strip()

# 交互式控制台持续对话
async def chat_console():
    agent = VectorReActAgent()
    await agent.load_knowledge()
    print("====交互式对话控制台启动，输入exit退出程序====")
    while True:
        user_input = input("\n请输入你的问题：")
        if user_input.strip().lower() == "exit":
            print("对话已自动保存至 chat_memory_day8.json，程序退出")
            break
        if not user_input.strip():
            continue
        answer = await agent.run_react_loop(user_input, max_loop=3)
        print(f"\n====Agent完整回答====\n{answer}")

if __name__ == "__main__":
    try:
        asyncio.run(chat_console())
    except Exception as e:
        import traceback
        print(f"\n全局兜底异常捕获：{e}")
        print(traceback.format_exc())
        input("按回车键关闭窗口")
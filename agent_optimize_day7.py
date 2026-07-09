import asyncio
import re
import json
import os
 
from llm_client import AsyncLLMClient, LLMRequest, Message
from simple_rag_keyword_day03 import SimpleRAG

# 持久化记忆工具
class ChatMemory:
    def __init__(self, save_path="chat_memory.json"):
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

# RAG知识库工具（带重试容错）
class RAGTool:
    def __init__(self, rag: SimpleRAG):
        self.rag = rag
        self.name = "doc_search"
        self.desc = "查询年假、加班调休、报销人事制度"
        self.retry_times = 2

    async def run(self, query: str) -> str:
        err_msg = ""
        for i in range(self.retry_times):
            try:
                res = await self.rag.ask(query)
                return f"【知识库文档】\n{''.join(res['context'])}"
            except Exception as e:
                err_msg = str(e)
                await asyncio.sleep(0.5)
        return f"【知识库查询失败，重试{self.retry_times}次】{err_msg}"

# 数学计算器（容错）
class CalcTool:
    def __init__(self):
        self.name = "calculator"
        self.desc = "四则数学运算"
        self.retry_times = 1

    async def run(self, expr: str) -> str:
        err_msg = ""
        for i in range(self.retry_times):
            try:
                safe_expr = re.sub(r"[^\d\+\-\*\/\(\)\.]", "", expr)
                result = eval(safe_expr)
                return f"【计算结果】{safe_expr} = {result}"
            except Exception as e:
                err_msg = str(e)
        return f"【计算失败】{err_msg}，请检查算式格式"

# ReAct优化智能体
class OptimizedReActAgent:
    def __init__(self):
        self.llm = AsyncLLMClient(provider="tongyi")
        self.rag = SimpleRAG()
        self.tools = {
            "doc_search": RAGTool(self.rag),
            "calculator": CalcTool()
        }
        self.memory = ChatMemory()
        self.loop_memory = []
        self.doc_text = "年假1-3年5天，3-10年10天，不可跨年；周末加班调休有效期6个月。"

    async def load_knowledge(self):
        print("加载企业知识库...")
        await self.rag.load_doc(self.doc_text)
        print("知识库加载完成，优化版ReAct Agent就绪\n")

    def build_prompt(self, user_question: str) -> str:
        base = f"""用户问题：{user_question}
可用工具：doc_search(人事制度查询)、calculator(数学计算)
输出格式：
硬性规则：禁止输出**、#、加粗、markdown格式符号，只输出纯文本
输出格式：
需要工具：Thought:你的思考\nAction:工具名|参数
无需工具：直接输出简洁最终答案，结束循环
本轮循环记录：
"""
        for item in self.loop_memory:
            base += f"{item}\n"
        return base

    async def run_react_loop(self, user_question: str, max_loop=3):
        self.memory.add("user", user_question)
        for step in range(max_loop):
            prompt = self.build_prompt(user_question)
            try:
                req = LLMRequest(messages=[Message(role="user", content=prompt)])
                resp = await self.llm.chat(req)
                content = resp.content.strip()
                print(f"====第{step+1}轮输出：\n{content}\n")
            except Exception as e:
                print(f"大模型请求异常，跳过本轮：{e}")
                await asyncio.sleep(1)
                continue

            if "Action:" in content:
                self.loop_memory.append(content)
                action_part = content.split("Action:")[-1].strip()
                tool_name, args = action_part.split("|", 1)
                # 移除所有*、多余空格，只保留纯英文名称
                tool_name_clean = re.sub(r"[\*\s]", "", tool_name.strip())
                tool = self.tools[tool_name_clean]
                print(f"调用工具 {tool_name} 参数：{args}")
                await asyncio.sleep(0.8)
                obs = await tool.run(args.strip())
                print(f"工具返回：{obs}\n")
                self.loop_memory.append(f"Observation:{obs}")
            else:
                self.memory.add("assistant", content)
                return content
        final_prompt = f"结合所有信息简洁回答用户问题：{user_question}\n{self.loop_memory}"
        final_res = await self.llm.chat(LLMRequest(messages=[Message(role="user", content=final_prompt)]))
        self.memory.add("assistant", final_res.content.strip())
        return final_res.content.strip()

async def main():
    agent = OptimizedReActAgent()
    await agent.load_knowledge()
    question = "我入职2年，每年5天年假，工作4年总年假多少天？年假能不能跨年使用？"
    print("用户提问：", question, "\n")
    answer = await agent.run_react_loop(question, max_loop=3)
    print("====最终完整回答====\n", answer)
    print("\n对话已自动保存至 chat_memory.json，重启程序可读取历史")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        import traceback
        print(f"\n程序全局异常兜底：{e}")
        print(traceback.format_exc())
        input("按回车关闭窗口")
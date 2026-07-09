import asyncio
import re
from llm_client import AsyncLLMClient, LLMRequest, Message
from simple_rag_keyword_day03 import SimpleRAG

# 工具1：知识库检索
class RAGTool:
    def __init__(self, rag: SimpleRAG):
        self.rag = rag
        self.name = "doc_search"
        self.desc = "查询公司年假、加班调休、报销人事制度"

    async def run(self, query: str) -> str:
        res = await self.rag.ask(query)
        return f"【知识库】\n{''.join(res['context'])}"

# 工具2：计算器
class CalcTool:
    def __init__(self):
        self.name = "calculator"
        self.desc = "四则数学计算"

    async def run(self, expr: str) -> str:
        try:
            safe_expr = re.sub(r"[^\d\+\-\*\/\(\)\.]", "", expr)
            result = eval(safe_expr)
            return f"【计算结果】{safe_expr} = {result}"
        except Exception as e:
            return f"【计算失败】{e}"

# 多工具Agent
class MultiToolAgent:
    def __init__(self):
        self.llm = AsyncLLMClient(provider="tongyi")
        self.rag = SimpleRAG()
        self.tools = {
            "doc_search": RAGTool(self.rag),
            "calculator": CalcTool()
        }
        self.chat_history = []
        self.doc_text = """
年假：1-3年5天，3-10年10天，10年+15天；年假当年清零不可跨年。
调休：周末加班2倍折算，有效期6个月；法定加班无调休。
报销：差旅费当月报，单笔>5000需总监审批。
        """

    async def load_knowledge(self):
        print("加载知识库...")
        await self.rag.load_doc(self.doc_text)
        print("知识库加载完成\n")

    async def think_and_reply(self, user_input: str) -> str:
        self.chat_history.append(Message(role="user", content=user_input))
        system_prompt = """
工具：
1. doc_search：人事制度查询，调用格式 TOOL:doc_search|关键词
2. calculator：数学计算，调用格式 TOOL:calculator|算式
人事问题用doc_search，数学计算用calculator，无需工具直接输出答案。
        """
        # 该行与上方system_prompt同级，统一4空格缩进
        messages = [Message(role="system", content=system_prompt)] + self.chat_history
        req = LLMRequest(messages=messages)
        resp = await self.llm.chat(req)
        output = resp.content.strip()

        if output.startswith("TOOL:"):
            _, tool_part = output.split("TOOL:", 1)
            tool_name, args = tool_part.split("|", 1)
            tool = self.tools[tool_name.strip()]
            print(f"调用工具：{tool.name} 参数：{args}")
            tool_result = await tool.run(args.strip())
            final_msg = Message(role="user", content=f"{tool_result}\n回答用户：{user_input}")
            final_req = LLMRequest(messages=[Message(role="system", content="仅依据工具结果作答")] + self.chat_history + [final_msg])
            final_resp = await self.llm.chat(final_req)
            final_ans = final_resp.content.strip()
        else:
            final_ans = output

        self.chat_history.append(Message(role="assistant", content=final_ans))
        return final_ans

async def main():
    agent = MultiToolAgent()
    await agent.load_knowledge()
    q1 = "我入职2年，每年5天年假，工作4年一共能休多少天？年假可以留到下一年吗？"
    print("====问题1：", q1)
    ans1 = await agent.think_and_reply(q1)
    print("回答：\n", ans1, "\n")
    q2 = "那再额外加班折算10天调休，加上年假总休息天数是多少？"
    print("====问题2：", q2)
    ans2 = await agent.think_and_reply(q2)
    print("回答：\n", ans2)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        import traceback
        print(f"\n异常：{e}")
        print(traceback.format_exc())
        input("按回车关闭")
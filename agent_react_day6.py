import asyncio
import re
from llm_client import AsyncLLMClient, LLMRequest, Message
from simple_rag_keyword_day03 import SimpleRAG

# 知识库检索工具
class RAGTool:
    def __init__(self, rag: SimpleRAG):
        self.rag = rag
        self.name = "doc_search"
        self.desc = "查询年假、加班调休、报销人事制度"

    async def run(self, query: str) -> str:
        res = await self.rag.ask(query)
        return f"【知识库文档】\n{''.join(res['context'])}"

# 数学计算工具
class CalcTool:
    def __init__(self):
        self.name = "calculator"
        self.desc = "四则数学运算"

    async def run(self, expr: str) -> str:
        try:
            safe_expr = re.sub(r"[^\d\+\-\*\/\(\)\.]", "", expr)
            result = eval(safe_expr)
            return f"【计算结果】{safe_expr} = {result}"
        except Exception as e:
            return f"【计算失败】{str(e)}"

# ReAct 循环智能体
class ReActAgent:
    def __init__(self):
        self.llm = AsyncLLMClient(provider="tongyi")
        self.rag = SimpleRAG()
        self.tools = {
            "doc_search": RAGTool(self.rag),
            "calculator": CalcTool()
        }
        # 存放每一轮：思考、行动、观察
        self.loop_memory = []
        # 精简知识库，降低接口负载防超时
        self.doc_text = "年假1-3年5天，3-10年10天，不可跨年；周末加班调休有效期6个月。"

    async def load_knowledge(self):
        print("加载企业知识库...")
        await self.rag.load_doc(self.doc_text)
        print("知识库就绪，ReAct Agent 启动\n")

    def build_prompt(self, user_question: str) -> str:
        # 极致精简系统提示词，大幅减少token，缓解超时
        base = f"""任务：{user_question}
工具：doc_search(查人事), calculator(算数学)
格式规则：
1. 需要工具：Thought:思考内容\nAction:TOOL名称|参数
2. 无需工具直接输出答案，结束循环
历史记录：
"""
        # 拼接循环历史
        for item in self.loop_memory:
            base += f"{item}\n"
        base += "输出：\n"
        return base

    async def run_react_loop(self, user_question: str, max_loop=3):
        # 最多循环3轮，防止无限请求接口
        for step in range(max_loop):
            prompt = self.build_prompt(user_question)
            req = LLMRequest(messages=[Message(role="user", content=prompt)])
            resp = await self.llm.chat(req)
            content = resp.content.strip()
            print(f"====第{step+1}轮输出：\n{content}\n")

            # 判断是否需要调用工具
            if "Action:" in content:
                self.loop_memory.append(content)
                # 解析工具指令
                action_part = content.split("Action:")[-1].strip()
                tool_name, args = action_part.split("|", 1)
                tool = self.tools[tool_name.strip()]
                print(f"执行工具 {tool_name} 参数：{args}")
                await asyncio.sleep(0.8) # 间隔防接口限流
                obs = await tool.run(args.strip())
                print(f"工具观察结果：{obs}\n")
                self.loop_memory.append(f"Observation:{obs}")
            else:
                # 无Action，直接输出最终答案，终止循环
                return content
        # 达到最大轮次强制收尾
        final_prompt = f"结合历史记录完整回答用户问题：{user_question}\n{self.loop_memory}"
        final_res = await self.llm.chat(LLMRequest(messages=[Message(role="user", content=final_prompt)]))
        return final_res.content.strip()

async def main():
    agent = ReActAgent()
    await agent.load_knowledge()
    # 复合复杂问题，需要多轮思考+双工具调用
    question = "我入职2年，每年5天年假，工作4年总年假多少天？年假能不能跨年使用？"
    print("用户提问：", question, "\n")
    answer = await agent.run_react_loop(question, max_loop=3)
    print("====最终完整回答====\n", answer)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        import traceback
        print(f"\n程序异常：{e}")
        print(traceback.format_exc())
        input("按回车关闭窗口")
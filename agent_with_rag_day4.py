import asyncio
import traceback
from llm_client import AsyncLLMClient, LLMRequest, Message
from simple_rag_keyword_day03 import SimpleRAG

# Agent工具封装：知识库检索工具
class RAGTool:
    def __init__(self, rag: SimpleRAG):
        self.rag = rag
        self.name = "company_doc_search"
        self.desc = "用于查询公司年假、加班调休内部制度文档，当用户询问人事规则时必须调用"

    async def run(self, query: str) -> str:
        res = await self.rag.ask(query)
        context_text = "\n".join(res["context"])
        return f"【知识库检索结果】\n{context_text}"

# 基础Agent智能体
class DocAgent:
    def __init__(self):
        self.llm = AsyncLLMClient(provider="tongyi")
        self.rag = SimpleRAG()
        self.tool = RAGTool(self.rag)
        # 加载企业制度知识库
        self.doc_content = """
# 年假制度
1. 满1年不满3年：5天带薪年假；
2. 满3到10年：10天带薪年假；
3. 10年及以上：15天带薪年假；
4. 年假当年12.31清零，不可跨年累计。
# 加班调休规则
1. 工作日延时加班1小时以上可累计调休；
2. 周末加班按2倍时长折算调休；
3. 法定节假日加班仅发放加班费，不折算调休；
4. 调休有效期6个月，逾期作废。
# 报销规范
1. 差旅费需附带交通、住宿发票，当月发生当月报销；
2. 单笔报销超5000元需要部门总监二次审批。
        """

    async def init_knowledge(self):
        print("正在加载企业制度知识库...")
        await self.rag.load_doc(self.doc_content)
        print("知识库加载完成，Agent就绪\n")

    async def think_and_answer(self, user_question: str):
        # Agent思考系统提示词：自主判断是否调用检索工具
        system_prompt = """
你是企业人事智能Agent，拥有一个工具 company_doc_search。
规则：
1. 如果用户问题和年假、调休、报销、人事制度相关，必须调用工具查询内部文档，不能凭空回答；
2. 调用工具格式固定：SEARCH:{用户检索关键词}
3. 如果不需要查文档，直接输出最终答案；
4. 拿到检索结果后，结合文档完整、清晰回答用户问题。
"""
        # 第一轮思考：判断是否检索知识库
        req1 = LLMRequest(
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=f"用户问题：{user_question}")
            ]
        )
        resp1 = await self.llm.chat(req1)
        think_content = resp1.content.strip()

        # 检测是否需要调用检索工具
        if think_content.startswith("SEARCH:"):
            search_query = think_content.replace("SEARCH:", "").strip()
            print(f"Agent自主发起知识库检索，检索关键词：{search_query}")
            # 执行RAG检索
            search_result = await self.tool.run(search_query)
            print("知识库检索完成，结合文档生成最终答案...\n")
            # 第二轮：携带文档生成最终回答
            final_prompt = f"""
已有知识库检索资料：
{search_result}
用户原始问题：{user_question}
请严格依据上面文档完整回答，禁止编造信息。
            """
            req2 = LLMRequest(
                messages=[
                    Message(role="system", content="企业知识库问答助手，只基于提供文档作答"),
                    Message(role="user", content=final_prompt)
                ]
            )
            final_resp = await self.llm.chat(req2)
            return final_resp.content
        else:
            # 无需检索，直接输出答案
            return think_content

# 测试入口
async def main():
    agent = DocAgent()
    await agent.init_knowledge()

    # 测试1：需要调用知识库（人事制度类）
    q1 = "我入职2年，周末加班的调休多久过期？年假能放到明年吗？"
    print("=====问题1：", q1)
    ans1 = await agent.think_and_answer(q1)
    print("Agent回答：\n", ans1, "\n")

    # 测试2：不需要检索的通用问题（无匹配文档）
    q2 = "简单介绍下什么是AI Agent"
    print("=====问题2：", q2)
    ans2 = await agent.think_and_answer(q2)
    print("Agent回答：\n", ans2)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n程序捕获到异常：{e}")
        import traceback
        print("完整异常堆栈：")
        print(traceback.format_exc())
        input("按回车键关闭窗口...")


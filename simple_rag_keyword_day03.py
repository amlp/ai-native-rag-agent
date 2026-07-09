import asyncio
from llm_client import AsyncLLMClient, LLMRequest, Message

# 简易关键词检索，无向量、无JSON解析报错
class KeywordDB:
    def __init__(self):
        self.text_chunks = []

    def add_text(self, chunk: str):
        self.text_chunks.append(chunk)

    def search(self, query: str, top_k=3):
        query_words = set(query.replace("，"," ").replace("。"," ").split())
        score_list = []
        for chunk in self.text_chunks:
            chunk_words = set(chunk.replace("，"," ").replace("。"," ").split())
            hit = len(query_words & chunk_words)
            score_list.append(hit)
        sorted_idx = sorted(range(len(score_list)), key=lambda i:-score_list[i])
        return [self.text_chunks[i] for i in sorted_idx[:top_k]]

# 文本分块工具
def split_text(text: str, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    total = len(text)
    while start < total:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# 完整RAG主类
class SimpleRAG:
    def __init__(self):
        self.llm_client = AsyncLLMClient(provider="tongyi")
        self.db = KeywordDB()

    async def load_doc(self, doc_text: str):
        chunks = split_text(doc_text)
        print(f"文档分块完成，共{len(chunks)}块")
        for ck in chunks:
            self.db.add_text(ck)
        print("文档入库完成")

    async def ask(self, question: str, top_k=3):
        context_list = self.db.search(question, top_k)
        context = "\n---参考文档片段---\n".join(context_list)
        prompt = f"""
严格仅使用下面提供的文档回答用户问题，禁止编造文档不存在的信息：
【参考文档】
{context}
【用户问题】
{question}
        """
        llm_req = LLMRequest(
            messages=[
                Message(role="system", content="企业知识库问答助手，只依据参考文档作答"),
                Message(role="user", content=prompt)
            ]
        )
        resp = await self.llm_client.chat(llm_req)
        return {
            "context": context_list,
            "answer": resp.content,
            "token_total": resp.total_tokens
        }

# 测试入口
async def main():
    rag = SimpleRAG()
    doc = """
# 年假制度
1. 满1年不满3年：5天年假；
2. 满3到10年：10天；
3. 10年以上：15天；
4. 年假当年清零，不可跨年。
# 加班调休
周末加班1:2折算调休，有效期6个月；法定加班只发加班费，无调休。
"""
    await rag.load_doc(doc)
    q1 = "入职2年年假几天？能留到明年吗？"
    res1 = await rag.ask(q1)
    print("问题：", q1)
    print("检索匹配文档：", res1["context"])
    print("RAG回答：", res1["answer"])

    print("\n====分割线====\n")
    q2 = "周末加班调休多久过期？"
    res2 = await rag.ask(q2)
    print("问题：", q2)
    print("检索匹配文档：", res2["context"])
    print("RAG回答：", res2["answer"])

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n程序异常：{e}")
        input("按回车键关闭窗口...")
import asyncio
import logging
import os
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import aiohttp

# 日志初始化
def setup_logger():
    logger = logging.getLogger("llm_client")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    file_handler = logging.FileHandler("llm_run.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()
load_dotenv()
print("读取BASE_URL:", os.getenv("TONGYI_BASE_URL"))
print("读取API_KEY:", os.getenv("TONGYI_API_KEY"))

# 数据模型
class Message(BaseModel):
    role: str = Field(..., pattern=r"^(user|assistant|system)$")
    content: str

class LLMRequest(BaseModel):
    model: str = "KCoder"
    messages: list[Message]
    temperature: float = Field(0.2, ge=0, le=1)

class LLMResponse(BaseModel):
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class AsyncLLMClient:
    def __init__(self, provider: str = "tongyi", max_concurrent: int = 8):
        self.provider = provider
        self.semaphore = asyncio.Semaphore(max_concurrent)  # 并发限流
        if provider == "tongyi":
            self.api_key = os.getenv("TONGYI_API_KEY")
            self.base_url = os.getenv("TONGYI_BASE_URL")
        else:
            raise ValueError("当前仅支持tongyi(KCoder)")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def chat(self, req: LLMRequest) -> LLMResponse:
        """单次对话接口"""
        async with self.semaphore:
            payload = req.model_dump()
            url = self.base_url
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        headers=self.headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=120)
                    ) as resp:
                        resp_text = await resp.text()
                        if resp.status != 200:
                            logger.error(f"接口异常 状态码:{resp.status} 返回:{resp_text}")
                            raise Exception(f"KCoder调用失败：{resp_text}")
                        data = await resp.json()
                        choice = data["choices"][0]["message"]
                        usage = data["usage"]
                        return LLMResponse(
                            content=choice["content"],
                            prompt_tokens=usage["prompt_tokens"],
                            completion_tokens=usage["completion_tokens"],
                            total_tokens=usage["total_tokens"]
                        )
            except Exception as e:
                logger.error(f"单次调用异常: {str(e)}")
                raise e

    async def batch_chat(self, req_list: list[LLMRequest]) -> list[LLMResponse | str]:
        """批量并发调用，失败返回异常字符串"""
        tasks = []
        for req in req_list:
            tasks.append(self._safe_chat(req))
        start = time.time()
        result_list = await asyncio.gather(*tasks)
        cost = round(time.time() - start, 2)
        total_tokens = 0
        success_cnt = 0
        for res in result_list:
            if isinstance(res, LLMResponse):
                total_tokens += res.total_tokens
                success_cnt += 1
        logger.info(f"批量执行完成，总耗时{cost}s，成功{success_cnt}/{len(req_list)}，总Token消耗{total_tokens}")
        return result_list

    async def _safe_chat(self, req: LLMRequest):
        """捕获单条异常，不打断整个批量任务"""
        try:
            return await self.chat(req)
        except Exception as e:
            return f"调用失败: {str(e)}"

# 测试入口
async def main():
    client = AsyncLLMClient(provider="tongyi", max_concurrent=5)
    # 单条测试
    test_single = LLMRequest(
        messages=[
            Message(role="system", content="你是AI开发工程师"),
            Message(role="user", content="RAG和AI Agent区别")
        ]
    )
    res_single = await client.chat(test_single)
    logger.info("单条输出：\n" + res_single.content)

    # 批量测试：3条问题并发
    batch_reqs = [
        LLMRequest(messages=[Message(role="user", content="什么是RAG")]),
        LLMRequest(messages=[Message(role="user", content="什么是AI Agent")]),
        LLMRequest(messages=[Message(role="user", content="RAG如何结合Agent使用")]),
    ]
    batch_res = await client.batch_chat(batch_reqs)
    for idx, item in enumerate(batch_res):
        logger.info(f"【批量问答{idx+1}】{item}")

if __name__ == "__main__":
    asyncio.run(main())
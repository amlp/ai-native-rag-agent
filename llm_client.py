import asyncio
import logging
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import aiohttp

# 日志初始化
def setup_logger():
    logger = logging.getLogger("llm_client")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    file_handler = logging.FileHandler("llm_run.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()
# 加载环境变量
load_dotenv()
# 调试打印读取结果，排查.env读取失败
print("读取BASE_URL:", os.getenv("TONGYI_BASE_URL"))
print("读取API_KEY:", os.getenv("TONGYI_API_KEY"))

# 数据模型，完全匹配KCoder OpenAI兼容入参
class Message(BaseModel):
    role: str = Field(..., pattern=r"^(user|assistant|system)$")
    content: str

class LLMRequest(BaseModel):
    model: str = "KCoder"  # 知网固定模型名
    messages: list[Message]
    temperature: float = Field(0.2, ge=0, le=1)  # 代码场景推荐0.2
    max_tokens: int = Field(4096, gt=0)  # 文档推荐4096

class LLMResponse(BaseModel):
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class AsyncLLMClient:
    def __init__(self, provider: str = "tongyi"):
        self.provider = provider
        if provider == "tongyi":
            self.api_key = os.getenv("TONGYI_API_KEY")
            self.base_url = os.getenv("TONGYI_BASE_URL")
        else:
            raise ValueError("当前仅支持tongyi(KCoder)")
        # 官方标准鉴权头 Bearer 格式（CodeGeeX插件统一标准）
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def chat(self, req: LLMRequest) -> LLMResponse:
        payload = req.model_dump()
        # 自动拼接标准chat/completions端点
        url = self.base_url
        print("最终请求完整地址：", url)
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
            logger.error(f"调用异常详情: {str(e)}")
            raise e

    async def batch_chat(self, req_list: list[LLMRequest]) -> list[LLMResponse]:
        tasks = [self.chat(req) for req in req_list]
        return await asyncio.gather(*tasks)

# 测试入口
async def main():
    client = AsyncLLMClient(provider="tongyi")
    test_req = LLMRequest(
        messages=[
            Message(role="system", content="你是专业AI开发工程师，简洁回答问题"),
            Message(role="user", content="简单说明RAG和AI Agent的区别")
        ]
    )
    res = await client.chat(test_req)
    logger.info("模型输出内容：\n" + res.content)
    logger.info(f"总消耗Token：{res.total_tokens}")

if __name__ == "__main__":
    asyncio.run(main())
from collections.abc import AsyncIterator

from agentic_rag.config import Settings
from agentic_rag.schemas import RouteDecision
from agentic_rag.services.local_inference import LocalInferenceService


class LLMServices:
    # 这是一个很薄的服务层：
    # 它的作用主要是把“工作流层”和“具体推理实现”解耦。
    def __init__(self, settings: Settings, inference_service: LocalInferenceService) -> None:
        self.settings = settings
        self.inference_service = inference_service

    async def route_question(self, question: str) -> RouteDecision:
        # 路由判断委托给统一推理服务。
        return await self.inference_service.route_question(question)

    async def answer_directly(self, question: str) -> str:
        # 直接回答同样委托给统一推理服务。
        return await self.inference_service.answer_directly(question)

    async def stream_answer_directly(self, question: str) -> AsyncIterator[str]:
        async for chunk in self.inference_service.stream_generate(
            self.settings.direct_qa_system_prompt,
            question,
            max_new_tokens=1200,
            temperature=0.2,
        ):
            yield chunk

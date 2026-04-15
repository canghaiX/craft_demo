from agentic_rag.config import Settings
from agentic_rag.schemas import RouteDecision
from agentic_rag.services.local_inference import LocalInferenceService


class LLMServices:
    def __init__(self, settings: Settings, inference_service: LocalInferenceService) -> None:
        self.settings = settings
        self.inference_service = inference_service

    async def route_question(self, question: str) -> RouteDecision:
        return await self.inference_service.route_question(question)

    async def answer_directly(self, question: str) -> str:
        return await self.inference_service.answer_directly(question)

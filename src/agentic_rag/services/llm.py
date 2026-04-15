from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agentic_rag.config import Settings
from agentic_rag.schemas import RouteDecision


class LLMServices:
    def __init__(self, settings: Settings) -> None:
        common_kwargs = {
            "api_key": settings.openai_api_key,
            "base_url": settings.openai_base_url,
            "temperature": 0,
        }
        self.router_llm = ChatOpenAI(model=settings.router_model, **common_kwargs)
        self.answer_llm = ChatOpenAI(model=settings.agent_model, **common_kwargs)
        self.settings = settings

        self.router_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", settings.router_system_prompt),
                (
                    "human",
                    "Question:\n{question}\n\n"
                    "Return the best route and a short reason.",
                ),
            ]
        )
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", settings.direct_qa_system_prompt),
                ("human", "{question}"),
            ]
        )

    async def route_question(self, question: str) -> RouteDecision:
        if any(keyword in question.lower() for keyword in ["pdf", "文档", "资料", "手册", "合同"]):
            return RouteDecision(route="lightrag", reason="Question explicitly refers to document corpus.")

        chain = self.router_prompt | self.router_llm.with_structured_output(RouteDecision)
        return await chain.ainvoke({"question": question})

    async def answer_directly(self, question: str) -> str:
        chain = self.answer_prompt | self.answer_llm
        response = await chain.ainvoke({"question": question})
        return response.content if isinstance(response.content, str) else str(response.content)

from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, StateGraph

from agentic_rag.schemas import RouteDecision
from agentic_rag.services.lightrag_service import LightRAGService
from agentic_rag.services.llm import LLMServices


class AgentState(TypedDict, total=False):
    question: str
    forced_route: Literal["auto", "direct", "lightrag"]
    route: Literal["direct", "lightrag"]
    reason: str
    answer: str


class AgenticRAGWorkflow:
    def __init__(self, llm_services: LLMServices, lightrag_service: LightRAGService) -> None:
        self.llm_services = llm_services
        self.lightrag_service = lightrag_service
        self.graph = self._build_graph().compile()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("router", self._router_node)
        graph.add_node("direct_answer", self._direct_answer_node)
        graph.add_node("lightrag_answer", self._lightrag_answer_node)
        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router",
            self._pick_route,
            {
                "direct": "direct_answer",
                "lightrag": "lightrag_answer",
            },
        )
        graph.add_edge("direct_answer", END)
        graph.add_edge("lightrag_answer", END)
        return graph

    async def _router_node(self, state: AgentState) -> AgentState:
        forced = state.get("forced_route", "auto")
        if forced == "direct":
            decision = RouteDecision(route="direct", reason="已按请求强制走 direct。")
        elif forced == "lightrag":
            decision = RouteDecision(route="lightrag", reason="已按请求强制走 lightrag。")
        else:
            decision = await self.llm_services.route_question(state["question"])

        return {"route": decision.route, "reason": decision.reason}

    def _pick_route(self, state: AgentState) -> Annotated[str, "route"]:
        return state["route"]

    async def _direct_answer_node(self, state: AgentState) -> AgentState:
        answer = await self.llm_services.answer_directly(state["question"])
        return {"answer": answer}

    async def _lightrag_answer_node(self, state: AgentState) -> AgentState:
        answer = await self.lightrag_service.query(state["question"])
        return {"answer": answer}

    async def ainvoke(self, question: str, forced_route: Literal["auto", "direct", "lightrag"]) -> AgentState:
        return await self.graph.ainvoke({"question": question, "forced_route": forced_route})

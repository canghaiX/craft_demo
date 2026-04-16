from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, StateGraph

from agentic_rag.schemas import RouteDecision
from agentic_rag.services.lightrag_service import LightRAGService
from agentic_rag.services.llm import LLMServices


class AgentState(TypedDict, total=False):
    # LangGraph 在节点间传递的状态对象。
    question: str
    forced_route: Literal["auto", "direct", "lightrag"]
    route: Literal["direct", "lightrag"]
    reason: str
    answer: str


class AgenticRAGWorkflow:
    # 这个工作流负责把“问答系统”拆成路由与回答两个阶段。
    def __init__(self, llm_services: LLMServices, lightrag_service: LightRAGService) -> None:
        self.llm_services = llm_services
        self.lightrag_service = lightrag_service
        self.graph = self._build_graph().compile()

    def _build_graph(self) -> StateGraph:
        # 图结构很简单：
        # router -> direct_answer 或 lightrag_answer -> END
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
        # 用户可以强制指定路由；
        # 否则交给路由模型判断。
        forced = state.get("forced_route", "auto")
        if forced == "direct":
            decision = RouteDecision(route="direct", reason="已按请求强制走 direct。")
        elif forced == "lightrag":
            decision = RouteDecision(route="lightrag", reason="已按请求强制走 lightrag。")
        else:
            decision = await self.llm_services.route_question(state["question"])

        return {"route": decision.route, "reason": decision.reason}

    def _pick_route(self, state: AgentState) -> Annotated[str, "route"]:
        # 条件边读取 route 字段，决定走哪条分支。
        return state["route"]

    async def _direct_answer_node(self, state: AgentState) -> AgentState:
        # 普通直答分支。
        answer = await self.llm_services.answer_directly(state["question"])
        return {"answer": answer}

    async def _lightrag_answer_node(self, state: AgentState) -> AgentState:
        # 知识图谱检索问答分支。
        answer = await self.lightrag_service.query(state["question"])
        return {"answer": answer}

    async def ainvoke(self, question: str, forced_route: Literal["auto", "direct", "lightrag"]) -> AgentState:
        # 对外暴露的异步调用入口。
        return await self.graph.ainvoke({"question": question, "forced_route": forced_route})

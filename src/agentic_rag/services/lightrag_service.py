from __future__ import annotations

import re
from typing import Any

from agentic_rag.config import Settings
from agentic_rag.services.local_inference import LocalInferenceService


class LightRAGService:
    def __init__(self, settings: Settings, inference_service: LocalInferenceService) -> None:
        self.settings = settings
        self.inference_service = inference_service
        self._rag: Any | None = None

    async def _build_rag(self) -> Any:
        try:
            from lightrag import LightRAG, QueryParam
            from lightrag.utils import EmbeddingFunc
        except ImportError as exc:
            raise RuntimeError(
                "尚未安装 LightRAG 依赖，请先执行 `pip install -e .`。"
            ) from exc

        async def llm_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> Any:
            history_text = ""
            if history_messages:
                history_lines = [
                    f"{message.get('role', 'user')}: {message.get('content', '')}"
                    for message in history_messages
                ]
                history_text = "\n\n对话历史：\n" + "\n".join(history_lines)
            user_prompt = f"{history_text}\n\n{prompt}".strip()
            return await self.inference_service.generate(
                system_prompt or "你是一名有帮助的智能助手。",
                user_prompt,
                max_new_tokens=int(kwargs.get("max_tokens", 512)),
                temperature=float(kwargs.get("temperature", 0.1)),
            )

        async def embedding_func_impl(texts: list[str]) -> list[list[float]]:
            return await self.inference_service.embed(texts)

        embedding_func = EmbeddingFunc(
            embedding_dim=self.settings.lightrag_embed_dim,
            func=embedding_func_impl,
        )

        rag = LightRAG(
            working_dir=str(self.settings.lightrag_working_dir),
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
        )
        await rag.initialize_storages()

        # Some LightRAG versions require explicit pipeline state initialization.
        initialize_pipeline = getattr(rag, "initialize_pipeline_status", None)
        if callable(initialize_pipeline):
            maybe_awaitable = initialize_pipeline()
            if maybe_awaitable is not None and hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable

        setattr(rag, "_query_param_cls", QueryParam)
        return rag

    async def get_rag(self) -> Any:
        if self._rag is None:
            self._rag = await self._build_rag()
        return self._rag

    async def ingest_document(self, text: str, file_path: str | None = None) -> int:
        rag = await self.get_rag()
        if not text.strip():
            raise ValueError("解析后的 PDF 文本为空，无法建立索引。")

        payload = text if not file_path else f"[SOURCE: {file_path}]\n\n{text}"
        await rag.ainsert(payload)

        # We return a simple estimate because LightRAG does not expose chunk count consistently.
        return max(1, len(text) // 1200)

    async def query(self, question: str, mode: str | None = None) -> str:
        rag = await self.get_rag()
        query_param_cls = getattr(rag, "_query_param_cls")
        query_mode = mode or self.settings.lightrag_query_mode

        for candidate in await self._build_query_candidates(question):
            result = await rag.aquery(
                candidate,
                param=query_param_cls(mode=query_mode),
            )
            answer = self._normalize_query_result(result)
            if answer is not None:
                return answer

        return (
            "LightRAG 已正确命中，但查询阶段没有产出最终答案。"
            "当前知识图谱看起来已经存在，不过这个问题返回了空结果。"
        )

    async def _build_query_candidates(self, question: str) -> list[str]:
        candidates: list[str] = []

        def add_candidate(text: str) -> None:
            normalized = re.sub(r"\s+", " ", text).strip()
            if normalized and normalized not in candidates:
                candidates.append(normalized)

        add_candidate(question)

        # 针对图谱中更常见的关系表达，补一些中文同义改写。
        synonym_variants = [
            question.replace("作者", "发布者"),
            question.replace("作者", "发布机构"),
            question.replace("谁写的", "由谁发布"),
            question.replace("是谁写的", "由谁发布"),
        ]
        for variant in synonym_variants:
            add_candidate(variant)

        if self._contains_cjk(question):
            rewritten = await self._rewrite_question_for_graph(question)
            add_candidate(rewritten)

        return candidates

    async def _rewrite_question_for_graph(self, question: str) -> str | None:
        system_prompt = (
            "你负责把用户问题改写成更适合知识图谱检索的简短英文问题。"
            "优先保留文档名、标准号、材料名、表格名等专有名词。"
            "如果用户问“作者”，但更合理的图谱关系是“发布者、发布机构、published by、publisher”，"
            "请改写成对应表达。"
            "只返回一行英文问题，不要解释。"
        )
        try:
            rewritten = await self.inference_service.generate(
                system_prompt,
                question,
                max_new_tokens=96,
                temperature=0.0,
            )
        except RuntimeError:
            return None

        normalized = rewritten.strip().strip('"').strip("'")
        return normalized or None

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _normalize_query_result(self, result: Any) -> str | None:
        if result is None:
            return None

        if isinstance(result, str):
            normalized = result.strip()
            if not normalized or normalized.lower() == "none":
                return None
            return normalized

        if isinstance(result, dict):
            for key in ("response", "answer", "result", "content"):
                value = result.get(key)
                normalized = self._normalize_query_result(value)
                if normalized is not None:
                    return normalized
            return None

        for attr in ("response", "answer", "result", "content"):
            if hasattr(result, attr):
                normalized = self._normalize_query_result(getattr(result, attr))
                if normalized is not None:
                    return normalized

        text = str(result).strip()
        if not text or text.lower() == "none":
            return None
        return text

    async def close(self) -> None:
        if self._rag is None:
            return

        finalize = getattr(self._rag, "finalize_storages", None)
        if callable(finalize):
            maybe_awaitable = finalize()
            if maybe_awaitable is not None and hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable
        self._rag = None

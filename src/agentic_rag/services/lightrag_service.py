from __future__ import annotations

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
                "LightRAG dependencies are not installed. Run `pip install -e .` first."
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
                history_text = "\n\nConversation history:\n" + "\n".join(history_lines)
            user_prompt = f"{history_text}\n\n{prompt}".strip()
            return await self.inference_service.generate(
                system_prompt or "You are a helpful assistant.",
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
            raise ValueError("Parsed PDF text is empty, nothing can be indexed.")

        payload = text if not file_path else f"[SOURCE: {file_path}]\n\n{text}"
        await rag.ainsert(payload)

        # We return a simple estimate because LightRAG does not expose chunk count consistently.
        return max(1, len(text) // 1200)

    async def query(self, question: str, mode: str | None = None) -> str:
        rag = await self.get_rag()
        query_param_cls = getattr(rag, "_query_param_cls")
        result = await rag.aquery(
            question,
            param=query_param_cls(mode=mode or self.settings.lightrag_query_mode),
        )
        answer = self._normalize_query_result(result)
        if answer is not None:
            return answer

        return (
            "LightRAG routed the request correctly, but the query stage did not produce a final answer. "
            "The knowledge graph appears to exist, but this question returned an empty result."
        )

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

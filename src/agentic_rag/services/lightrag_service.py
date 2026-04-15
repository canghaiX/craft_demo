from __future__ import annotations

from typing import Any

from agentic_rag.config import Settings


class LightRAGService:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required before using LightRAG.")

        self.settings = settings
        self._rag: Any | None = None

    async def _build_rag(self) -> Any:
        try:
            from lightrag import LightRAG, QueryParam
            from lightrag.llm.openai import openai_complete_if_cache, openai_embed
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
            return await openai_complete_if_cache(
                self.settings.lightrag_llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
                **kwargs,
            )

        embedding_func = EmbeddingFunc(
            embedding_dim=self.settings.lightrag_embed_dim,
            func=lambda texts: openai_embed(
                texts,
                model=self.settings.lightrag_embed_model,
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            ),
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
        return result if isinstance(result, str) else str(result)

    async def close(self) -> None:
        if self._rag is None:
            return

        finalize = getattr(self._rag, "finalize_storages", None)
        if callable(finalize):
            maybe_awaitable = finalize()
            if maybe_awaitable is not None and hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable
        self._rag = None

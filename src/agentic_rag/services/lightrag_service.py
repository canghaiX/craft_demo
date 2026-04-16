from __future__ import annotations

import asyncio
import hashlib
import math
import re
from functools import partial
from collections.abc import Iterable
from typing import Any
from uuid import uuid4

import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from agentic_rag.config import Settings
from agentic_rag.services.local_inference import LocalInferenceService

_LIGHTRAG_CONTEXTS: dict[str, tuple[Settings, LocalInferenceService]] = {}


def _get_registered_context(context_id: str) -> tuple[Settings, LocalInferenceService]:
    context = _LIGHTRAG_CONTEXTS.get(context_id)
    if context is None:
        raise RuntimeError("LightRAG 回调上下文不存在，服务可能已经关闭或尚未正确初始化。")
    return context


async def _lightrag_embedding_func(context_id: str, texts: Iterable[str]) -> np.ndarray:
    settings, inference_service = _get_registered_context(context_id)
    payload = [text if isinstance(text, str) and text.strip() else " " for text in texts]
    if not payload:
        return np.empty((0, settings.lightrag_embed_dim), dtype=np.float32)
    return await inference_service.embed(payload)


async def _lightrag_llm_func(
    context_id: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> str:
    _, inference_service = _get_registered_context(context_id)
    user_prompt = LightRAGService._merge_history_and_prompt(prompt, history_messages or [])
    max_new_tokens = int(
        kwargs.get("max_tokens")
        or kwargs.get("max_new_tokens")
        or kwargs.get("max_completion_tokens")
        or 1024
    )
    return await inference_service.generate_for_lightrag(
        system_prompt or "You are a helpful assistant.",
        user_prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0 if keyword_extraction else 0.1,
    )


class LightRAGService:
    # 正式的 LightRAG 适配层：
    # 1. 使用官方 LightRAG Core 维护图谱与索引
    # 2. 通过自定义 LLM / embedding 回调复用项目现有推理能力
    # 3. 保持 API 与 LangGraph 上层调用方式基本不变
    def __init__(self, settings: Settings, inference_service: LocalInferenceService) -> None:
        self.settings = settings
        self.inference_service = inference_service
        self._context_id = f"lightrag-{uuid4().hex}"
        _LIGHTRAG_CONTEXTS[self._context_id] = (settings, inference_service)
        self._rag: LightRAG | None = None
        self._init_lock = asyncio.Lock()
        self._closed = False

    async def ingest_document(self, text: str, file_path: str | None = None) -> int:
        if not text.strip():
            raise ValueError("解析后的 PDF 文本为空，无法建立知识库。")

        rag = await self._get_rag()
        source_file = self._normalize_source_file(file_path)
        doc_id = self._build_doc_id(text, source_file)

        # 对同一路径文档使用稳定 doc_id，这样重复导入时会走“替换更新”而不是无限堆积。
        existing_doc = await rag.aget_docs_by_ids([doc_id])
        if self._doc_exists(existing_doc, doc_id):
            await rag.adelete_by_doc_id(doc_id, delete_llm_cache=False)

        await rag.ainsert(text, ids=[doc_id], file_paths=[source_file])

        doc_status = await rag.aget_docs_by_ids([doc_id])
        return self._extract_chunks_count(doc_status, doc_id, text)

    async def query(self, question: str, mode: str | None = None) -> str:
        if not question.strip():
            raise ValueError("问题不能为空。")

        rag = await self._get_rag()
        attempted_modes: list[str] = []

        for query_text in await self._build_query_candidates(question):
            for candidate_mode in self._build_query_modes(mode):
                attempted_modes.append(f"{candidate_mode}:{query_text}")
                query_param = QueryParam(
                    mode=candidate_mode,
                    response_type=self.settings.lightrag_response_type,
                    top_k=self.settings.lightrag_top_k,
                    chunk_top_k=self.settings.lightrag_chunk_top_k,
                    include_references=self.settings.lightrag_include_references,
                )
                try:
                    result = await rag.aquery(query_text, param=query_param)
                except Exception:
                    continue

                normalized = self._normalize_query_result(result)
                if normalized is not None and not self._is_insufficient_answer(normalized):
                    return normalized

        tried = ", ".join(attempted_modes[:8])
        return (
            "LightRAG 已执行查询，但没有返回可用答案。"
            f" 已尝试的问题/模式组合：{tried or '无'}。"
            f" 请先确认 `LIGHTRAG_WORKING_DIR={self.settings.lightrag_working_dir}` 下确实已生成 LightRAG 图谱与索引文件，"
            "并优先尝试更短、更贴近文档原文实体名的英文问法。"
        )

    async def close(self) -> None:
        self._closed = True
        try:
            if self._rag is None:
                return
            finalize = getattr(self._rag, "finalize_storages", None)
            if callable(finalize):
                await finalize()
            self._rag = None
        finally:
            _LIGHTRAG_CONTEXTS.pop(self._context_id, None)

    async def _get_rag(self) -> LightRAG:
        if self._closed:
            raise RuntimeError("LightRAGService 已关闭，不能继续使用。")

        if self._rag is not None:
            return self._rag

        async with self._init_lock:
            if self._rag is not None:
                return self._rag

            self.settings.lightrag_working_dir.mkdir(parents=True, exist_ok=True)
            rag = LightRAG(
                working_dir=str(self.settings.lightrag_working_dir),
                llm_model_name=self.settings.lightrag_llm_model,
                llm_model_func=partial(_lightrag_llm_func, self._context_id),
                llm_model_max_async=self.settings.lightrag_llm_max_async,
                embedding_func=self._build_embedding_func(),
                embedding_func_max_async=self.settings.lightrag_embedding_max_async,
                chunk_token_size=self.settings.lightrag_chunk_token_size,
                chunk_overlap_token_size=self.settings.lightrag_chunk_overlap_token_size,
                enable_llm_cache=self.settings.lightrag_enable_llm_cache,
                tiktoken_model_name=self.settings.lightrag_tiktoken_model_name,
            )
            await rag.initialize_storages()
            self._rag = rag
            return rag

    def _build_embedding_func(self) -> EmbeddingFunc:
        return EmbeddingFunc(
            embedding_dim=self.settings.lightrag_embed_dim,
            max_token_size=self.settings.lightrag_embedding_max_tokens,
            model_name=self.settings.lightrag_embed_model,
            func=partial(_lightrag_embedding_func, self._context_id),
        )

    def _build_query_modes(self, requested_mode: str | None) -> list[str]:
        candidates = [
            requested_mode,
            self.settings.lightrag_query_mode,
            "mix",
            "hybrid",
            "local",
            "global",
            "naive",
        ]
        normalized: list[str] = []
        for candidate in candidates:
            if not candidate:
                continue
            mode = str(candidate).strip()
            if mode and mode not in normalized:
                normalized.append(mode)
        return normalized

    async def _build_query_candidates(self, question: str) -> list[str]:
        candidates: list[str] = []

        def add_candidate(text: str | None) -> None:
            if text is None:
                return
            normalized = re.sub(r"\s+", " ", text).strip()
            if normalized and normalized not in candidates:
                candidates.append(normalized)

        add_candidate(question)
        if self._contains_cjk(question):
            add_candidate(await self._rewrite_question_for_search(question))
        return candidates

    async def _rewrite_question_for_search(self, question: str) -> str | None:
        try:
            rewritten = await self.inference_service.generate_for_lightrag(
                "你负责把用户问题改写成更适合技术文档检索的简短英文问题。"
                "优先保留标准号、组织名、材料名、表格名和专有名词。"
                "只返回一行英文问题，不要解释。",
                question,
                max_new_tokens=96,
                temperature=0.0,
            )
        except Exception:
            return None

        normalized = rewritten.strip().strip('"').strip("'")
        return normalized or None

    @staticmethod
    def _merge_history_and_prompt(prompt: str, history_messages: list[dict[str, Any]]) -> str:
        if not history_messages:
            return prompt

        lines: list[str] = []
        for message in history_messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user")).strip() or "user"
            content = str(message.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        lines.append(f"user: {prompt}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_source_file(file_path: str | None) -> str:
        if file_path is None:
            return "unknown_source"
        normalized = re.sub(r"[\\/]+", "/", file_path).strip()
        return normalized or "unknown_source"

    @staticmethod
    def _build_doc_id(text: str, source_file: str) -> str:
        # 有来源路径时优先以路径生成稳定 ID，便于同一文档重复导入时覆盖更新。
        base = source_file if source_file != "unknown_source" else text
        digest = hashlib.md5(base.encode("utf-8")).hexdigest()
        return f"doc-{digest}"

    @staticmethod
    def _doc_exists(doc_payload: Any, doc_id: str) -> bool:
        if isinstance(doc_payload, dict):
            candidate = doc_payload.get(doc_id)
            if candidate is not None:
                return True
        return False

    def _extract_chunks_count(self, doc_payload: Any, doc_id: str, text: str) -> int:
        candidate = None
        if isinstance(doc_payload, dict):
            candidate = doc_payload.get(doc_id)

        if candidate is not None:
            chunk_count = self._extract_int(candidate, "chunks_count")
            if chunk_count is not None:
                return chunk_count

            chunks = self._extract_attr(candidate, "chunks")
            if isinstance(chunks, list) and chunks:
                return len(chunks)

        approx = math.ceil(len(text) / max(self.settings.lightrag_chunk_token_size, 1))
        return max(1, approx)

    def _normalize_query_result(self, result: Any) -> str | None:
        if result is None:
            return None

        answer = self._extract_answer_text(result)
        references = self._extract_references(result)

        if answer and references:
            return f"{answer}\n\n参考来源：\n{references}"
        return answer

    @staticmethod
    def _is_insufficient_answer(answer: str) -> bool:
        lowered = answer.lower()
        markers = [
            "没有返回可用答案",
            "没有足够信息",
            "未找到",
            "找不到",
            "insufficient",
            "not enough information",
            "no relevant",
            "not found",
            "i don't know",
        ]
        return any(marker in lowered for marker in markers)

    def _extract_answer_text(self, result: Any) -> str | None:
        if isinstance(result, str):
            normalized = result.strip()
            if normalized and normalized.lower() != "none":
                return normalized
            return None

        if isinstance(result, dict):
            for key in ("response", "answer", "result", "content"):
                value = result.get(key)
                normalized = self._extract_answer_text(value)
                if normalized is not None:
                    return normalized
            return None

        for attr in ("response", "answer", "result", "content"):
            value = self._extract_attr(result, attr)
            normalized = self._extract_answer_text(value)
            if normalized is not None:
                return normalized

        text = str(result).strip()
        if text and text.lower() != "none":
            return text
        return None

    def _extract_references(self, result: Any) -> str | None:
        refs = None
        if isinstance(result, dict):
            refs = result.get("references")
        else:
            refs = self._extract_attr(result, "references")

        if refs is None:
            return None

        if isinstance(refs, str):
            normalized = refs.strip()
            return normalized or None

        if isinstance(refs, list):
            lines: list[str] = []
            for item in refs:
                normalized = self._format_reference_item(item)
                if normalized:
                    lines.append(normalized)
            if lines:
                return "\n".join(lines[:8])
        return None

    def _format_reference_item(self, item: Any) -> str | None:
        if isinstance(item, str):
            normalized = item.strip()
            return f"- {normalized}" if normalized else None

        if isinstance(item, dict):
            source = str(item.get("file_path") or item.get("source") or item.get("source_file") or "").strip()
            chunk = str(item.get("chunk_id") or item.get("chunk_index") or "").strip()
            snippet = str(item.get("content") or item.get("text") or item.get("evidence") or "").strip()
        else:
            source = str(self._extract_attr(item, "file_path") or self._extract_attr(item, "source") or "").strip()
            chunk = str(self._extract_attr(item, "chunk_id") or self._extract_attr(item, "chunk_index") or "").strip()
            snippet = str(self._extract_attr(item, "content") or self._extract_attr(item, "text") or "").strip()

        parts = [part for part in [source, f"chunk={chunk}" if chunk else "", snippet[:160]] if part]
        if not parts:
            return None
        return f"- {' | '.join(parts)}"

    @staticmethod
    def _extract_attr(obj: Any, attr: str) -> Any:
        if hasattr(obj, attr):
            return getattr(obj, attr)
        return None

    @staticmethod
    def _extract_int(obj: Any, attr: str) -> int | None:
        value = LightRAGService._extract_attr(obj, attr)
        if isinstance(value, int):
            return value
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            return None
        return None

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in text)

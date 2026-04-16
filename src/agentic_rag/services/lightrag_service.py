from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agentic_rag.config import Settings
from agentic_rag.services.local_inference import LocalInferenceService


class LightRAGService:
    # 这个服务类负责两件事：
    # 1. 初始化并持有 LightRAG 实例
    # 2. 封装文档入库、知识图谱问答、标准三元组抽取等项目侧逻辑
    def __init__(self, settings: Settings, inference_service: LocalInferenceService) -> None:
        self.settings = settings
        self.inference_service = inference_service
        self._rag: Any | None = None

    async def _build_rag(self) -> Any:
        # 这里延迟导入 LightRAG，避免项目在未安装依赖时一启动就报错。
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
            # LightRAG 在抽取、总结、问答时会回调这里。
            # 我们把它内部传来的 prompt 转交给项目统一的推理服务。
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
            # 向量化入口同样转给项目统一的 embedding 服务。
            return await self.inference_service.embed(texts)

        # EmbeddingFunc 是 LightRAG 约定的封装类型。
        embedding_func = EmbeddingFunc(
            embedding_dim=self.settings.lightrag_embed_dim,
            func=embedding_func_impl,
        )

        # working_dir 指向图谱、向量、缓存等文件的落盘目录。
        rag = LightRAG(
            working_dir=str(self.settings.lightrag_working_dir),
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
        )
        await rag.initialize_storages()

        # 某些版本的 LightRAG 需要显式初始化 pipeline 状态，否则后续查询可能异常。
        initialize_pipeline = getattr(rag, "initialize_pipeline_status", None)
        if callable(initialize_pipeline):
            maybe_awaitable = initialize_pipeline()
            if maybe_awaitable is not None and hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable

        # 这里把 QueryParam 类挂到实例上，后面 query() 时可以直接取出来用。
        setattr(rag, "_query_param_cls", QueryParam)
        return rag

    async def get_rag(self) -> Any:
        # 懒加载：第一次真正用到时才构建 LightRAG 实例。
        if self._rag is None:
            self._rag = await self._build_rag()
        return self._rag

    async def ingest_document(self, text: str, file_path: str | None = None) -> int:
        # 文档入库流程：
        # 1. 交给 LightRAG 建图、建索引
        # 2. 额外抽取一份标准三元组 sidecar 文件，便于后续学习和分析
        rag = await self.get_rag()
        if not text.strip():
            raise ValueError("解析后的 PDF 文本为空，无法建立索引。")

        payload = text if not file_path else f"[SOURCE: {file_path}]\n\n{text}"
        await rag.ainsert(payload)
        await self._extract_and_store_standard_triples(text, file_path=file_path)

        # LightRAG 不同版本不一定稳定暴露 chunk 数，所以这里返回一个估算值。
        return max(1, len(text) // 1200)

    async def query(self, question: str, mode: str | None = None) -> str:
        # 问答流程：
        # 1. 生成多个候选问法
        # 2. 逐个调用 LightRAG 查询
        # 3. 只要有一个返回有效答案就立即返回
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
        # 这个方法的目标是提高命中率：
        # 原问题先保留，再补一些同义改写，最后必要时再生成英文检索问法。
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
        # 当前图谱内容大多是英文实体和关系。
        # 当用户用中文提问时，这里让模型先把问题改写成更适合图谱检索的英文问法。
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
        # 用最简单的方式判断文本里是否包含中日韩统一表意文字。
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    async def _extract_and_store_standard_triples(
        self,
        text: str,
        *,
        file_path: str | None = None,
    ) -> None:
        # 这是新增的“标准三元组 sidecar”流程：
        # 不改 LightRAG 内部原始抽取逻辑，而是在项目侧额外做一份标准 SPO 抽取。
        triples: list[dict[str, Any]] = []
        for chunk_index, chunk_text in enumerate(self._chunk_text_for_triples(text), start=1):
            prompt = self._build_triple_extraction_prompt(
                chunk_text,
                file_path=file_path,
                chunk_index=chunk_index,
            )
            response = await self.inference_service.generate(
                "你是一名知识图谱构建专家。请从文本中抽取标准谓语三元组。",
                prompt,
                max_new_tokens=1200,
                temperature=0.0,
            )
            triples.extend(
                self._parse_triples_response(
                    response,
                    file_path=file_path,
                    chunk_index=chunk_index,
                )
            )

        if triples:
            self._write_standard_triples(triples)

    def _chunk_text_for_triples(self, text: str, chunk_size: int = 2400, overlap: int = 200) -> list[str]:
        # 把长文本切成可控大小，避免一次 prompt 过长。
        # overlap 用来减少切块边界处的信息丢失。
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(len(normalized), start + chunk_size)
            chunks.append(normalized[start:end])
            if end >= len(normalized):
                break
            start = max(0, end - overlap)
        return chunks

    def _build_triple_extraction_prompt(
        self,
        chunk_text: str,
        *,
        file_path: str | None,
        chunk_index: int,
    ) -> str:
        # 这里定义“标准三元组抽取”的提示词。
        # 要求模型直接输出 JSON，且 predicate 用 snake_case。
        source_name = file_path or "unknown_source"
        return (
            "请从下面的文档片段中抽取标准谓语三元组。\n"
            "输出要求：\n"
            "1. 只返回 JSON 数组。\n"
            "2. 每个元素必须包含 `subject`、`predicate`、`object`、`evidence` 四个字段。\n"
            "3. `predicate` 必须使用英文小写 snake_case，例如 `published_by`、`references`、`part_of`。\n"
            "4. 只抽取文本中明确出现或可直接判断的关系，不要臆造。\n"
            "5. 专有名词尽量保留原文写法。\n"
            "6. 如果没有可抽取的三元组，返回空数组 `[]`。\n\n"
            f"来源文件：{source_name}\n"
            f"片段编号：{chunk_index}\n\n"
            "文档片段：\n"
            f"{chunk_text}"
        )

    def _parse_triples_response(
        self,
        response: str,
        *,
        file_path: str | None,
        chunk_index: int,
    ) -> list[dict[str, Any]]:
        # 模型输出是字符串，这里负责把 JSON 数组解析成 Python 字典列表。
        # 如果模型返回格式不合法，就安全地返回空列表。
        match = re.search(r"\[\s*.*\s*\]", response, re.DOTALL)
        if not match:
            return []

        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

        if not isinstance(payload, list):
            return []

        triples: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue

            subject = str(item.get("subject", "")).strip()
            predicate = self._normalize_predicate(str(item.get("predicate", "")).strip())
            obj = str(item.get("object", "")).strip()
            evidence = str(item.get("evidence", "")).strip()

            if not subject or not predicate or not obj:
                continue

            triples.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "evidence": evidence,
                    "source_file": file_path or "unknown_source",
                    "chunk_index": chunk_index,
                }
            )
        return triples

    def _normalize_predicate(self, predicate: str) -> str:
        # 把谓语统一成标准英文 snake_case，便于后续检索和统计。
        normalized = predicate.strip().lower()
        normalized = normalized.replace("-", "_").replace(" ", "_")
        normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized

    def _write_standard_triples(self, triples: list[dict[str, Any]]) -> None:
        # 把新抽到的三元组和已有文件合并、去重，再落盘。
        output_path = Path(self.settings.lightrag_working_dir) / "standard_triples.json"
        existing: list[dict[str, Any]] = []
        if output_path.exists():
            try:
                existing_payload = json.loads(output_path.read_text(encoding="utf-8"))
                if isinstance(existing_payload, list):
                    existing = [item for item in existing_payload if isinstance(item, dict)]
            except json.JSONDecodeError:
                existing = []

        merged = existing + triples
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str, str, int]] = set()
        for item in merged:
            key = (
                str(item.get("subject", "")),
                str(item.get("predicate", "")),
                str(item.get("object", "")),
                str(item.get("evidence", "")),
                str(item.get("source_file", "")),
                int(item.get("chunk_index", 0)),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        output_path.write_text(
            json.dumps(deduped, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _normalize_query_result(self, result: Any) -> str | None:
        # LightRAG 不同版本返回结构不完全一致。
        # 这里统一把结果整理成字符串答案；如果没有有效内容则返回 None。
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
        # 关闭并释放 LightRAG 持有的底层存储资源。
        if self._rag is None:
            return

        finalize = getattr(self._rag, "finalize_storages", None)
        if callable(finalize):
            maybe_awaitable = finalize()
            if maybe_awaitable is not None and hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable
        self._rag = None

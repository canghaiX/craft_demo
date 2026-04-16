from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agentic_rag.config import Settings
from agentic_rag.services.local_inference import LocalInferenceService


class LightRAGService:
    # 这里继续保留 LightRAGService 这个类名，原因是项目其他模块已经依赖了它。
    # 但当前版本的核心实现已经从“依赖 LightRAG 原生查询”改成：
    #
    # 1. PDF 文本 -> 分块 -> 直接抽标准三元组 -> 落盘
    # 2. PDF 文本 -> 分块 -> 原文 chunk 落盘
    # 3. 问答时同时检索：
    #    - 标准三元组
    #    - 原文 chunk
    # 4. 把两类证据一起喂给模型生成答案
    #
    # 也就是说，这份代码现在实现的是“混合问答模式”：
    # 三元组检索 + 原文 chunk 检索
    def __init__(self, settings: Settings, inference_service: LocalInferenceService) -> None:
        self.settings = settings
        self.inference_service = inference_service

        # 标准三元组存储文件。
        self._triples_file = Path(self.settings.lightrag_working_dir) / "standard_triples.json"
        # 原文 chunk 存储文件。
        self._chunks_file = Path(self.settings.lightrag_working_dir) / "source_chunks.json"

    async def ingest_document(self, text: str, file_path: str | None = None) -> int:
        # 这是当前主流程的入口：
        # PDF 被 parse 成大文本后，会调用这个函数完成“入库”。
        #
        # 这一步做两件事：
        # 1. 把原文切块并落盘，作为后续原文检索的依据
        # 2. 基于每个文本块直接抽标准三元组，再落盘
        #
        # 所以现在项目中的“知识库存储”不再只有一个文件，
        # 而是至少包含：
        # - source_chunks.json
        # - standard_triples.json
        if not text.strip():
            raise ValueError("解析后的 PDF 文本为空，无法建立知识库。")

        chunk_records = self._build_chunk_records(text, file_path=file_path)
        self._write_source_chunks(chunk_records)

        all_triples: list[dict[str, Any]] = []
        for chunk in chunk_records:
            prompt = self._build_triple_extraction_prompt(
                chunk_text=chunk["content"],
                file_path=chunk["source_file"],
                chunk_index=int(chunk["chunk_index"]),
            )
            response = await self.inference_service.generate(
                "你是一名知识图谱构建专家，请把输入文本直接抽取为标准谓语三元组。",
                prompt,
                max_new_tokens=1400,
                temperature=0.0,
            )
            parsed_triples = self._parse_triples_response(
                response,
                file_path=chunk["source_file"],
                chunk_index=int(chunk["chunk_index"]),
            )
            all_triples.extend(parsed_triples)

        normalized_triples = self._normalize_triples(all_triples)
        self._write_standard_triples(normalized_triples)

        # 返回 chunk 数，便于上层感知这次处理了多少块。
        return len(chunk_records)

    async def query(self, question: str, mode: str | None = None) -> str:
        # 当前问答的完整流程是：
        #
        # 1. 读取标准三元组
        # 2. 读取原文 chunk
        # 3. 根据问题构造多个候选问法
        # 4. 对每个候选问法同时检索：
        #    - 三元组证据
        #    - 原文证据
        # 5. 把两类证据一起交给模型总结答案
        #
        # 这就是你要求的“混合问答模式”。
        triples = self._load_standard_triples()
        chunks = self._load_source_chunks()

        if not triples and not chunks:
            return "当前还没有可用的三元组或原文块，请先重新导入 PDF。"

        for candidate in await self._build_query_candidates(question):
            relevant_triples = self._search_relevant_triples(candidate, triples)
            relevant_chunks = self._search_relevant_chunks(candidate, chunks)

            # 如果这轮候选问法什么都没命中，就继续试下一个改写版本。
            if not relevant_triples and not relevant_chunks:
                continue

            answer = await self._answer_with_evidence(
                question=question,
                query_candidate=candidate,
                triples=relevant_triples,
                chunks=relevant_chunks,
            )
            normalized = self._normalize_query_result(answer)
            if normalized is not None:
                return normalized

        return (
            "已经走到混合问答流程，但没有检索到足够相关的三元组或原文片段。"
            "建议换一种更贴近实体名、表格名、标准号或关系表达的问法再试一次。"
        )

    def _build_chunk_records(self, text: str, file_path: str | None = None) -> list[dict[str, Any]]:
        # 这个函数负责把整篇文本变成“带元信息的 chunk 记录”。
        # 纯字符串列表不够用，因为后续检索时我们还需要知道：
        # - 来自哪个文件
        # - 是第几个 chunk
        # - chunk 内容是什么
        #
        # 所以这里输出的是字典列表，而不是单纯的字符串列表。
        chunk_texts = self._chunk_text(text)
        chunk_records: list[dict[str, Any]] = []
        for chunk_index, chunk_text in enumerate(chunk_texts, start=1):
            chunk_records.append(
                {
                    "chunk_index": chunk_index,
                    "source_file": file_path or "unknown_source",
                    "content": chunk_text,
                }
            )
        return chunk_records

    def _chunk_text(self, text: str, chunk_size: int = 2400, overlap: int = 300) -> list[str]:
        # 这里负责“分块”。
        #
        # 为什么要分块：
        # 1. 整篇 PDF 太长，不能直接一次送给模型
        # 2. 原文检索时，chunk 是最基本的证据单元
        # 3. 三元组抽取时，chunk 也是最基本的抽取单元
        #
        # overlap=300 表示相邻块之间保留 300 个字符重叠，
        # 这样可以减少信息刚好落在块边界时被切断的问题。
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
        *,
        chunk_text: str,
        file_path: str | None,
        chunk_index: int,
    ) -> str:
        # 这一步定义“模型如何从一个文本块中抽标准三元组”。
        # 我们明确要求：
        # - 只返回 JSON 数组
        # - 每个元素必须有 subject/predicate/object/evidence
        # - predicate 必须是 snake_case
        source_name = file_path or "unknown_source"
        return (
            "请把下面的文档片段直接抽取为标准谓语三元组。\n"
            "请严格遵守以下规则：\n"
            "1. 只返回 JSON 数组，不要返回任何解释。\n"
            "2. 数组中的每个元素必须包含 `subject`、`predicate`、`object`、`evidence` 四个字段。\n"
            "3. `predicate` 必须使用英文小写 snake_case，例如 `published_by`、`references`、`part_of`。\n"
            "4. 只抽取文本中明确出现或可直接判断的关系，不要臆造。\n"
            "5. 如果没有可抽取的三元组，请返回 `[]`。\n"
            "6. `evidence` 尽量保留能支持该三元组的原始短句。\n\n"
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
        # 模型返回的是普通字符串，这里需要从字符串中提取 JSON 数组。
        # 如果模型格式不稳定，这里宁可返回空列表，也不要让整个流程报错。
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
        # 这个函数专门负责谓语归一化。
        # 典型目标是把不同写法统一成稳定形式，例如：
        # - "Published By"
        # - "published by"
        # - "published-by"
        # 最终都变成：
        # - "published_by"
        normalized = predicate.strip().lower()
        normalized = normalized.replace("-", "_").replace(" ", "_")
        normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized

    def _normalize_triples(self, triples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # 这一步是“清洗 / 归一化”阶段。
        # 要做的事情包括：
        # 1. 去掉字段为空的三元组
        # 2. 压缩多余空白
        # 3. 在内存中先做一轮去重
        cleaned: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str, str, int]] = set()

        for item in triples:
            subject = re.sub(r"\s+", " ", str(item.get("subject", "")).strip())
            predicate = self._normalize_predicate(str(item.get("predicate", "")).strip())
            obj = re.sub(r"\s+", " ", str(item.get("object", "")).strip())
            evidence = re.sub(r"\s+", " ", str(item.get("evidence", "")).strip())
            source_file = str(item.get("source_file", "unknown_source")).strip()
            chunk_index = int(item.get("chunk_index", 0))

            if not subject or not predicate or not obj:
                continue

            normalized_item = {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "evidence": evidence,
                "source_file": source_file,
                "chunk_index": chunk_index,
            }
            key = (
                normalized_item["subject"],
                normalized_item["predicate"],
                normalized_item["object"],
                normalized_item["evidence"],
                normalized_item["source_file"],
                normalized_item["chunk_index"],
            )
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(normalized_item)

        return cleaned

    def _write_standard_triples(self, triples: list[dict[str, Any]]) -> None:
        # 把标准三元组写到本地文件中。
        # 这里采用“旧文件 + 新结果 合并后再去重重写”的策略。
        output_path = self._triples_file
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

    def _write_source_chunks(self, chunks: list[dict[str, Any]]) -> None:
        # 原文 chunk 的写盘逻辑和标准三元组类似：
        # 也是先合并旧文件，再去重，再重写。
        output_path = self._chunks_file
        existing: list[dict[str, Any]] = []
        if output_path.exists():
            try:
                existing_payload = json.loads(output_path.read_text(encoding="utf-8"))
                if isinstance(existing_payload, list):
                    existing = [item for item in existing_payload if isinstance(item, dict)]
            except json.JSONDecodeError:
                existing = []

        merged = existing + chunks
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, int, str]] = set()
        for item in merged:
            key = (
                str(item.get("source_file", "")),
                int(item.get("chunk_index", 0)),
                str(item.get("content", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        output_path.write_text(
            json.dumps(deduped, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_standard_triples(self) -> list[dict[str, Any]]:
        # 问答时读取标准三元组文件。
        if not self._triples_file.exists():
            return []
        try:
            payload = json.loads(self._triples_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def _load_source_chunks(self) -> list[dict[str, Any]]:
        # 问答时读取原文 chunk 文件。
        if not self._chunks_file.exists():
            return []
        try:
            payload = json.loads(self._chunks_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    async def _build_query_candidates(self, question: str) -> list[str]:
        # 这里负责生成多个“候选检索问法”。
        # 原因是：
        # 同一个问题换一种表达，可能更容易命中三元组和 chunk。
        candidates: list[str] = []

        def add_candidate(text: str | None) -> None:
            if text is None:
                return
            normalized = re.sub(r"\s+", " ", text).strip()
            if normalized and normalized not in candidates:
                candidates.append(normalized)

        add_candidate(question)
        add_candidate(question.replace("作者", "发布者"))
        add_candidate(question.replace("作者", "发布机构"))
        add_candidate(question.replace("谁写的", "由谁发布"))
        add_candidate(question.replace("是谁写的", "由谁发布"))

        if self._contains_cjk(question):
            add_candidate(await self._rewrite_question_for_search(question))

        return candidates

    async def _rewrite_question_for_search(self, question: str) -> str | None:
        # 这一步主要针对中文问题。
        # 因为很多抽出的 subject/predicate/object 依然更偏英文或技术名词形式，
        # 所以先让模型把问题改成更利于检索的英文表达。
        system_prompt = (
            "你负责把用户问题改写成更适合知识库检索的简短英文问题。"
            "优先保留标准号、材料名、组织名、表格名等专有名词。"
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

    def _search_relevant_triples(
        self,
        query_text: str,
        triples: list[dict[str, Any]],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        # 这里做的是一个轻量级三元组检索器。
        # 原理很朴素：
        # - 把问题切成 token
        # - token 命中 subject/object/predicate/evidence 就加分
        # - 按分数排序取前几条
        tokens = self._tokenize_for_search(query_text)
        if not tokens:
            return []

        scored: list[tuple[int, dict[str, Any]]] = []
        for item in triples:
            subject = str(item.get("subject", "")).lower()
            predicate = str(item.get("predicate", "")).lower()
            obj = str(item.get("object", "")).lower()
            evidence = str(item.get("evidence", "")).lower()

            score = 0
            for token in tokens:
                if token in subject:
                    score += 4
                if token in obj:
                    score += 4
                if token in predicate:
                    score += 3
                if token in evidence:
                    score += 1

            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def _search_relevant_chunks(
        self,
        query_text: str,
        chunks: list[dict[str, Any]],
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        # 这里是原文 chunk 检索器。
        # 和三元组检索相比，它更偏向保留上下文细节：
        # - 问题中的 token 命中 chunk 内容越多，分数越高
        # - 最终取前几个最相关的 chunk 作为原文证据
        tokens = self._tokenize_for_search(query_text)
        if not tokens:
            return []

        scored: list[tuple[int, dict[str, Any]]] = []
        for item in chunks:
            content = str(item.get("content", "")).lower()
            score = 0
            for token in tokens:
                if token in content:
                    score += 2

            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]

    async def _answer_with_evidence(
        self,
        *,
        question: str,
        query_candidate: str,
        triples: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
    ) -> str:
        # 这是“混合问答模式”的最后一步。
        # 我们会把两类证据一起送给模型：
        #
        # 1. 标准三元组：提供关系骨架
        # 2. 原文 chunk：提供上下文和细节
        #
        # 然后让模型基于它们生成最终回答。
        triple_block = json.dumps(triples, ensure_ascii=False, indent=2)
        chunk_block = json.dumps(chunks, ensure_ascii=False, indent=2)

        user_prompt = (
            f"用户问题：{question}\n"
            f"检索问法：{query_candidate}\n\n"
            "下面给出两类证据，请严格基于这些证据回答：\n"
            "1. 标准三元组证据：用于说明实体关系\n"
            "2. 原文 chunk 证据：用于补充上下文、细节、表格或条件说明\n\n"
            "如果证据不足，请明确说明“现有证据不足”。\n\n"
            "【标准三元组证据】\n"
            f"{triple_block}\n\n"
            "【原文 chunk 证据】\n"
            f"{chunk_block}"
        )

        return await self.inference_service.generate(
            "你是一名严谨的知识问答助手，请综合三元组证据和原文证据回答问题，不要脱离证据自由发挥。",
            user_prompt,
            max_new_tokens=700,
            temperature=0.1,
        )

    def _tokenize_for_search(self, text: str) -> list[str]:
        # 这里做的是非常朴素的 token 提取。
        # 优势是简单稳定，对技术文档里的标准号、材料号、组织名比较友好。
        lowered = text.lower()
        tokens = re.findall(r"[a-z0-9_.\-]+", lowered)
        unique_tokens: list[str] = []
        for token in tokens:
            token = token.strip("._-")
            if len(token) < 2:
                continue
            if token not in unique_tokens:
                unique_tokens.append(token)
        return unique_tokens

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        # 判断文本中是否包含中文字符。
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _normalize_query_result(self, result: Any) -> str | None:
        # 不同上游可能返回 str / dict / 其他对象，这里统一做结果整理。
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
        # 当前版本没有持有需要显式释放的 LightRAG 底层资源，因此保留空实现。
        return

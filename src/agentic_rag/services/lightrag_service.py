from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agentic_rag.config import Settings
from agentic_rag.services.local_inference import LocalInferenceService


class LightRAGService:
    # 这里保留原来的类名，是为了尽量少改项目其他模块的调用方式。
    # 但当前版本的核心职责已经调整为：
    # 1. 把 PDF 文本直接抽取成标准三元组
    # 2. 把标准三元组落盘到本地 JSON 文件
    # 3. 问答时从标准三元组中检索相关关系，再交给模型生成答案
    def __init__(self, settings: Settings, inference_service: LocalInferenceService) -> None:
        self.settings = settings
        self.inference_service = inference_service
        self._triples_file = Path(self.settings.lightrag_working_dir) / "standard_triples.json"

    async def ingest_document(self, text: str, file_path: str | None = None) -> int:
        # 当前主流程完全按照“PDF -> 文本 -> 分块 -> 模型抽三元组 -> 清洗 -> 落盘”执行。
        # 这里不再调用 LightRAG 的 ainsert() 建原生图谱，而是把标准三元组当成主产物。
        if not text.strip():
            raise ValueError("解析后的 PDF 文本为空，无法抽取标准三元组。")

        chunk_texts = self._chunk_text_for_triples(text)
        all_triples: list[dict[str, Any]] = []

        # 逐块抽取的原因是：
        # 1. 整篇 PDF 往往太长，无法一次性安全送给模型
        # 2. 分块后更容易保留局部上下文
        # 3. 后续可以追踪每条三元组来自哪个 chunk
        for chunk_index, chunk_text in enumerate(chunk_texts, start=1):
            prompt = self._build_triple_extraction_prompt(
                chunk_text,
                file_path=file_path,
                chunk_index=chunk_index,
            )
            response = await self.inference_service.generate(
                "你是一名知识图谱构建专家，请把输入文本直接抽取为标准谓语三元组。",
                prompt,
                max_new_tokens=1400,
                temperature=0.0,
            )
            parsed_triples = self._parse_triples_response(
                response,
                file_path=file_path,
                chunk_index=chunk_index,
            )
            all_triples.extend(parsed_triples)

        normalized_triples = self._normalize_triples(all_triples)
        self._write_standard_triples(normalized_triples)

        # 这里返回“切分后的 chunk 数”，因为现在项目的核心处理单元就是文本块。
        return len(chunk_texts)

    async def query(self, question: str, mode: str | None = None) -> str:
        # 问答流程调整为：
        # 1. 从标准三元组文件中读取已抽取好的三元组
        # 2. 根据问题生成多个候选检索问法
        # 3. 在三元组中做简单的相关性检索
        # 4. 把检索到的三元组作为证据，交给模型生成自然语言回答
        triples = self._load_standard_triples()
        if not triples:
            return "当前还没有可用的标准三元组，请先重新导入 PDF 并完成抽取。"

        for candidate in await self._build_query_candidates(question):
            relevant_triples = self._search_relevant_triples(candidate, triples)
            if not relevant_triples:
                continue

            answer = await self._answer_with_triples(
                question=question,
                query_candidate=candidate,
                triples=relevant_triples,
            )
            normalized = self._normalize_query_result(answer)
            if normalized is not None:
                return normalized

        return (
            "已经命中了标准三元组问答流程，但没有检索到足够相关的三元组来回答这个问题。"
            "建议换一种更贴近实体、关系或标准名称的问法再试一次。"
        )

    def _chunk_text_for_triples(
        self,
        text: str,
        chunk_size: int = 2400,
        overlap: int = 200,
    ) -> list[str]:
        # 这一步负责“分块”。
        # 处理逻辑很简单：
        # 1. 先把连续空白压成单个空格，减少模型看到的噪声
        # 2. 再按固定窗口切块
        # 3. 每块之间保留少量重叠，降低边界处信息被切断的风险
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
        # 这一步负责“告诉模型该怎么抽三元组”。
        # 这里要求模型直接输出标准 JSON 数组，而不是先输出 entity/relation 再后处理。
        source_name = file_path or "unknown_source"
        return (
            "请把下面的文档片段直接抽取为标准谓语三元组。\n"
            "请严格遵守以下规则：\n"
            "1. 只返回 JSON 数组，不要返回解释文字。\n"
            "2. 数组中的每个元素都必须是对象，且包含 `subject`、`predicate`、`object`、`evidence` 四个字段。\n"
            "3. `predicate` 必须使用英文小写 snake_case，例如 `published_by`、`references`、`part_of`。\n"
            "4. 只抽取文本中明确出现或可直接判断的关系，不要臆造。\n"
            "5. 如果没有可以抽取的三元组，请返回空数组 `[]`。\n"
            "6. `evidence` 请尽量保留能支持这条三元组的原始短句。\n\n"
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
        # 模型返回的是普通字符串，这里需要把其中的 JSON 数组提取出来并转回 Python 对象。
        # 如果模型没有严格按要求输出合法 JSON，就直接返回空列表，避免中断整个入库流程。
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
        # 规范化谓语的目标是把模型输出的各种写法收敛为统一形式。
        # 例如：
        # "Published By" -> "published_by"
        # "published-by" -> "published_by"
        normalized = predicate.strip().lower()
        normalized = normalized.replace("-", "_").replace(" ", "_")
        normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized

    def _normalize_triples(self, triples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # 这一步负责“清洗 / 归一化”：
        # 1. 再次过滤不完整字段
        # 2. 统一字符串空白
        # 3. 在内存中先做一次去重
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
        # 把新抽到的三元组和已有文件合并、去重，再落盘。
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

    def _load_standard_triples(self) -> list[dict[str, Any]]:
        # 问答时统一从本地标准三元组文件读取。
        if not self._triples_file.exists():
            return []
        try:
            payload = json.loads(self._triples_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    async def _build_query_candidates(self, question: str) -> list[str]:
        # 这个方法的目标是提高问答命中率：
        # 1. 原问题直接保留
        # 2. 对中文里常见但不利于检索的表达做一点点改写
        # 3. 如果问题是中文，再额外生成英文检索问法
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
            add_candidate(await self._rewrite_question_for_graph(question))

        return candidates

    async def _rewrite_question_for_graph(self, question: str) -> str | None:
        # 标准三元组中的 subject / predicate / object 目前主要是英文形式。
        # 所以当用户用中文提问时，先把问题改成更利于英文三元组检索的写法。
        system_prompt = (
            "你负责把用户问题改写成更适合标准三元组检索的简短英文问题。"
            "优先保留标准号、材料名、表格名、组织名等专有名词。"
            "如果用户问“作者”，但更合理的关系是“发布者、发布机构、published_by”，"
            "请改写成更贴近该关系的英文问法。"
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
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        # 这里做的是一个轻量级的本地检索，而不是复杂向量检索。
        # 评分思路：
        # - 查询里的每个 token，如果命中 subject/object/predicate/evidence，就累加分数
        # - subject/object 命中的权重更高，因为它们通常最关键
        tokens = self._tokenize_for_search(query_text)
        if not tokens:
            return []

        scored: list[tuple[int, dict[str, Any]]] = []
        for item in triples:
            haystack_subject = str(item.get("subject", "")).lower()
            haystack_predicate = str(item.get("predicate", "")).lower()
            haystack_object = str(item.get("object", "")).lower()
            haystack_evidence = str(item.get("evidence", "")).lower()

            score = 0
            for token in tokens:
                if token in haystack_subject:
                    score += 4
                if token in haystack_object:
                    score += 4
                if token in haystack_predicate:
                    score += 3
                if token in haystack_evidence:
                    score += 1

            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]

    async def _answer_with_triples(
        self,
        *,
        question: str,
        query_candidate: str,
        triples: list[dict[str, Any]],
    ) -> str:
        # 这一步把“检索到的标准三元组”作为外部证据，交给模型总结成自然语言回答。
        # 这样就形成了：
        # 问题 -> 三元组检索 -> 模型基于检索结果回答
        evidence_block = json.dumps(triples, ensure_ascii=False, indent=2)
        user_prompt = (
            f"用户问题：{question}\n"
            f"检索问法：{query_candidate}\n\n"
            "下面是从标准三元组知识库中检索到的候选证据，请严格基于这些证据回答。\n"
            "如果证据不足，请明确说明“现有三元组证据不足”。\n\n"
            f"{evidence_block}"
        )
        return await self.inference_service.generate(
            "你是一名严谨的知识图谱问答助手，请只依据给定三元组证据回答问题。",
            user_prompt,
            max_new_tokens=512,
            temperature=0.1,
        )

    def _tokenize_for_search(self, text: str) -> list[str]:
        # 这里做非常朴素的分词，目的是给本地检索提供可用 token。
        # 对英文和数字类实体比较有效，例如：
        # ASME、B16.5-2013、Graphite、425°C
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
        # 判断问题里是否包含中文字符。
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _normalize_query_result(self, result: Any) -> str | None:
        # 不同上游函数返回的可能是字符串，也可能是其他对象。
        # 这里统一把结果压成一个可直接展示的字符串答案。
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
        # 当前版本不再持有 LightRAG 的底层资源，因此这里保留空实现即可。
        return

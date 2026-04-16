from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agentic_rag.config import Settings
from agentic_rag.services.local_inference import LocalInferenceService


class StandardTripleExtractor:
    # 独立的标准三元组层：
    # 1. 基于清洗后的 chunk 抽取 subject/predicate/object/evidence
    # 2. 将谓语统一为 snake_case
    # 3. 同时落盘 source chunks，方便后续做证据补充
    def __init__(self, settings: Settings, inference_service: LocalInferenceService) -> None:
        self.settings = settings
        self.inference_service = inference_service
        self._triples_file = Path(self.settings.lightrag_working_dir) / "standard_triples.json"
        self._chunks_file = Path(self.settings.lightrag_working_dir) / "source_chunks.json"

    async def extract_and_store(self, text: str, file_path: str | None = None) -> int:
        if not text.strip():
            return 0

        chunk_records = self._build_chunk_records(text, file_path=file_path)
        all_triples: list[dict[str, Any]] = []

        for chunk in chunk_records:
            prompt = self._build_triple_extraction_prompt(
                chunk_text=chunk["content"],
                file_path=chunk["source_file"],
                chunk_index=int(chunk["chunk_index"]),
            )
            response = await self.inference_service.generate_for_lightrag(
                "你是一名知识图谱构建专家，请从技术文档片段中抽取标准三元组。",
                prompt,
                max_new_tokens=1200,
                temperature=0.0,
            )
            parsed_triples = self._parse_triples_response(
                response,
                file_path=chunk["source_file"],
                chunk_index=int(chunk["chunk_index"]),
            )
            all_triples.extend(parsed_triples)

        self._write_source_chunks(chunk_records, file_path=file_path)
        self._write_standard_triples(self._normalize_triples(all_triples), file_path=file_path)
        return len(chunk_records)

    def search_triples(self, query_text: str, limit: int = 10) -> list[dict[str, Any]]:
        triples = self._load_json_list(self._triples_file)
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
                    score += 5
                if token in obj:
                    score += 5
                if token in predicate:
                    score += 3
                if token in evidence:
                    score += 1
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def search_chunks(self, query_text: str, limit: int = 6) -> list[dict[str, Any]]:
        chunks = self._load_json_list(self._chunks_file)
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

    def _build_chunk_records(self, text: str, file_path: str | None = None) -> list[dict[str, Any]]:
        chunk_texts = self._chunk_text(text)
        source_file = file_path or "unknown_source"
        return [
            {
                "chunk_index": index,
                "source_file": source_file,
                "content": chunk_text,
            }
            for index, chunk_text in enumerate(chunk_texts, start=1)
        ]

    def _chunk_text(self, text: str, chunk_size: int = 2400, overlap: int = 300) -> list[str]:
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
        source_name = file_path or "unknown_source"
        return (
            "请把下面的技术文档片段抽取成标准谓语三元组。\n"
            "请严格遵守以下规则：\n"
            "1. 只返回 JSON 数组，不要返回任何解释。\n"
            "2. 每个元素必须包含 subject、predicate、object、evidence 四个字段。\n"
            "3. predicate 必须使用英文小写 snake_case，例如 published_by、references、defines、part_of。\n"
            "4. 只保留文本中明确表达的关系，不要臆造。\n"
            "5. 如果该片段没有明确关系，请返回 []。\n"
            "6. evidence 尽量保留支持该关系的原文短句。\n\n"
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

            subject = re.sub(r"\s+", " ", str(item.get("subject", "")).strip())
            predicate = self._normalize_predicate(str(item.get("predicate", "")).strip())
            obj = re.sub(r"\s+", " ", str(item.get("object", "")).strip())
            evidence = re.sub(r"\s+", " ", str(item.get("evidence", "")).strip())
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

    def _normalize_triples(self, triples: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

    def _write_standard_triples(self, triples: list[dict[str, Any]], file_path: str | None = None) -> None:
        existing = self._load_json_list(self._triples_file)
        source_file = file_path or "unknown_source"
        preserved = [item for item in existing if str(item.get("source_file")) != source_file]
        payload = self._normalize_triples(preserved + triples)
        self._triples_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_source_chunks(self, chunks: list[dict[str, Any]], file_path: str | None = None) -> None:
        existing = self._load_json_list(self._chunks_file)
        source_file = file_path or "unknown_source"
        preserved = [item for item in existing if str(item.get("source_file")) != source_file]

        merged = preserved + chunks
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

        self._chunks_file.write_text(json.dumps(deduped, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _normalize_predicate(predicate: str) -> str:
        normalized = predicate.strip().lower()
        normalized = normalized.replace("-", "_").replace(" ", "_")
        normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized

    @staticmethod
    def _load_json_list(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    @staticmethod
    def _tokenize_for_search(text: str) -> list[str]:
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

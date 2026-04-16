from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class PdfCleanResult:
    # 保存清洗后的文本和基础统计，便于后续排查规则效果。
    text: str
    removed_lines: int
    removed_pages: int


class PdfTextCleaner:
    # 面向标准类 PDF 的轻量清洗器：
    # 1. 去掉版权/授权/空白页等噪声
    # 2. 压缩重复空白
    # 3. 保留页码标记，方便后续回溯
    _NOISE_PATTERNS = [
        re.compile(r"^copyright\s+asme\s+international$", re.IGNORECASE),
        re.compile(r"^provided by ihs under license with asme$", re.IGNORECASE),
        re.compile(r"^not for resale$", re.IGNORECASE),
        re.compile(r"^no reproduction or networking permitted.*$", re.IGNORECASE),
        re.compile(r"^--[`',\-\s]+$", re.IGNORECASE),
        re.compile(r"^intentionally\s+left\s+blank$", re.IGNORECASE),
        re.compile(r"^this page is intentionally left blank$", re.IGNORECASE),
        re.compile(r"^asme b16\.5-20\d{2}$", re.IGNORECASE),
        re.compile(r"^\(revision of asme b16\.5-\d{4}\)$", re.IGNORECASE),
        re.compile(r"^an american national standard$", re.IGNORECASE),
    ]

    def clean(self, text: str) -> PdfCleanResult:
        pages = self._split_pages(text)
        cleaned_pages: list[str] = []
        removed_lines = 0
        removed_pages = 0

        for page_index, page_body in pages:
            cleaned_body, page_removed_lines = self._clean_page_body(page_body)
            removed_lines += page_removed_lines
            if not cleaned_body.strip():
                removed_pages += 1
                continue
            cleaned_pages.append(f"[Page {page_index}]\n{cleaned_body}")

        return PdfCleanResult(
            text="\n\n".join(cleaned_pages).strip(),
            removed_lines=removed_lines,
            removed_pages=removed_pages,
        )

    def _split_pages(self, text: str) -> list[tuple[int, str]]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        matches = list(re.finditer(r"^\[Page\s+(\d+)\]\s*$", normalized, flags=re.MULTILINE))
        if not matches:
            return [(1, normalized.strip())]

        pages: list[tuple[int, str]] = []
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
            page_number = int(match.group(1))
            page_body = normalized[start:end].strip()
            pages.append((page_number, page_body))
        return pages

    def _clean_page_body(self, page_body: str) -> tuple[str, int]:
        lines = [line.strip() for line in page_body.splitlines()]
        kept_lines: list[str] = []
        removed_lines = 0

        for line in lines:
            normalized = self._normalize_line(line)
            if not normalized:
                continue
            if self._is_noise_line(normalized):
                removed_lines += 1
                continue
            kept_lines.append(normalized)

        cleaned = "\n".join(kept_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip(), removed_lines

    @staticmethod
    def _normalize_line(line: str) -> str:
        normalized = line.replace("\u00a0", " ")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _is_noise_line(self, line: str) -> bool:
        if len(line) <= 2:
            return True
        if re.fullmatch(r"[`\-_,.\s]+", line):
            return True
        if re.fullmatch(r"\d+", line):
            return True
        for pattern in self._NOISE_PATTERNS:
            if pattern.match(line):
                return True
        return False

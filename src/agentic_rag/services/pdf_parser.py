from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from agentic_rag.services.pdf_cleaner import PdfTextCleaner


@dataclass(slots=True)
class ParsedPdf:
    # 解析 PDF 后返回的结构化结果。
    file_name: str
    pages: int
    text: str
    cleaned_characters: int
    removed_lines: int
    removed_pages: int


class PdfParser:
    def __init__(self) -> None:
        self.cleaner = PdfTextCleaner()

    def parse(self, pdf_path: Path) -> ParsedPdf:
        # PDF 解析逻辑比较直接：
        # 逐页提取文本，并保留页码信息，方便后续问答时回溯来源。
        reader = PdfReader(str(pdf_path))
        page_texts: list[str] = []

        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                page_texts.append(f"[Page {index}]\n{text}")

        cleaned = self.cleaner.clean("\n\n".join(page_texts).strip())
        return ParsedPdf(
            file_name=pdf_path.name,
            pages=len(reader.pages),
            text=cleaned.text,
            cleaned_characters=len(cleaned.text),
            removed_lines=cleaned.removed_lines,
            removed_pages=cleaned.removed_pages,
        )

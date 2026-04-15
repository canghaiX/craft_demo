from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(slots=True)
class ParsedPdf:
    file_name: str
    pages: int
    text: str


class PdfParser:
    def parse(self, pdf_path: Path) -> ParsedPdf:
        reader = PdfReader(str(pdf_path))
        page_texts: list[str] = []

        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                page_texts.append(f"[Page {index}]\n{text}")

        return ParsedPdf(
            file_name=pdf_path.name,
            pages=len(reader.pages),
            text="\n\n".join(page_texts).strip(),
        )

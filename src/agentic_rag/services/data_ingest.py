from pathlib import Path

from agentic_rag.config import Settings
from agentic_rag.schemas import BatchIngestFileResult, BatchIngestResponse
from agentic_rag.services.lightrag_service import LightRAGService
from agentic_rag.services.pdf_parser import PdfParser


class DataDirectoryIngestor:
    # 负责“扫描目录 -> 解析 PDF -> 导入 LightRAG”的批量流程。
    def __init__(
        self,
        settings: Settings,
        pdf_parser: PdfParser,
        lightrag_service: LightRAGService,
    ) -> None:
        self.settings = settings
        self.pdf_parser = pdf_parser
        self.lightrag_service = lightrag_service

    def discover_pdfs(self, source_dir: Path | None = None) -> list[Path]:
        # 递归扫描 source_dir 下所有 PDF 文件。
        root = (source_dir or self.settings.pdf_source_dir).resolve()
        if not root.exists():
            return []
        return sorted(path for path in root.rglob("*.pdf") if path.is_file())

    async def ingest_directory(self, source_dir: Path | None = None) -> BatchIngestResponse:
        # 批量导入的主流程：
        # 逐个 PDF 解析并调用 lightrag_service.ingest_document。
        root = (source_dir or self.settings.pdf_source_dir).resolve()
        pdf_files = self.discover_pdfs(root)
        indexed_files: list[BatchIngestFileResult] = []

        for pdf_file in pdf_files:
            parsed = self.pdf_parser.parse(pdf_file)
            chunks_indexed = await self.lightrag_service.ingest_document(
                parsed.text,
                file_path=str(pdf_file.relative_to(root)),
            )
            indexed_files.append(
                BatchIngestFileResult(
                    file_name=str(pdf_file.relative_to(root)),
                    pages=parsed.pages,
                    chunks_indexed=chunks_indexed,
                    characters=parsed.cleaned_characters,
                )
            )

        return BatchIngestResponse(
            source_dir=str(root),
            files_found=len(pdf_files),
            files_indexed=len(indexed_files),
            indexed_files=indexed_files,
        )

import argparse
import asyncio
import json

import uvicorn

from agentic_rag.app import build_data_ingestor, get_lightrag_service


def ingest_data_pdfs() -> None:
    lightrag_service = get_lightrag_service()

    async def _run() -> None:
        try:
            ingestor = build_data_ingestor(lightrag_service)
            result = await ingestor.ingest_directory()
            print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
        finally:
            await lightrag_service.close()

    asyncio.run(_run())


def run() -> None:
    uvicorn.run("agentic_rag.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic RAG service and ingestion CLI.")
    parser.add_argument(
        "command",
        nargs="?",
        default="serve",
        choices=["serve", "ingest-data-pdfs"],
        help="`serve` starts the API, `ingest-data-pdfs` loads PDFs from PDF_SOURCE_DIR into LightRAG.",
    )
    args = parser.parse_args()

    if args.command == "ingest-data-pdfs":
        ingest_data_pdfs()
    else:
        run()

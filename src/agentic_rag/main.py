import argparse
import asyncio
import json
import os

import uvicorn

from agentic_rag.app import build_data_ingestor, get_lightrag_service


def ingest_data_pdfs() -> None:
    # 命令行批量导入入口。
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
    # 命令行启动 API 服务。
    # 端口默认设为 8010，避免和常见的 vLLM 8000 端口冲突。
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8010"))
    uvicorn.run("agentic_rag.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    # 一个简单的双模式 CLI：
    # serve 启动服务，ingest-data-pdfs 批量导入文档。
    parser = argparse.ArgumentParser(description="Agentic RAG 服务与数据导入命令行工具。")
    parser.add_argument(
        "command",
        nargs="?",
        default="serve",
        choices=["serve", "ingest-data-pdfs"],
        help="`serve` 启动 API 服务，`ingest-data-pdfs` 会把 PDF_SOURCE_DIR 中的 PDF 导入 LightRAG。",
    )
    args = parser.parse_args()

    if args.command == "ingest-data-pdfs":
        ingest_data_pdfs()
    else:
        run()

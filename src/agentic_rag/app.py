from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from agentic_rag.config import get_settings
from agentic_rag.graph.workflow import AgenticRAGWorkflow
from agentic_rag.schemas import BatchIngestResponse, ChatRequest, ChatResponse, IngestResponse
from agentic_rag.services.data_ingest import DataDirectoryIngestor
from agentic_rag.services.lightrag_service import LightRAGService
from agentic_rag.services.llm import LLMServices
from agentic_rag.services.local_inference import get_local_inference_service
from agentic_rag.services.pdf_parser import PdfParser


# 这里在模块加载时就创建全局配置与 PDF 解析器，
# 这样接口函数里可以直接复用，避免每次请求重复初始化。
settings = get_settings()
pdf_parser = PdfParser()


def get_lightrag_service() -> LightRAGService:
    # 统一封装 LightRAGService 的构建逻辑，
    # 方便 API 与命令行入口共用同一套初始化方式。
    return LightRAGService(settings, inference_service=get_local_inference_service())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # FastAPI 生命周期钩子：
    # 启动时先占位，关闭时统一释放 LightRAG 底层资源。
    app.state.lightrag_service = None
    app.state.workflow = None
    yield
    lightrag_service = getattr(app.state, "lightrag_service", None)
    if lightrag_service is not None:
        await lightrag_service.close()


app = FastAPI(
    title="Agentic LightRAG 演示服务",
    version="0.1.0",
    description="基于 LangGraph 的 Agentic RAG 服务，支持 PDF 导入、标准三元组抽取与基于三元组的问答。",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    # 最基础的健康检查接口，常用于 notebook 或部署探针测试。
    return {"status": "正常"}


def ensure_services(app: FastAPI) -> tuple[LightRAGService, AgenticRAGWorkflow]:
    # 这里做的是“按需初始化”：
    # 第一次请求到来时再创建 LightRAGService 与工作流对象，
    # 后续请求直接复用，避免重复构建。
    if getattr(app.state, "lightrag_service", None) is None:
        app.state.lightrag_service = get_lightrag_service()
    if getattr(app.state, "workflow", None) is None:
        llm_services = LLMServices(settings, inference_service=get_local_inference_service())
        app.state.workflow = AgenticRAGWorkflow(
            llm_services=llm_services,
            lightrag_service=app.state.lightrag_service,
        )
    return app.state.lightrag_service, app.state.workflow


def build_data_ingestor(lightrag_service: LightRAGService) -> DataDirectoryIngestor:
    # 把批量导入依赖集中组装起来，保持 API 层代码更简洁。
    return DataDirectoryIngestor(
        settings=settings,
        pdf_parser=pdf_parser,
        lightrag_service=lightrag_service,
    )


@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(request: Request, file: UploadFile = File(...)) -> IngestResponse:
    # 上传单个 PDF 的接口：
    # 1. 临时保存上传文件
    # 2. 解析文本
    # 3. 直接抽取标准三元组并落盘
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="请上传 PDF 文件。")

    suffix = Path(file.filename).suffix or ".pdf"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = Path(tmp.name)

    try:
        lightrag_service, _ = ensure_services(request.app)
        parsed = pdf_parser.parse(temp_path)
        chunk_count = await lightrag_service.ingest_document(parsed.text, file_path=file.filename)
        return IngestResponse(
            file_name=parsed.file_name,
            pages=parsed.pages,
            chunks_indexed=chunk_count,
            characters=len(parsed.text),
            storage_dir=str(settings.lightrag_working_dir),
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/ingest/data-pdfs", response_model=BatchIngestResponse)
async def ingest_data_pdfs(request: Request) -> BatchIngestResponse:
    # 从 data 目录批量导入 PDF。
    try:
        lightrag_service, _ = ensure_services(request.app)
        ingestor = build_data_ingestor(lightrag_service)
        return await ingestor.ingest_directory()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    # 统一问答入口：
    # 由 LangGraph 工作流决定走 direct 还是 lightrag。
    try:
        _, workflow = ensure_services(request.app)
        result = await workflow.ainvoke(payload.question, payload.force_route)
        return ChatResponse(
            answer=result["answer"],
            route=result["route"],
            reason=result["reason"],
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

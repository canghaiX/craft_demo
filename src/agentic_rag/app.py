from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from agentic_rag.config import get_settings
from agentic_rag.graph.workflow import AgenticRAGWorkflow
from agentic_rag.schemas import ChatRequest, ChatResponse, IngestResponse
from agentic_rag.services.lightrag_service import LightRAGService
from agentic_rag.services.llm import LLMServices
from agentic_rag.services.pdf_parser import PdfParser


settings = get_settings()
pdf_parser = PdfParser()


def get_lightrag_service() -> LightRAGService:
    return LightRAGService(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.lightrag_service = None
    app.state.workflow = None
    yield
    lightrag_service = getattr(app.state, "lightrag_service", None)
    if lightrag_service is not None:
        await lightrag_service.close()


app = FastAPI(
    title="Agentic LightRAG Demo",
    version="0.1.0",
    description="LangGraph-based Agentic RAG service with PDF ingestion and LightRAG knowledge graph extraction.",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def ensure_services(app: FastAPI) -> tuple[LightRAGService, AgenticRAGWorkflow]:
    if getattr(app.state, "lightrag_service", None) is None:
        app.state.lightrag_service = get_lightrag_service()
    if getattr(app.state, "workflow", None) is None:
        llm_services = LLMServices(settings)
        app.state.workflow = AgenticRAGWorkflow(
            llm_services=llm_services,
            lightrag_service=app.state.lightrag_service,
        )
    return app.state.lightrag_service, app.state.workflow


@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(request: Request, file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

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


@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
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

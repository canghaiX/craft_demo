from typing import Literal

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    # 单个 PDF 上传导入后的返回体。
    file_name: str
    pages: int
    chunks_indexed: int
    characters: int
    storage_dir: str


class BatchIngestFileResult(BaseModel):
    # 批量导入时，单个文件的处理结果。
    file_name: str
    pages: int
    chunks_indexed: int
    characters: int


class BatchIngestResponse(BaseModel):
    # 批量导入接口的总体返回体。
    source_dir: str
    files_found: int
    files_indexed: int
    indexed_files: list[BatchIngestFileResult]


class ChatRequest(BaseModel):
    # force_route 允许用户显式指定走 direct 或 lightrag。
    question: str = Field(min_length=1)
    force_route: Literal["auto", "direct", "lightrag"] = "auto"


class RouteDecision(BaseModel):
    # 路由模型或启发式规则输出的统一结构。
    route: Literal["direct", "lightrag"]
    reason: str


class ChatResponse(BaseModel):
    # 问答接口最终返回给前端/调用方的结构。
    answer: str
    route: Literal["direct", "lightrag"]
    reason: str

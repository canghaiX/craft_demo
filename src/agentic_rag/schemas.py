from typing import Literal

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    file_name: str
    pages: int
    chunks_indexed: int
    characters: int
    storage_dir: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    force_route: Literal["auto", "direct", "lightrag"] = "auto"


class RouteDecision(BaseModel):
    route: Literal["direct", "lightrag"]
    reason: str


class ChatResponse(BaseModel):
    answer: str
    route: Literal["direct", "lightrag"]
    reason: str

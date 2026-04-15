from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    model_backend: str = Field(default="local", alias="MODEL_BACKEND")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")

    agent_model: str = Field(default="gpt-4o-mini", alias="AGENT_MODEL")
    router_model: str = Field(default="gpt-4o-mini", alias="ROUTER_MODEL")
    agent_model_path: Path = Field(default=Path("/data/models/qwen2.5-7B"), alias="AGENT_MODEL_PATH")
    router_model_path: Path = Field(default=Path("/data/models/qwen2.5-7B"), alias="ROUTER_MODEL_PATH")

    lightrag_llm_model: str = Field(default="gpt-4o-mini", alias="LIGHTRAG_LLM_MODEL")
    lightrag_llm_model_path: Path = Field(
        default=Path("/data/models/qwen2.5-7B"),
        alias="LIGHTRAG_LLM_MODEL_PATH",
    )
    lightrag_embed_model: str = Field(
        default="text-embedding-3-large",
        alias="LIGHTRAG_EMBED_MODEL",
    )
    lightrag_embed_model_path: Path = Field(
        default=Path("/data/models/bge-m3"),
        alias="LIGHTRAG_EMBED_MODEL_PATH",
    )
    lightrag_embed_dim: int = Field(default=3072, alias="LIGHTRAG_EMBED_DIM")
    lightrag_working_dir: Path = Field(default=Path("./data/lightrag"), alias="LIGHTRAG_WORKING_DIR")
    lightrag_query_mode: str = Field(default="hybrid", alias="LIGHTRAG_QUERY_MODE")
    pdf_source_dir: Path = Field(default=Path("./data"), alias="PDF_SOURCE_DIR")

    direct_qa_system_prompt: str = Field(
        default=(
            "You are a helpful assistant. Answer directly when the user asks a general question. "
            "If you are missing document-specific evidence, say that clearly."
        ),
        alias="DIRECT_QA_SYSTEM_PROMPT",
    )
    router_system_prompt: str = Field(
        default=(
            "Route to `lightrag` when the question depends on uploaded PDFs, internal facts, "
            "citations, or relationships between entities in the corpus. Route to `direct` for "
            "general knowledge, brainstorming, writing help, or when no corpus grounding is needed."
        ),
        alias="ROUTER_SYSTEM_PROMPT",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.lightrag_working_dir.mkdir(parents=True, exist_ok=True)
    settings.pdf_source_dir.mkdir(parents=True, exist_ok=True)
    return settings

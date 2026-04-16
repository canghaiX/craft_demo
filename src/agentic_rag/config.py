from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # 统一配置对象：
    # 所有环境变量最终都会映射到这里，项目其他模块直接依赖 Settings。
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    model_backend: str = Field(default="vllm", alias="MODEL_BACKEND")
    openai_api_key: str = Field(default="EMPTY", alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    embedding_base_url: str | None = Field(default=None, alias="EMBEDDING_BASE_URL")

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
    lightrag_query_mode: str = Field(default="mix", alias="LIGHTRAG_QUERY_MODE")
    lightrag_response_type: str = Field(default="Multiple Paragraphs", alias="LIGHTRAG_RESPONSE_TYPE")
    lightrag_top_k: int = Field(default=40, alias="LIGHTRAG_TOP_K")
    lightrag_chunk_top_k: int = Field(default=20, alias="LIGHTRAG_CHUNK_TOP_K")
    lightrag_include_references: bool = Field(default=True, alias="LIGHTRAG_INCLUDE_REFERENCES")
    lightrag_enable_llm_cache: bool = Field(default=True, alias="LIGHTRAG_ENABLE_LLM_CACHE")
    lightrag_chunk_token_size: int = Field(default=1200, alias="LIGHTRAG_CHUNK_TOKEN_SIZE")
    lightrag_chunk_overlap_token_size: int = Field(
        default=100,
        alias="LIGHTRAG_CHUNK_OVERLAP_TOKEN_SIZE",
    )
    lightrag_embedding_max_tokens: int = Field(default=8192, alias="LIGHTRAG_EMBEDDING_MAX_TOKENS")
    lightrag_llm_max_async: int = Field(default=8, alias="LIGHTRAG_LLM_MAX_ASYNC")
    lightrag_embedding_max_async: int = Field(default=8, alias="LIGHTRAG_EMBEDDING_MAX_ASYNC")
    lightrag_tiktoken_model_name: str = Field(
        default="gpt-4o-mini",
        alias="LIGHTRAG_TIKTOKEN_MODEL_NAME",
    )
    pdf_source_dir: Path = Field(default=Path("./data"), alias="PDF_SOURCE_DIR")

    direct_qa_system_prompt: str = Field(
        default=(
            "你是一名有帮助的智能助手。用户提出通用问题时请直接回答；"
            "如果缺少文档中的特定证据，也请明确说明。"
        ),
        alias="DIRECT_QA_SYSTEM_PROMPT",
    )
    router_system_prompt: str = Field(
        default=(
            "如果问题依赖已上传 PDF、文档内部事实、引用依据或语料中的实体关系，请路由到 `lightrag`。"
            "如果问题属于通用知识、头脑风暴、写作帮助，或不需要语料支撑，请路由到 `direct`。"
        ),
        alias="ROUTER_SYSTEM_PROMPT",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # 配置对象全局缓存，避免重复读取 .env。
    # 同时在启动时确保核心工作目录存在。
    settings = Settings()
    settings.lightrag_working_dir.mkdir(parents=True, exist_ok=True)
    settings.pdf_source_dir.mkdir(parents=True, exist_ok=True)
    return settings

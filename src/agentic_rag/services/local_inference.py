from __future__ import annotations

import asyncio
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI

from agentic_rag.config import Settings, get_settings
from agentic_rag.schemas import RouteDecision


class LocalGenerator:
    # 本地生成模型封装：
    # 当 MODEL_BACKEND=local 时，项目会直接通过 transformers 调本地模型。
    def __init__(self, model_path: str) -> None:
        self._ensure_path(model_path, "大语言模型")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "缺少本地 LLM 推理依赖，请先安装 `transformers`、`accelerate` 和 `torch`。"
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )

    @staticmethod
    def _ensure_path(model_path: str, model_type: str) -> None:
        # 提前检查路径，避免模型加载到一半才报错。
        if not Path(model_path).exists():
            raise RuntimeError(f"{model_type} 模型路径不存在：{model_path}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        # 把 system/user 消息套入 chat template，再送给因果语言模型生成。
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
        )
        new_tokens = generated_ids[:, model_inputs.input_ids.shape[1] :]
        response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return response.strip()


class LocalEmbedder:
    # 本地向量模型封装，用于把文本转换为 embedding。
    def __init__(self, model_path: str) -> None:
        LocalGenerator._ensure_path(model_path, "向量模型")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "缺少本地向量模型依赖，请先安装 `sentence-transformers` 和 `torch`。"
            ) from exc

        self.model = SentenceTransformer(model_path, trust_remote_code=True)

    def embed(self, texts: list[str]) -> np.ndarray:
        # normalize_embeddings=True 便于后续做向量相似度检索。
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class RemoteInferenceBackend:
    # 远程推理后端：
    # 通过 OpenAI 兼容接口调用 vLLM 等服务端模型。
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_base_url:
            raise RuntimeError("当 MODEL_BACKEND 设置为 `vllm` 时，必须提供 OPENAI_BASE_URL。")

        self.settings = settings
        self.llm_client = AsyncOpenAI(
            api_key=settings.openai_api_key or "EMPTY",
            base_url=str(settings.openai_base_url),
        )
        self.embedding_client = AsyncOpenAI(
            api_key=settings.openai_api_key or "EMPTY",
            base_url=str(settings.embedding_base_url or settings.openai_base_url),
        )

    async def generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        # 走 OpenAI Chat Completions 兼容接口生成答案。
        response = await self.llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = [part.text for part in content if getattr(part, "type", None) == "text"]
            return "".join(text_parts).strip()
        return ""

    async def embed(self, texts: list[str]) -> np.ndarray:
        # 远程 embedding 接口通常单独部署，因此这里和 llm_client 分开。
        response = await self.embedding_client.embeddings.create(
            model=self.settings.lightrag_embed_model,
            input=texts,
        )
        vectors = [item.embedding for item in response.data]
        return np.asarray(vectors, dtype=np.float32)


class LocalInferenceBackend:
    # 本地模式下，生成模型和向量模型都运行在当前机器上。
    def __init__(self, settings: Settings) -> None:
        self.generator = LocalGenerator(str(settings.agent_model_path))
        self.router_generator = self.generator
        if settings.router_model_path != settings.agent_model_path:
            self.router_generator = LocalGenerator(str(settings.router_model_path))
        self.embedder = LocalEmbedder(str(settings.lightrag_embed_model_path))

    async def generate(
        self,
        generator: LocalGenerator,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        # 本地模型推理是阻塞操作，放到线程池里避免卡住事件循环。
        return await asyncio.to_thread(
            generator.generate,
            system_prompt,
            user_prompt,
            max_new_tokens,
            temperature,
        )

    async def embed(self, texts: list[str]) -> np.ndarray:
        return await asyncio.to_thread(self.embedder.embed, texts)


class LocalInferenceService:
    # 这是项目统一的“推理门面”：
    # 上层不关心是本地模型还是远程 vLLM，都通过这里调用。
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backend_type = settings.model_backend.lower()
        if self.backend_type == "local":
            self.backend = LocalInferenceBackend(settings)
        else:
            self.backend = RemoteInferenceBackend(settings)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        use_router_model: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        # 根据 backend_type 自动选择本地或远程生成路径。
        if self.backend_type == "local":
            generator = self.backend.router_generator if use_router_model else self.backend.generator
            return await self.backend.generate(
                generator,
                system_prompt,
                user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        model = self.settings.router_model if use_router_model else self.settings.agent_model
        return await self.backend.generate(
            model,
            system_prompt,
            user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    async def embed(self, texts: list[str]) -> np.ndarray:
        return await self.backend.embed(texts)

    async def route_question(self, question: str) -> RouteDecision:
        # 路由判断优先做一层启发式短路：
        # 如果问题明显提到文档，就不用再调用路由模型。
        if any(keyword in question.lower() for keyword in ["pdf", "文档", "资料", "手册", "合同"]):
            return RouteDecision(route="lightrag", reason="问题明确提到了文档语料。")

        user_prompt = (
            "问题：\n"
            f"{question}\n\n"
            "请严格返回 JSON，并包含 `route` 和 `reason` 两个键。\n"
            "可选路由值只能是 `direct` 或 `lightrag`。"
        )
        response = await self.generate(
            self.settings.router_system_prompt,
            user_prompt,
            use_router_model=True,
            max_new_tokens=128,
            temperature=0.0,
        )
        return self._parse_route_decision(response)

    async def answer_directly(self, question: str) -> str:
        # direct 路由下的普通模型回答。
        return await self.generate(
            self.settings.direct_qa_system_prompt,
            question,
            max_new_tokens=512,
            temperature=0.2,
        )

    def _parse_route_decision(self, response: str) -> RouteDecision:
        # 优先尝试把模型返回当 JSON 解析；
        # 如果模型没有严格按 JSON 返回，再做字符串兜底判断。
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
                route = payload.get("route", "direct")
                reason = payload.get("reason", "已从本地路由模型的返回结果中解析。")
                if route in {"direct", "lightrag"}:
                    return RouteDecision(route=route, reason=str(reason))
            except json.JSONDecodeError:
                pass

        lowered = response.lower()
        if "lightrag" in lowered:
            return RouteDecision(route="lightrag", reason=response.strip() or "路由模型选择了 lightrag。")
        return RouteDecision(route="direct", reason=response.strip() or "路由模型选择了 direct。")


@lru_cache(maxsize=1)
def get_local_inference_service() -> LocalInferenceService:
    # 通过缓存确保全局只创建一个推理服务实例。
    return LocalInferenceService(get_settings())

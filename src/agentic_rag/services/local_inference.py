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
    def __init__(self, model_path: str) -> None:
        self._ensure_path(model_path, "LLM")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local LLM inference dependencies are missing. Install `transformers`, `accelerate`, and `torch`."
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
        if not Path(model_path).exists():
            raise RuntimeError(f"{model_type} model path does not exist: {model_path}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
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
    def __init__(self, model_path: str) -> None:
        LocalGenerator._ensure_path(model_path, "Embedding")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Local embedding dependencies are missing. Install `sentence-transformers` and `torch`."
            ) from exc

        self.model = SentenceTransformer(model_path, trust_remote_code=True)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class RemoteInferenceBackend:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_base_url:
            raise RuntimeError("OPENAI_BASE_URL is required when MODEL_BACKEND is set to `vllm`.")

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
        response = await self.embedding_client.embeddings.create(
            model=self.settings.lightrag_embed_model,
            input=texts,
        )
        vectors = [item.embedding for item in response.data]
        return np.asarray(vectors, dtype=np.float32)


class LocalInferenceBackend:
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
        if any(keyword in question.lower() for keyword in ["pdf", "文档", "资料", "手册", "合同"]):
            return RouteDecision(route="lightrag", reason="Question explicitly refers to document corpus.")

        user_prompt = (
            "Question:\n"
            f"{question}\n\n"
            "Return strict JSON with keys `route` and `reason`.\n"
            "Valid routes are `direct` or `lightrag`."
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
        return await self.generate(
            self.settings.direct_qa_system_prompt,
            question,
            max_new_tokens=512,
            temperature=0.2,
        )

    def _parse_route_decision(self, response: str) -> RouteDecision:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
                route = payload.get("route", "direct")
                reason = payload.get("reason", "Parsed from local router model.")
                if route in {"direct", "lightrag"}:
                    return RouteDecision(route=route, reason=str(reason))
            except json.JSONDecodeError:
                pass

        lowered = response.lower()
        if "lightrag" in lowered:
            return RouteDecision(route="lightrag", reason=response.strip() or "Router selected lightrag.")
        return RouteDecision(route="direct", reason=response.strip() or "Router selected direct.")


@lru_cache(maxsize=1)
def get_local_inference_service() -> LocalInferenceService:
    return LocalInferenceService(get_settings())

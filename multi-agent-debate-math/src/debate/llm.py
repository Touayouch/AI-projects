"""Client LLM unique pour tout serveur compatible OpenAI (Ollama, vLLM).

Chaque appel renvoie le texte ET sa consommation en tokens : c'est la base des
comparaisons à budget égal entre débat et vote majoritaire.
"""

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

Message = dict[str, str]


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    finish_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        timeout: float = 600.0,
        max_concurrency: int = 8,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, http_client=http_client
        )
        # Limite le nombre de requêtes simultanées envoyées au serveur
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> Completion:
        async with self.semaphore:
            start = time.perf_counter()
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            latency = time.perf_counter() - start

        choice = response.choices[0]
        usage = response.usage
        if usage is None:
            raise RuntimeError(f"Le serveur n'a pas renvoyé la consommation en tokens ({model})")
        return Completion(
            text=choice.message.content or "",
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency_s=round(latency, 3),
            finish_reason=choice.finish_reason,
        )

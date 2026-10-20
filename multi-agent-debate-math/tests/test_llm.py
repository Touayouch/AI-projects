import asyncio
import json

import httpx
import pytest

from debate.llm import LLMClient


def chat_response(content: str, prompt_tokens: int = 12, completion_tokens: int = 5) -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def make_client(handler, max_concurrency: int = 8) -> LLMClient:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return LLMClient("http://test/v1", max_concurrency=max_concurrency, http_client=http_client)


async def test_complete_returns_text_and_token_usage():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response("391", 20, 3))

    completion = await make_client(handler).complete(
        [{"role": "user", "content": "17 × 23 ?"}], model="m", temperature=0.5, seed=1
    )

    assert completion.text == "391"
    assert (completion.prompt_tokens, completion.completion_tokens) == (20, 3)
    assert completion.total_tokens == 23
    assert completion.finish_reason == "stop"
    assert requests[0]["temperature"] == 0.5
    assert requests[0]["seed"] == 1


async def test_missing_usage_is_an_error():
    def handler(request):
        body = chat_response("x")
        del body["usage"]
        return httpx.Response(200, json=body)

    with pytest.raises(RuntimeError, match="tokens"):
        await make_client(handler).complete([{"role": "user", "content": "q"}], model="m")


async def test_concurrency_is_limited():
    active = 0
    peak = 0

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(200, json=chat_response("ok"))

    client = make_client(handler, max_concurrency=2)
    await asyncio.gather(
        *(client.complete([{"role": "user", "content": "q"}], model="m") for _ in range(6))
    )
    assert peak <= 2


async def test_requests_are_routed_by_model():
    hosts = []

    def handler(request):
        hosts.append((request.url.host, json.loads(request.content)["model"]))
        return httpx.Response(200, json=chat_response("ok"))

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = LLMClient(
        "http://default/v1", http_client=http_client, endpoints={"b": "http://server-b/v1"}
    )
    await client.complete([{"role": "user", "content": "q"}], model="a")
    await client.complete([{"role": "user", "content": "q"}], model="b")
    assert hosts == [("default", "a"), ("server-b", "b")]

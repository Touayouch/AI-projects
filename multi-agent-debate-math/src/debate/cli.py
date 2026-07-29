"""Ligne de commande du projet.

debate ping "Combien font 17 × 23 ?"   # vérifie la connexion au serveur LLM
"""

import argparse
import asyncio
import logging

from debate.config import get_settings
from debate.llm import LLMClient


def cmd_ping(args: argparse.Namespace) -> None:
    settings = get_settings()
    client = LLMClient(
        settings.llm_base_url, settings.llm_api_key, timeout=settings.request_timeout
    )
    completion = asyncio.run(
        client.complete(
            [{"role": "user", "content": args.prompt}],
            model=args.model or settings.default_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            seed=settings.seed,
        )
    )
    print(completion.text)
    print(
        f"\n[{completion.model}] {completion.prompt_tokens} tokens en entrée, "
        f"{completion.completion_tokens} en sortie, {completion.latency_s:.1f} s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="debate")
    sub = parser.add_subparsers(required=True)

    ping = sub.add_parser("ping", help="Teste la connexion au serveur LLM")
    ping.add_argument("prompt", nargs="?", default="Combien font 17 × 23 ? Réponds en une ligne.")
    ping.add_argument("--model")
    ping.set_defaults(func=cmd_ping)

    args = parser.parse_args()
    logging.basicConfig(level=get_settings().log_level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()

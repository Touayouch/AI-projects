.PHONY: install lint format test models ping check-grader

install:
	uv sync

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

test:
	uv run pytest -v

models:
	ollama pull hf.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M
	ollama cp hf.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M qwen2.5-instruct:1.5b

ping:
	uv run debate ping

check-grader:
	uv run debate check-grader

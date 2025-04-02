.PHONY: install lint format test corpus models ingest ingest-recursive api chroma-up chroma-down

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

corpus:
	uv run rag download

models:
	ollama pull bge-m3
	ollama pull qwen3:4b

ingest:
	uv run rag ingest --strategy semantic

ingest-recursive:
	uv run rag ingest --strategy recursive

api:
	uv run uvicorn rag.api.main:app --reload --port 8080

chroma-up:
	docker compose up -d chroma

chroma-down:
	docker compose down

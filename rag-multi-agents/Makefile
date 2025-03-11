.PHONY: install lint format test api chroma-up chroma-down

install:
	uv sync --extra anthropic

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

test:
	uv run pytest -v

api:
	uv run uvicorn rag.api.main:app --reload --port 8080

chroma-up:
	docker compose up -d chroma

chroma-down:
	docker compose down

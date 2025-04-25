.PHONY: install lint format test corpus models ingest ingest-recursive eval-retrieval graph api chroma-up chroma-down

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
	# Repli sur HuggingFace si le registre Ollama est inaccessible
	ollama pull bge-m3 || (ollama pull hf.co/gpustack/bge-m3-GGUF:Q8_0 && ollama cp hf.co/gpustack/bge-m3-GGUF:Q8_0 bge-m3)
	ollama pull hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
	ollama cp hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M qwen3-instruct:4b

ingest:
	uv run rag ingest --strategy semantic

ingest-recursive:
	uv run rag ingest --strategy recursive

eval-retrieval:
	uv run rag eval-retrieval --strategy semantic

graph:
	uv run rag graph

api:
	uv run uvicorn rag.api.main:app --reload --port 8080

chroma-up:
	docker compose up -d chroma

chroma-down:
	docker compose down

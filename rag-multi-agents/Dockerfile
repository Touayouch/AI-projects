FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dépendances d'abord, pour profiter du cache Docker quand seul le code change
COPY pyproject.toml README.md ./
COPY src/rag/__init__.py src/rag/__init__.py
RUN pip install .

COPY src ./src
RUN pip install --no-deps --force-reinstall .

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["uvicorn", "rag.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

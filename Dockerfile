# Container for Google Cloud Run (also works on any Docker host).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.hf
WORKDIR /app

COPY requirements-server.txt .
RUN pip install -r requirements-server.txt

COPY app app
COPY frontend frontend
COPY data data

# Download the embedding + reranker models and build the PDF index now, at build time,
# so the container starts in seconds instead of re-embedding every PDF on each start.
RUN GROQ_API_KEY=build-time-placeholder python -c "from app.rag_engine import RagEngine; RagEngine()"

# Cloud Run sets $PORT (8080 by default).
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}

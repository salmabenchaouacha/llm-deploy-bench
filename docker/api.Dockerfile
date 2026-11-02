# API gateway image (FastAPI + RAG client code). The LLM itself runs in the official
# llama.cpp CUDA image (service `llm` in docker-compose.yml): this image needs no GPU.
#
# Built by `make serve-up`, which passes the local uv version so the lockfile written on
# the host is read by the same uv in the image.
ARG UV_VERSION=0.9.5
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:3.11-slim
COPY --from=uv /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# Dependencies first: this layer stays cached while only the source code changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
COPY configs ./configs
RUN uv sync --locked --no-dev

RUN useradd --create-home --uid 10001 app
USER app
ENV PATH=/app/.venv/bin:$PATH \
    LLMBENCH_LOG_JSON=1
EXPOSE 8000
CMD ["uvicorn", "--factory", "llmbench.api.app:create_app", "--host", "0.0.0.0", "--port", "8000"]

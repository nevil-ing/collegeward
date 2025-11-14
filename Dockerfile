FROM python:3.11.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/

WORKDIR  /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY app ./app

FROM python:3.11.14-slim
WORKDIR /app

COPY --from=builder /app /app

EXPOSE 8005

#run the app
CMD ["/app/.venv/bin/fastapi", "run", "app/main.py", "--port", "8005", "--host", "0.0.0.0"]
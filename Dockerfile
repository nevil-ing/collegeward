FROM python:3.11.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/

WORKDIR  /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY app ./app
COPY alembic.ini ./
COPY alembic ./alembic


FROM python:3.11.14-slim
WORKDIR /app

COPY --from=builder /app /app

RUN mkdir -p logs

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh


EXPOSE 8005

#run the app
START ["/app/start.sh"]
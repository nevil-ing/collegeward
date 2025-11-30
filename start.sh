#!/usr/bin/bash

set -e
echo "Running  Alembic migrations...."
/app/.venv/bin/alembic upgrade head

echo "starting Application..."
exec /app/.venv/bin/fastapi run app/main.py --port 8005 --host 0.0.0.0

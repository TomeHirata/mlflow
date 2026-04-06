#!/bin/bash
set -e

# MLflow server automatically creates tables and runs migrations on startup
# via SqlAlchemyStore initialization — no separate `mlflow db upgrade` needed.
exec mlflow server \
    --backend-store-uri "$DATABASE_URL" \
    --default-artifact-root "${ARTIFACT_ROOT:-/tmp/mlflow-artifacts}" \
    --host 0.0.0.0 \
    --port "${PORT:-5000}" \
    --workers "${WORKERS:-2}"

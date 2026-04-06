#!/bin/bash
set -e

# Run database migrations (creates tables if they don't exist)
mlflow db upgrade "$DATABASE_URL"

# Start MLflow tracking server
exec mlflow server \
    --backend-store-uri "$DATABASE_URL" \
    --default-artifact-root "${ARTIFACT_ROOT:-/tmp/mlflow-artifacts}" \
    --host 0.0.0.0 \
    --port "${PORT:-5000}" \
    --workers "${WORKERS:-2}"

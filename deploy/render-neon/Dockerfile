FROM python:3.10-slim

# Install system dependencies for psycopg2
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

# Install MLflow with PostgreSQL support
RUN pip install --no-cache-dir mlflow psycopg2-binary

EXPOSE 5000

COPY deploy/render-neon/start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]

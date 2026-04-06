# Deploy MLflow to Render + Neon (Free Tier)

## 1. Create a free Neon PostgreSQL database

1. Sign up at [neon.tech](https://neon.tech) (free tier: 0.5 GB storage, autoscaling)
2. Create a new project (e.g., `mlflow`)
3. Copy the connection string — it looks like:
   ```
   postgresql://neondb_owner:abc123@ep-cool-name-123456.us-east-2.aws.neon.tech/neondb?sslmode=require
   ```

## 2. Deploy to Render

### Option A: One-click Blueprint

1. Push this repo to GitHub
2. Go to [dashboard.render.com](https://dashboard.render.com) → **New** → **Blueprint**
3. Connect your repo and select it
4. Render will detect `deploy/render-neon/render.yaml`
5. Set the `DATABASE_URL` env var to your Neon connection string
6. Deploy

### Option B: Manual setup

1. Go to [dashboard.render.com](https://dashboard.render.com) → **New** → **Web Service**
2. Connect your GitHub repo
3. Configure:
   - **Environment**: Docker
   - **Dockerfile Path**: `deploy/render-neon/Dockerfile`
   - **Plan**: Free
4. Add environment variables:
   - `DATABASE_URL` = your Neon connection string
   - `MLFLOW_SQLALCHEMYSTORE_POOLCLASS` = `NullPool`
   - `MLFLOW_SERVER_ENABLE_JOB_EXECUTION` = `false`
5. Deploy

## 3. Use it

Once deployed, your MLflow UI is at `https://mlflow-xxxx.onrender.com`.

Point your local MLflow client at it:

```bash
export MLFLOW_TRACKING_URI="https://mlflow-xxxx.onrender.com"

# Log a run
python -c "
import mlflow
mlflow.set_experiment('my-experiment')
with mlflow.start_run():
    mlflow.log_param('lr', 0.01)
    mlflow.log_metric('accuracy', 0.95)
    print('Run logged successfully!')
"
```

## Limitations

| Aspect | Limitation |
|---|---|
| **Render free tier** | Spins down after 15 min of inactivity; cold starts take ~30s |
| **Neon free tier** | 0.5 GB storage, compute suspends after 5 min idle |
| **Artifacts** | Stored in ephemeral `/tmp` — lost on redeploy. For persistent artifacts, use S3 (see below) |

## Optional: Persistent artifact storage with S3

For durable artifact storage, set `ARTIFACT_ROOT` to an S3 URI and add AWS credentials:

```yaml
envVars:
  - key: ARTIFACT_ROOT
    value: s3://my-bucket/mlflow-artifacts
  - key: AWS_ACCESS_KEY_ID
    sync: false
  - key: AWS_SECRET_ACCESS_KEY
    sync: false
```

And add `boto3` to the Dockerfile:

```dockerfile
RUN pip install --no-cache-dir mlflow psycopg2-binary boto3
```

# Anomaly Detection MLOps

Local development stack for the anomaly-detection API, worker pipeline, model artifacts, and UI.

## Services

The Compose stack starts:

- PostgreSQL for application metadata, datasets, jobs, and the model registry;
- Redis for Celery queues/results and application state;
- Azurite Blob storage for datasets, features, models, and reports;
- an Alembic migration job;
- a Blob-container initialization job;
- FastAPI backend;
- Celery worker;
- Celery Beat scheduler;
- React/Nginx frontend.

Data is persisted in the named volumes `postgres_data`, `redis_data`, and `azurite_data`.

## Start locally

Requirements: Docker Engine with Docker Compose v2.

```bash
cp .env.example .env
# Replace SECRET_KEY in .env with a long random local value.
docker compose up --build
```

On PowerShell:

```powershell
Copy-Item .env.example .env
# Replace SECRET_KEY in .env with a long random local value.
docker compose up --build
```

Open:

- UI: <http://localhost:3000>
- API documentation: <http://localhost:8000/docs>
- API health: <http://localhost:8000/health>

The API health response checks PostgreSQL, Redis, and Blob storage. Compose waits for database migrations and storage-container initialization before starting the API and worker.

## Useful commands

```bash
# View service state
docker compose ps

# Follow API and worker logs
docker compose logs -f backend celery-worker celery-beat

# Re-run migrations
docker compose run --rm migrate

# Re-create/check Blob containers
docker compose run --rm storage-init

# Stop containers but retain data
docker compose down

# Stop and remove local database, Redis, and Blob data
docker compose down --volumes
```

The final command permanently deletes the local Compose volumes; use it only when a clean local reset is intended.

## Local storage layout

Azurite provides the same Blob API used by the application. The storage initializer creates these containers:

- `datasets`
- `models`
- `features`
- `monitoring`
- `audit-logs`
- `experiments`
- `backups`
- `temp-processing`

Do not use the emulator account/key outside local development.

## Current scope

This bootstrap focuses on making the existing platform services run together. Broader security hardening and deeper ML redesign are tracked separately in the workspace audit.

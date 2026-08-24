FROM python:3.11-slim as builder

WORKDIR /app

# Install uv binaries (fast package installer)
COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

# Copy dependency manifest first to maximize cache hits
COPY pyproject.toml ./

# Resolve deps with Poetry securely
RUN uv pip install --system poetry==2.3.3 poetry-plugin-export==1.9.0 && \
    poetry export --without-hashes --only main -f requirements.txt -o requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv binaries
COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

# Copy the exported requirements from builder stage
COPY --from=builder /app/requirements.txt ./

# Upgrade core python packages to fix base image vulnerabilities
RUN uv pip install --system --no-cache --upgrade pip setuptools wheel>=0.46.2 jaraco.context>=6.1.0

# Install application dependencies
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application
COPY . .

# Create non-root user (silences Celery ROOT_DISCOURAGED warning)
RUN useradd -r -s /bin/false appuser && chown -R appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

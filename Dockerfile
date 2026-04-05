# Unified-M Pipeline & API Image
# ================================
# Multi-stage build: slim runtime with only what's needed.
# Uses uv lockfile for fast, reproducible, hash-verified installs.

# ---- Python API stage ----
FROM python:3.11-slim AS base

# Create a non-root user before anything else
RUN groupadd --system appgroup && useradd --system --gid appgroup --no-create-home appuser

WORKDIR /app

# System deps for scientific Python
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (advanced Python package manager — lockfile = reproducible retrieval)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency manifests first (Docker layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies only (not the project itself) for layer caching.
# --frozen: fail if lockfile is out of date (CI safety net)
# --no-dev: skip [dependency-groups] dev (production image)
# --no-install-project: install deps first, project source comes next
ENV UV_COMPILE_BYTECODE=0
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source + README (needed by hatch to build the wheel metadata)
COPY src/ src/
COPY config.yaml README.md ./
RUN uv sync --frozen --no-dev --no-editable

# Hand ownership to the non-root user and switch
RUN chown -R appuser:appgroup /app
USER appuser

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Default: run the API server using the venv Python directly
CMD [".venv/bin/python", "-m", "cli", "serve", "--host", "0.0.0.0", "--port", "8000"]

# ---- UI build stage ----
FROM node:20-slim AS ui-build

RUN npm install -g bun

WORKDIR /app/ui
COPY ui/package.json ui/bun.lock* ./
RUN bun install --frozen-lockfile || bun install

COPY ui/ .
RUN bun run build

# ---- Nginx stage for serving built UI ----
FROM nginx:alpine AS ui

COPY --from=ui-build /app/ui/dist /usr/share/nginx/html

# SPA fallback: serve index.html for all routes
RUN printf 'server {\n\
    listen 80;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    location /api/ {\n\
        proxy_pass http://api:8000;\n\
    }\n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

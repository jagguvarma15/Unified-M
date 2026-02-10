# Unified-M Pipeline & API Image
# ================================
# Multi-stage build: slim runtime with only what's needed.

FROM python:3.14-slim AS base

WORKDIR /app

# System deps for scientific Python
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config.yaml ./

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Default: run the API server
CMD ["python", "-m", "cli", "serve", "--host", "0.0.0.0", "--port", "8000"]

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
    }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app/src

# Install uv from the official distroless image. Cache-mount makes subsequent
# rebuilds near-instant (10-50x faster than pip).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies from pyproject.toml. We copy only the metadata + src
# first so dependency-resolution caches survive source-only changes downstream.
COPY pyproject.toml README.md ./
COPY src/ /app/src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache .

# Copy the rest of the project (entry shim, scripts, fixtures, configs).
COPY . /app/

EXPOSE 8001

CMD ["python", "main.py"]

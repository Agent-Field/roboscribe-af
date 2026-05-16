FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

# Install uv from the official distroless image — ~10-50x faster than pip,
# and cache-mount makes subsequent rebuilds near-instant.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache -r /app/requirements.txt

COPY . /app/

EXPOSE 8001

CMD ["python", "main.py"]

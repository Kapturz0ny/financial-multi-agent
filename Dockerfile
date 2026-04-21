FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the official binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Enable bytecode compilation for faster startups
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONPATH=/app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

COPY src/ ./src/

# Create the .context directory for session storage
RUN mkdir -p .context

EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["uv", "run", "streamlit", "run", "src/app.py"]
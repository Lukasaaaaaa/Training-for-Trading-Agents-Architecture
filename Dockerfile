FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.7.1

# Copy project files
COPY pyproject.toml poetry.lock ./
COPY trading_orchestrator ./trading_orchestrator
COPY examples ./examples

# Configure Poetry to not create virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Create directories for checkpoints and data
RUN mkdir -p /app/checkpoints /app/data /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV LOG_FORMAT=json

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import trading_orchestrator; print('healthy')"

# Run the CLI by default
CMD ["python", "-m", "trading_orchestrator.cli"]

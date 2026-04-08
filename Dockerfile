FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
COPY openenv.yaml .
COPY .env.example .env

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY env/ ./env/
COPY tasks/ ./tasks/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY inference.py ./

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_TOKEN=${HF_TOKEN}
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

# Run tests on build (optional)
RUN pytest tests/ -v --tb=short || true

# Expose the app port
EXPOSE 8080

# Default command: start the OpenEnv inference server
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8080"]

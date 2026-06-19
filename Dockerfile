# Small base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory inside container
WORKDIR /app

# Install dependencies first (better Docker layer caching)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model artifact
COPY src/ /app/src/
COPY models/ /app/models/

# Default command: run the CLI predictor
ENTRYPOINT ["python", "src/predict_cli.py"]
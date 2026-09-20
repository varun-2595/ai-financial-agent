FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configuration
COPY . .

# Create logs and data directory
RUN mkdir -p /app/logs /app/data

# Expose healthcheck / status port if needed
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Kolkata

CMD ["python", "main.py"]

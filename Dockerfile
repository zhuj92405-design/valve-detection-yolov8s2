FROM python:3.11-slim

LABEL maintainer="ValveAI"
LABEL description="YOLOv8s Valve Detection API Server"
LABEL version="1.0"

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir ultralytics fastapi uvicorn python-multipart pillow

# Copy model and inference code
COPY best.pt .
COPY server.py .

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run API server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

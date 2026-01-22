FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    prometheus-client \
    requests


# Copy app code
COPY src/olivetti_faces/system_monitoring.py /app/system_monitoring.py

EXPOSE 8080

CMD ["sh", "-c", "uvicorn system_monitoring:app --host 0.0.0.0 --port ${PORT}"]

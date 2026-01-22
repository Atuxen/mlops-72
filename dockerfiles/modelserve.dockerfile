FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    scikit-learn \
    joblib \
    pydantic \
    google-cloud-storage \
    prometheus-client


COPY src/olivetti_faces/model.py /app/model.py
COPY src/olivetti_faces/models /app/models

EXPOSE 8080

CMD ["sh", "-c", "uvicorn model:app --host 0.0.0.0 --port ${PORT}"]

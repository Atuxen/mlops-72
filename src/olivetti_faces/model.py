import os
import time
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional
from sklearn.svm import SVC
import joblib
from fastapi import FastAPI, HTTPException
from google.cloud import storage
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram
from fastapi import Request
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
from fastapi.responses import Response

VALID_KERNELS = {"linear", "poly", "rbf", "sigmoid", "precomputed"}


def build_svm(kernel: str, C: float, gamma: str) -> SVC:
    if kernel not in VALID_KERNELS:
        raise ValueError(f"Invalid kernel '{kernel}'. Must be one of {VALID_KERNELS}")
    return SVC(kernel=kernel, C=C, gamma=gamma, probability=True)


# -------------------------
# Prometheus metrics
# -------------------------
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total predictions attempted",
    ["endpoint"],
)

PREDICTION_ERRORS_TOTAL = Counter(
    "prediction_errors_total",
    "Total prediction errors",
    ["endpoint"],
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "prediction_latency_seconds",
    "Latency for prediction endpoints (seconds)",
    ["endpoint"],
)

MODEL_VERSION_INFO = Counter(
    "model_version_info",
    "Constant 1 labeled by deployed model version",
    ["version"],
)

# -------------------------
# Config (env vars)
# -------------------------
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "mlops-72-bucket")
MODEL_PREFIX = os.getenv("MODEL_PREFIX", "models")
MODEL_NAME = os.getenv("MODEL_NAME", "svm-face")

# Optional fallback if you still want local-baked model for dev
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "/app/models/svm.pkl")

LOCAL_DIR = Path("/tmp/model")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# Globals populated at startup
svm = None
pca = None
model_version: Optional[str] = None


# -------------------------
# Startup: download latest bundle from GCS
# -------------------------
def _load_from_gcs() -> None:
    global svm, model_version

    client = storage.Client()
    bucket = client.bucket(MODEL_BUCKET)

    latest_path = f"{MODEL_PREFIX}/{MODEL_NAME}/LATEST.json"
    latest_blob = bucket.blob(latest_path)
    latest = json.loads(latest_blob.download_as_text())

    model_version = latest["version"]
    bundle_blob_name = latest["bundle"]  # e.g. models/svm-face/versions/.../svm.pkl
    MODEL_VERSION_INFO.labels(version=model_version).inc()

    local_bundle = LOCAL_DIR / "svm.pkl"
    bucket.blob(bundle_blob_name).download_to_filename(str(local_bundle))

    loaded = joblib.load(local_bundle)
    if isinstance(loaded, dict):
        svm = loaded.get("model") or loaded.get("svm")
        pca_loaded = loaded.get("pca")
        if pca_loaded is None:
            raise RuntimeError("Bundle dict missing key 'pca'")
        globals()["pca"] = pca_loaded
    else:
        # If you ever store only svm directly, you *can't* do image->pca->svm
        svm = loaded
        globals()["pca"] = None

    print(f"Loaded model_version={model_version} from gs://{MODEL_BUCKET}/{bundle_blob_name}")


def _load_local_fallback() -> None:
    global svm, model_version
    loaded = joblib.load(LOCAL_MODEL_PATH)
    if isinstance(loaded, dict):
        svm = loaded.get("model") or loaded.get("svm")
        pca_loaded = loaded.get("pca")
        if pca_loaded is None:
            raise RuntimeError("Local bundle dict missing key 'pca'")
        globals()["pca"] = pca_loaded
    else:
        svm = loaded
        globals()["pca"] = None

    print(f"Loaded local model from {LOCAL_MODEL_PATH} (model_version={model_version})")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global svm
    try:
        # Prefer GCS in cloud. If it fails, fall back to local (optional).
        try:
            _load_from_gcs()
        except Exception as e:
            # If you *never* want local fallback, remove this block and just raise.
            print(f"GCS load failed: {e}. Trying local fallback from {LOCAL_MODEL_PATH} ...")
            _load_local_fallback()

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

    yield

    svm = None


# -------------------------
# API schemas
# -------------------------
class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="100-dim PCA feature vector")


class PredictImageRequest(BaseModel):
    pixels: List[float] = Field(
        ...,
        description="Flattened 64x64 grayscale image (length 4096). Values should match training scale.",
        min_length=64 * 64,
        max_length=64 * 64,
    )


class PredictResponse(BaseModel):
    label: int
    probabilities: List[float]
    model_version: str
    latency_ms: float


# -------------------------
# App
# -------------------------
app = FastAPI(title="svm-model-server", version="1.0.0", lifespan=lifespan)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    path = request.url.path
    method = request.method
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=status).inc()


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/readyz")
def readyz():
    if svm is None or model_version is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"ready": True, "model_version": model_version}


@app.post("/predict_image", response_model=PredictResponse)
def predict_image(req: PredictImageRequest):
    if svm is None or model_version is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if pca is None:
        raise HTTPException(status_code=503, detail="PCA not loaded (bundle missing 'pca')")

    # sklearn expects shape (n_samples, n_features)
    X_pixels = [req.pixels]  # (1, 4096)
    PREDICTIONS_TOTAL.labels(endpoint="/predict_image").inc()

    start = time.perf_counter()
    try:
        X_pca = pca.transform(X_pixels)  # (1, 100)
        label = int(svm.predict(X_pca)[0])
        probs = svm.predict_proba(X_pca)[0].tolist()
        latency_ms = (time.perf_counter() - start) * 1000.0

        return PredictResponse(
            label=label,
            probabilities=probs,
            model_version=model_version,
            latency_ms=latency_ms,
        )
    except Exception as e:
        PREDICTION_ERRORS_TOTAL.labels(endpoint="/predict_image").inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e
    finally:
        PREDICTION_LATENCY_SECONDS.labels(endpoint="/predict_image").observe(time.perf_counter() - start)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if svm is None or model_version is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # sklearn expects shape (n_samples, n_features)
    X = [req.features]

    start = time.perf_counter()
    try:
        label = int(svm.predict(X)[0])
        probs = svm.predict_proba(X)[0].tolist()
        latency_ms = (time.perf_counter() - start) * 1000.0

        return PredictResponse(
            label=label,
            probabilities=probs,
            model_version=model_version,
            latency_ms=latency_ms,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e

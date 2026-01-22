import os
import time
import threading
from typing import Any, Dict, Optional
import json

import requests
from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

# -------------------------
# Config (env vars)
# -------------------------
TARGET_BASE_URL = os.getenv("TARGET_BASE_URL", "http://localhost:8080").rstrip("/")
TIMEOUT_SECONDS = float(os.getenv("TIMEOUT_SECONDS", "2.0"))

PROBE_INTERVAL_SECONDS = float(os.getenv("PROBE_INTERVAL_SECONDS", "0"))

PREDICT_PATH = os.getenv("PREDICT_PATH", "/predict")
HEALTH_PATH = os.getenv("HEALTH_PATH", "/healthz")
READY_PATH = os.getenv("READY_PATH", "/readyz")

PREDICT_JSON_KEY = os.getenv("PREDICT_JSON_KEY", "text")
PREDICT_JSON_VALUE_RAW = os.getenv("PREDICT_JSON_VALUE", '"synthetic-check"')  # JSON string by default

# Parse once at startup/import
try:
    PREDICT_JSON_VALUE = json.loads(PREDICT_JSON_VALUE_RAW)
except json.JSONDecodeError as e:
    raise RuntimeError(
        f"PREDICT_JSON_VALUE must be valid JSON. Got: {PREDICT_JSON_VALUE_RAW}"
    ) from e

# -------------------------
# Prometheus metrics (about the TARGET)
# -------------------------
PROBE_SUCCESS = Gauge("probe_success", "1 if last probe succeeded, else 0", ["endpoint"])
PROBE_HTTP_STATUS = Gauge("probe_http_status", "HTTP status code of last probe", ["endpoint"])
PROBE_LATENCY_SECONDS = Histogram("probe_latency_seconds", "Latency of probe HTTP request in seconds", ["endpoint"])
PROBE_FAILURES_TOTAL = Counter("probe_failures_total", "Number of probe failures", ["endpoint", "reason"])
PROBES_TOTAL = Counter("probes_total", "Total number of probes attempted", ["endpoint"])

# -------------------------
# App
# -------------------------
app = FastAPI(title="synthetic-monitor", version="0.1.0")
app.mount("/metrics", make_asgi_app())

_last_results: Dict[str, Any] = {}
_lock = threading.Lock()


def _do_request(endpoint: str, method: str = "GET", json_body: Optional[dict] = None) -> Dict[str, Any]:
    url = f"{TARGET_BASE_URL}{endpoint}"
    PROBES_TOTAL.labels(endpoint=endpoint).inc()

    start = time.perf_counter()
    try:
        if method == "POST":
            resp = requests.post(url, json=json_body, timeout=TIMEOUT_SECONDS)
        else:
            resp = requests.get(url, timeout=TIMEOUT_SECONDS)

        latency = time.perf_counter() - start
        PROBE_LATENCY_SECONDS.labels(endpoint=endpoint).observe(latency)
        PROBE_HTTP_STATUS.labels(endpoint=endpoint).set(resp.status_code)

        ok = 200 <= resp.status_code < 300
        PROBE_SUCCESS.labels(endpoint=endpoint).set(1 if ok else 0)
        if not ok:
            PROBE_FAILURES_TOTAL.labels(endpoint=endpoint, reason="non_2xx").inc()

        return {
            "ok": ok,
            "endpoint": endpoint,
            "status": resp.status_code,
            "latency_seconds": latency,
            "body_preview": resp.text[:200],
        }

    except requests.Timeout:
        PROBE_SUCCESS.labels(endpoint=endpoint).set(0)
        PROBE_HTTP_STATUS.labels(endpoint=endpoint).set(0)
        PROBE_FAILURES_TOTAL.labels(endpoint=endpoint, reason="timeout").inc()
        return {"ok": False, "endpoint": endpoint, "error": "timeout"}

    except Exception as e:
        PROBE_SUCCESS.labels(endpoint=endpoint).set(0)
        PROBE_HTTP_STATUS.labels(endpoint=endpoint).set(0)
        PROBE_FAILURES_TOTAL.labels(endpoint=endpoint, reason="exception").inc()
        return {"ok": False, "endpoint": endpoint, "error": str(e)}


def run_probes() -> Dict[str, Any]:
    results = {}
    results["healthz"] = _do_request(HEALTH_PATH, "GET")
    results["readyz"] = _do_request(READY_PATH, "GET")

    payload = {PREDICT_JSON_KEY: PREDICT_JSON_VALUE}
    results["predict"] = _do_request(PREDICT_PATH, "POST", json_body=payload)

    with _lock:
        _last_results.clear()
        _last_results.update(results)

    return results


def _background_loop():
    while True:
        try:
            run_probes()
        except Exception:
            pass
        time.sleep(PROBE_INTERVAL_SECONDS)


@app.on_event("startup")
def startup():
    if PROBE_INTERVAL_SECONDS > 0:
        t = threading.Thread(target=_background_loop, daemon=True)
        t.start()


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/probe")
def probe():
    return run_probes()


@app.get("/last")
def last():
    with _lock:
        return dict(_last_results)

import json
import os
from pathlib import Path


def get_secret(
    name: str,
    *,
    local_json_path: str = ".secrets/secrets.json",
    gcp_project_id: str | None = None,
    gcp_secret_id: str | None = None,
) -> str | None:
    # 1) env var wins
    val = os.getenv(name)
    if val:
        return val

    # 2) local file (dev only)
    p = Path(local_json_path)
    if p.exists():
        data = json.loads(p.read_text())
        if name in data and data[name]:
            return str(data[name])

    # 3) Secret Manager (cloud)
    if gcp_project_id and gcp_secret_id:
        try:
            from google.cloud import secretmanager
        except ImportError:
            return None  # allow running without GCP libs locally

        client = secretmanager.SecretManagerServiceClient()
        res_name = f"projects/{gcp_project_id}/secrets/{gcp_secret_id}/versions/latest"
        return client.access_secret_version(request={"name": res_name}).payload.data.decode("utf-8")

    return None

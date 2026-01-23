import time
from pathlib import Path
import os
import json

from sklearn.svm import SVC
import hydra
from loguru import logger
from omegaconf import DictConfig
import torch
import joblib
import wandb
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from google.cloud import storage

from olivetti_faces.secrets_utils import get_secret


def build_svm(kernel: str, C: float, gamma: str) -> SVC:
    return SVC(kernel=kernel, C=C, gamma=gamma, probability=True)


def ensure_wandb_key(cfg):
    if not cfg.logging.use_wandb:
        return

    key = get_secret(
        "WANDB_API_KEY",
        local_json_path=".secrets/secrets.json",
        gcp_project_id=cfg.gcp.project_id,
        gcp_secret_id=cfg.logging.wandb_secret_id,  # e.g. "WANDB_API_KEY"
    )
    if not key:
        raise RuntimeError("WANDB is enabled but WANDB_API_KEY was not found in env, .secrets json, or Secret Manager.")
    os.environ["WANDB_API_KEY"] = key


@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    if cfg.logging.use_wandb:
        ensure_wandb_key(cfg)
        wandb.init(project=cfg.logging.project, config=dict(cfg))
        logger.info("wandb initialized")

    logger.info("Loading data...")

    x_train = torch.load("data/processed/train_images.pt")
    y_train = torch.load("data/processed/train_labels.pt")
    x_test = torch.load("data/processed/test_images.pt")
    y_test = torch.load("data/processed/test_labels.pt")

    x_train = x_train.view(len(x_train), -1).numpy()
    x_test = x_test.view(len(x_test), -1).numpy()

    t0 = time.perf_counter()
    pca = PCA(n_components=cfg.pca.n_components, random_state=cfg.seed)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    pca_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    model = build_svm(**cfg.model)
    model.fit(x_train, y_train.numpy())
    train_time = time.perf_counter() - t1

    acc = accuracy_score(y_test.numpy(), model.predict(x_test))

    logger.info(f"Accuracy: {acc}")
    from datetime import datetime, UTC

    # Save locally once
    Path("models").mkdir(exist_ok=True)
    local_model_path = "models/svm.pkl"
    joblib.dump({"model": model, "pca": pca}, local_model_path)
    logger.info(f"Saved model bundle to {local_model_path}")

    VERSION = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    BUCKET_NAME = "mlops-72-bucket"  # <-- no slash
    BUCKET_PREFIX = "models"  # <-- "folder" inside bucket
    MODEL_NAME = "svm-face"

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    base_prefix = f"{BUCKET_PREFIX}/{MODEL_NAME}/versions/{VERSION}"
    gcs_model_blob = f"{base_prefix}/svm.pkl"

    logger.info(f"Uploading to gs://{BUCKET_NAME}/{gcs_model_blob} ...")
    bucket.blob(gcs_model_blob).upload_from_filename(local_model_path)

    latest = {"version": VERSION, "bundle": gcs_model_blob}
    bucket.blob(f"{BUCKET_PREFIX}/{MODEL_NAME}/LATEST.json").upload_from_string(
        json.dumps(latest, indent=2),
        content_type="application/json",
    )

    logger.info(f"Promoted {VERSION} as LATEST")

    if cfg.logging.use_wandb:
        wandb.log(
            {
                "accuracy": acc,
                "pca_time_sec": pca_time,
                "train_time_sec": train_time,
            }
        )
        logger.info("Logging model artifact to wandb...")
        artifact = wandb.Artifact("svm_model", type="model")
        artifact.add_file("models/svm.pkl")
        wandb.log_artifact(artifact)


if __name__ == "__main__":
    train()

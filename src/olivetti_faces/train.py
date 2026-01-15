"""
Training pipeline with PCA + SVM.
Python >= 3.12.3
"""

import time
from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
import joblib
import wandb
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from olivetti_faces.model import build_svm


@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    if cfg.logging.use_wandb:
        wandb.init(project=cfg.logging.project, config=dict(cfg))

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

    if cfg.logging.use_wandb:
        wandb.log({
            "accuracy": acc,
            "pca_time_sec": pca_time,
            "train_time_sec": train_time,
        })

    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "pca": pca}, "models/svm.pkl")


if __name__ == "__main__":
    train()





"""from olivetti_faces.model import Model
from olivetti_faces.data import MyDataset

def train():
    dataset = MyDataset("data/raw")
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
"""

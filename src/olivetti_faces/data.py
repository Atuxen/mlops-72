"""
Dataset download and preprocessing.
Python >= 3.12.3
"""

import typer
from pathlib import Path
import shutil
import kagglehub
import numpy as np
import torch

app = typer.Typer()


@app.command()
def fetch_dataset(
    dataset: str = "martininf1n1ty/olivetti-faces-augmented-dataset",
    output_dir: Path = Path("data/external"),
) -> None:
    cache = Path(kagglehub.dataset_download(dataset))
    target = output_dir / dataset.split("/")[-1]
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        shutil.rmtree(target)

    shutil.copytree(cache, target)


@app.command()
def preprocess(
    train_frac: float = 0.7,
    test_frac: float = 0.2,
    drift_frac: float = 0.1,
    seed: int = 42,
) -> None:
    x = np.load("data/external/olivetti-faces-augmented-dataset/augmented_faces.npy")
    y = np.load("data/external/olivetti-faces-augmented-dataset/augmented_labels.npy")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))

    n_train = int(len(x) * train_frac)
    n_test = int(len(x) * test_frac)

    splits = {
        "train": idx[:n_train],
        "test": idx[n_train:n_train + n_test],
        "drift": idx[n_train + n_test:],
    }

    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)

    for name, ids in splits.items():
        images = torch.from_numpy(x[ids]).float().unsqueeze(1) / 255.0
        labels = torch.from_numpy(y[ids]).long()
        torch.save(images, out / f"{name}_images.pt")
        torch.save(labels, out / f"{name}_labels.pt")


if __name__ == "__main__":
    app()


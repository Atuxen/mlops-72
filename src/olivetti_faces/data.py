
import typer
from pathlib import Path
import shutil
import kagglehub
import torch
import numpy as np
import json


app = typer.Typer()

@app.command()
def fetch_kaggle_dataset(
    dataset: str = "martininf1n1ty/olivetti-faces-augmented-dataset",
    output_dir: Path = Path("data/external"),
):
    """
    Download a Kaggle dataset and copy it into data/external.
    """
    typer.echo("Downloading dataset (via kagglehub cache)...")
    cache_path = Path(kagglehub.dataset_download(dataset))

    dataset_name = dataset.split("/")[-1]
    target = output_dir / dataset_name

    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        typer.echo(f"Removing existing dataset at {target}")
        shutil.rmtree(target)

    shutil.copytree(cache_path, target)

    typer.echo(f"Dataset copied to: {target}")


@app.command()
def preprocess_data(
    external_path_x: str = "data/external/olivetti-faces-augmented-dataset/augmented_faces.npy",
    external_path_y: str = "data/external/olivetti-faces-augmented-dataset/augmented_labels.npy",
    processed_dir: str = "data/processed",
    train_frac: float = 0.7,
    test_frac: float = 0.2,
    drift_frac: float = 0.1,
    seed: int = 42,
) -> None:
    

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # --- load ---
    x = np.load(Path(external_path_x))
    y = np.load(Path(external_path_y))

    if len(x) != len(y):
        raise ValueError(f"x and y length mismatch: {len(x)} vs {len(y)}")

    n = len(x)
    if not np.isclose(train_frac + test_frac + drift_frac, 1.0):
        raise ValueError("train_frac + test_frac + drift_frac must sum to 1.0")

    # --- shuffle indices deterministically ---
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * train_frac)
    n_test = int(n * test_frac)
    n_drift = n - n_train - n_test

    train_idx = idx[:n_train]
    test_idx = idx[n_train:n_train + n_test]
    drift_idx = idx[n_train + n_test:]

    # --- helper to convert ---
    def to_torch(x: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(x)
        # If images are HxW, add channel dim -> Nx1xHxW
        if t.ndim == 3:
            t = t.unsqueeze(1)
        # If already NxHxWxC, convert to NxCxHxW
        elif t.ndim == 4 and t.shape[-1] in (1, 3):
            t = t.permute(0, 3, 1, 2)
        return t.float()

    x_train = to_torch(x[train_idx])
    y_train = torch.from_numpy(y[train_idx]).long()

    x_test = to_torch(x[test_idx])
    y_test = torch.from_numpy(y[test_idx]).long()

    x_drift = to_torch(x[drift_idx])
    y_drift = torch.from_numpy(y[drift_idx]).long()

    # --- normalization (simple) ---
    if x_train.max() > 1.0:
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        x_drift = x_drift / 255.0

    # --- save tensors ---
    torch.save(x_train, processed_dir / "train_images.pt")
    torch.save(y_train, processed_dir / "train_labels.pt")

    torch.save(x_test, processed_dir / "test_images.pt")
    torch.save(y_test, processed_dir / "test_labels.pt")

    torch.save(x_drift, processed_dir / "drift_images.pt")
    torch.save(y_drift, processed_dir / "drift_labels.pt")

if __name__ == "__main__":
    app()


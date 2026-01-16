import torch
from pathlib import Path
from omegaconf import OmegaConf
from olivetti_faces.train import train


def test_training_pipeline_runs(tmp_path, monkeypatch):
    """
    Unit test for Hydra-based training pipeline.
    Uses monkeypatching to avoid real data and filesystem writes.
    """

    # -------------------------------
    # 1. Dummy tensors (mocked data)
    # -------------------------------
    x_train = torch.randn(10, 1, 64, 64)
    y_train = torch.randint(0, 40, (10,))
    x_test = torch.randn(5, 1, 64, 64)
    y_test = torch.randint(0, 40, (5,))

    def fake_torch_load(path):
        name = Path(path).name
        return {
            "train_images.pt": x_train,
            "train_labels.pt": y_train,
            "test_images.pt": x_test,
            "test_labels.pt": y_test,
        }[name]

    monkeypatch.setattr("torch.load", fake_torch_load)

    # ------------------------------------
    # 2. Safe joblib.dump monkeypatch
    # ------------------------------------
    import joblib

    real_dump = joblib.dump  # SAVE ORIGINAL

    model_path = tmp_path / "svm.pkl"

    def fake_dump(obj, path):
        return real_dump(obj, model_path)

    monkeypatch.setattr("olivetti_faces.train.joblib.dump", fake_dump)

    # ------------------------------------
    # 3. Prevent real directory creation
    # ------------------------------------
    monkeypatch.setattr("pathlib.Path.mkdir", lambda self, exist_ok=True: None)

    # ------------------------------------
    # 4. Minimal Hydra config
    # ------------------------------------
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "pca": {"n_components": 5},
            "model": {"kernel": "linear", "C": 1.0, "gamma": "scale"},
            "logging": {"use_wandb": False, "project": "test_project"},
        }
    )

    # ------------------------------------
    # 5. Run training
    # ------------------------------------
    train(cfg)

    # ------------------------------------
    # 6. Assert model artifact exists
    # ------------------------------------
    assert model_path.exists()

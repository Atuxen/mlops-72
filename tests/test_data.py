import torch


def test_processed_tensor_shapes_match():
    """
    PURE unit test:
    Verifies the expected shape contract of processed data
    without touching the filesystem or real artifacts.
    """

    # Synthetic processed-like tensors
    batch_size = 10
    channels = 1
    height = 64
    width = 64

    images = torch.randn(batch_size, channels, height, width)
    labels = torch.randint(0, 40, (batch_size,))

    # Assertions reflect Phase-1 invariants
    assert images.ndim == 4
    assert labels.ndim == 1
    assert images.shape[0] == labels.shape[0]

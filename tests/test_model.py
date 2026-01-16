import pytest
from olivetti_faces.model import build_svm


def test_svm_build_rbf():
    """SVM should be instantiated with RBF kernel."""
    model = build_svm(kernel="rbf", C=1.0, gamma="scale")
    assert model.kernel == "rbf"
    assert model.C == 1.0
    assert model.gamma == "scale"


def test_invalid_kernel_raises():
    """Passing invalid kernel should raise ValueError."""
    with pytest.raises(ValueError):
        build_svm(kernel="invalid", C=1.0, gamma="scale")

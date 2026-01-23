from kfp import dsl

PROJECT_ID = "mlops-72"
REPO = "mlops-72-registry"


def datadrift_detection(
    drift_detected: dsl.OutputPath[str],
) -> dsl.ContainerOp:
    return dsl.ContainerSpec(
        image=f"europe-west1-docker.pkg.dev/{PROJECT_ID}/{REPO}/datadrift-image:latest",
        command=["bash", "-lc"],
        args=[
            f"""
          set -euo pipefail
          uv run invoke datadrift --out "{drift_detected}"
        """
        ],
    )


@dsl.container_component
def train_model() -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=f"europe-west1-docker.pkg.dev/{PROJECT_ID}/{REPO}/train-image:latest",
        command=["bash", "-lc"],
        args=[
            """
            uv run invoke train
            """
        ],
    )


@dsl.pipeline
def pipeline():
    drift = datadrift_detection()
    with dsl.If(drift.outputs["drift_detected"] == "true"):
        train_model()

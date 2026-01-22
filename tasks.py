import os
from loguru import logger
from kfp import compiler
from invoke import task, Context
from kfp import compiler
from pipelines.training_pipeline import pipeline  # import your pipeline() function


from invoke import Context, task
from pathlib import Path

WINDOWS = os.name == "nt"
PROJECT_NAME = "olivetti_faces"
PYTHON_VERSION = "3.12.3"

# Project commands
@task
def fetch_data(ctx: Context) -> None:
    """Fetch data."""
    ctx.run("uv run data fetch-dataset", echo=True, pty=False)


@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data, split in train, test, drift and save under data/processed."""
    ctx.run("uv run data preprocess", echo=True, pty=False)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run dvc config core.no_scm true", echo=True, pty=False)
    logger.info("Pulling data artifacts with DVC.")
    ctx.run(f"uv run dvc pull", echo=True, pty=False)
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=False)

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)

@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, dockerfile: str, name:str) -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t {name}:latest . -f dockerfiles/{dockerfile} --progress=plain",
        echo=True,
        pty=False,
    )
  
@task
def docker_run(ctx: Context, image: str, port: bool = False) -> None:
    secrets_dir = Path.cwd() / ".secrets"
    port_flag = "-p 8080:8080" if port else ""

    cmd = f"""
docker run --rm {port_flag} \
  -v "{secrets_dir}:/secrets:ro" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/mlops-72-4d552127cb1c.json \
  {image}:latest
""".strip()

    ctx.run(cmd, echo=True, pty=False)

@task
def datadrift(ctx: Context, out: str = "") -> None:
    """
    Run drift detection and optionally write result to `out` file.
    """
    logger.info("Starting data drift detection ...")
    ctx.run("uv run dvc config core.no_scm true", echo=True, pty=False)
    logger.info("Pulling data artifacts with DVC.")
    ctx.run("uv run dvc pull", echo=True, pty=False)

    # If you didn't pass --out, pick a default so the script won't crash
    if not out:
        out = "/tmp/drift.txt"

    # Ensure parent dir exists inside container
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running data drift detection -> writing to {out}")
    ctx.run(
        f'uv run src/{PROJECT_NAME}/datadrift.py --out "{out}"',
        echo=True,
        pty=False,
    )

@task
def submit_build_job(ctx: Context, jobfile: str) -> None:
    """Submit a build job to Cloud Build."""
    ctx.run(
        f"gcloud builds submit --config=cloudbuild/{jobfile} .",
        echo=True,
        pty=False,
    )


@task
def run_custom_job_datadrift(ctx: Context) -> None:
    ctx.run(
        "gcloud ai custom-jobs create "
        "--region=europe-west1 "
        "--display-name=datadrift-monitoring-job "
        "--service-account=service-mlops72@mlops-72.iam.gserviceaccount.com "
        "--worker-pool-spec=machine-type=n1-standard-4,replica-count=1,"
        "container-image-uri=europe-west1-docker.pkg.dev/mlops-72/mlops-72-registry/datadrift-image:latest",
        echo=True,
        pty=False,
    )



@task
def run_custom_job_train(ctx: Context) -> None:
    ctx.run(
        "gcloud ai custom-jobs create "
        "--region=europe-west1 "
        "--display-name=training-job "
        "--service-account=service-mlops72@mlops-72.iam.gserviceaccount.com "
        "--worker-pool-spec=machine-type=n1-standard-4,replica-count=1,"
        "container-image-uri=europe-west1-docker.pkg.dev/mlops-72/mlops-72-registry/train-image:latest",
        echo=True,
        pty=False,
    )


@task
def deploy_model_serving(ctx: Context) -> None:
    """Deploy model server to Cloud Run."""
    ctx.run(
        """
    gcloud run deploy modelserve \
      --project=mlops-72 \
      --region=europe-west1 \
      --image=europe-west1-docker.pkg.dev/mlops-72/mlops-72-registry/modelserve-image:latest \
      --service-account=service-mlops72@mlops-72.iam.gserviceaccount.com \
      --set-env-vars=MODEL_BUCKET=mlops-72-bucket,MODEL_PREFIX=models,MODEL_NAME=svm-face \
      --allow-unauthenticated
    """.strip(),
        echo=True,
        pty=False,
    )


@task
def deploy_public_toggle(ctx: Context, toggle:bool = False) -> None:
    """Deploy public toggle to Cloud Run."""
    if toggle:
        ctx.run(
            "gcloud run services add-iam-policy-binding modelserve "
            "--region=europe-west1 "
            "--project=mlops-72 "
            "--member=\"allUsers\" "
            "--role=\"roles/run.invoker\"",
            echo=True,
            pty=False,
        )
    else: 
        ctx.run(
            "gcloud run services remove-iam-policy-binding modelserve "
            "--region=europe-west1 "
            "--project=mlops-72 "
            "--member=\"allUsers\" "
            "--role=\"roles/run.invoker\"",
            echo=True,
            pty=False,
        )


@task 
def run_local_prometheus(ctx: Context) -> None:
    """Run local Prometheus instance."""
    ctx.run(
        "docker run --rm -p 9090:9090 -v "
        f'"{Path.cwd() / "prometheus.yml"}:/etc/prometheus/prometheus.yml" '
        "prom/prometheus",
        echo=True,
        pty=False,
    )




@task
def compile_pipeline(ctx: Context, output: str = "pipeline.json") -> None:
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=output,
    )
    print(f"Wrote {output}")







@task
def run_vertex_pipeline(
    ctx: Context,
    pipeline_file: str = "pipeline.json",
    pipeline_root: str = "gs://mlops-72-bucket/pipeline-root",
    project: str = "mlops-72",
    region: str = "europe-west1",
    service_account: str = "service-mlops72@mlops-72.iam.gserviceaccount.com",
) -> None:
    """
    Submit a compiled KFP pipeline spec (pipeline.json) to Vertex AI Pipelines using gcloud.

    Usage:
      uv run invoke run-vertex-pipeline
      uv run invoke run-vertex-pipeline --pipeline-file path/to/pipeline.json
    """
    cmd = f"""
gcloud ai pipeline-jobs submit \
  --project={project} \
  --region={region} \
  --file={pipeline_file} \
  --pipeline-root={pipeline_root} \
  --service-account={service_account}
""".strip()

    ctx.run(cmd, echo=True, pty=False)



@task
def submit_pipeline(ctx: Context) -> None:
    """
    Submits pipeline.json to Vertex AI Pipelines via Python SDK.
    Requires ADC:
      gcloud auth application-default login
    """
    ctx.run("uv run python pipeline-submit.py", echo=True, pty=False)

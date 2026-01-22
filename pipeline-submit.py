import google.cloud.aiplatform as aip

PROJECT_ID = "mlops-72"
REGION = "europe-west1"

PIPELINE_BUCKET = "mlops-72-bucket"
PIPELINE_ROOT = f"gs://{PIPELINE_BUCKET}/pipeline-root"

RUNTIME_SERVICE_ACCOUNT = "service-mlops72@mlops-72.iam.gserviceaccount.com"
TEMPLATE_PATH = "pipeline.json"

aip.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=PIPELINE_BUCKET,  # bucket only
)

job = aip.PipelineJob(
    display_name="train-smoke-test",
    template_path=TEMPLATE_PATH,
    pipeline_root=PIPELINE_ROOT,
    parameter_values={
        # MUST match your pipeline inputs; remove if your pipeline has none
        # "project_id": PROJECT_ID,
    },
    enable_caching=False,
)

job.submit(service_account=RUNTIME_SERVICE_ACCOUNT)
print("Submitted:", job.resource_name)

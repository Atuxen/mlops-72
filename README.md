# olivetti_faces

End_to_end_ML_project_olivetti_dataset
# mlops-72
Codebase for project in 02476 Machine Learning Operations group 72

## Project description

### Project goal

In this project, we plan to build an MLOps pipeline for a face recognition system for grey scale images using the [Olivetti faces dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html). The arhitecture of the pipeline flow will be starting from data ingestion followed by training, deployment, monitoring and retraining. If monitoring detects data drift or model drift beyond a prescribed threshold, the pipeline should trigger model retraining and keep record of previous versions using a version control system for both data and model parameters. We plan to ensure reproducibility in the deployment environments using Docker, and deploy the ML model in cloud to mimic the realistic production flow. Finally, we will provide project documentation on essential parts of the setup to run the end-to-end Machine learning pipeline flow.

### Data

The Olivetti faces dataset comprises 400 grayscale images of 40 individuals with varying facial expressions, change in posture and under different lighting conditions. Instead of considering the entire data set for training and test, we will initially use a subset of the dataset of specific postures and conditions. This will allow us to later introduce data drift, by running the inference on the remaining dataset with different conditions.

### Model

We are considering using a support vector machine model or a bayesian logistic regression since the data is limited but complex. Since the dimensionality of the dataset is high, we consider compressing the dataset using PCA independent of what classifier we choose.

Optional: We may also incorporate Neural networks and test their performance for different PCAs by configuring set of different hyperparameters.


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

````




## To-Do's:
### Week 1

- [x] Create a git repository (M5)
- [x] Make sure that all team members have write access to the GitHub repository (M5)
- [x] Create a dedicated environment for your project to keep track of your packages (M2)
- [x] Create the initial file structure using Cookiecutter with an appropriate template (M6)
- [x] Fill out the `data.py` file to download and preprocess the data (M6)
- [x] Add a model to `model.py` and a training procedure to `train.py` and get it running (M6)
- [x] Fill out `requirements.txt` / `requirements_dev.txt` or keep `pyproject.toml` / `uv.lock` up to date (M2 + M6)
- [x] Comply with good coding practices (PEP8) (M7)
- [x] Document essential parts of the code (M7)
- [x] Set up version control for your data (M8)
- [x] Add command line interfaces and project commands where it makes sense (M9)
- [x] Construct one or more Dockerfiles (M10)
- [x] Build the Docker images locally and make sure they work (M10)
- [x] Write one or more configuration files for your experiments (M11)
- [x] Use Hydra to load configurations and manage hyperparameters (M11)
- [x] Use profiling to optimize your code (M12)
- [x] Use logging to log important events (M14)
- [x] Use Weights & Biases to log training progress and metrics (M14)
- [x] Run a hyperparameter optimization sweep (M14)
- [-] Use PyTorch Lightning if applicable (M15)


### Week 2

- [x] Write unit tests for data processing (M16)
- [x] Write unit tests for model construction and training (M16)
- [x] Calculate code coverage (M16)
- [x] Set up continuous integration in GitHub (M17)
- [x] Add caching and multi-OS / Python / PyTorch testing (M17)
- [x] Add a linting step to CI (M17)
- [x] Add pre-commit hooks (M18)
- [x] Add a workflow that triggers when data changes (M19)
- [x] Add a workflow that triggers when the model registry changes (M19)
- [x] Create a GCP bucket for data and connect it to DVC (M21)
- [x] Create a workflow for automatic Docker builds (M21)
- [x] Run model training on GCP (Compute Engine or Vertex AI) (M21)
- [x] Create a FastAPI inference service (M22)
- [x] Deploy the model on GCP (Cloud Functions or Cloud Run) (M23)
- [-] Write API tests and add them to CI (M24)
- [-] Load test the API (M24)
- [-] Create a specialized ML deployment API using ONNX and/or BentoML (M25)
- [-] Create a frontend for the API (M26)


### Week 3

- [x] Test robustness to data drift (M27)
- [-] Collect input-output data from the deployed API (M27)
- [x] Deploy a drift detection service (M27)
- [x] Instrument the API with system metrics (M28)
- [x] Set up cloud monitoring (M28)
- [-] Create alerting in GCP (M28)
- [-] Optimize data loading with distributed pipelines if applicable (M29)
- [-] Optimize training with distributed training if applicable (M30)
- [x] Apply quantization, pruning, and compilation for faster inference (M31)


### Extra

- [-] Write documentation for the application (M32)
- [-] Publish documentation on GitHub Pages (M32)
- [-] Revisit and evaluate the initial project description
- [-] Create an architectural diagram of the MLOps pipeline
- [x] Ensure all team members understand all parts of the project
- [x] Upload all code to GitHub

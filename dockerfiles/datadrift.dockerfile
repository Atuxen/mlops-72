FROM python:3.12-slim

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install uv
RUN pip install uv \
    && uv sync --frozen --no-install-project

# Copy code and configs
COPY src/ src/
COPY configs/ configs/
COPY README.md README.md
COPY LICENSE LICENSE
COPY .dvc/ .dvc/
COPY data.dvc data.dvc
COPY tasks.py tasks.py
COPY pipelines/ pipelines/

# Run your training script
CMD ["uv", "run", "invoke", "datadrift"]



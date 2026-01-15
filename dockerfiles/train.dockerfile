FROM python:3.12-slim

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install uv
RUN pip install uv \
    && uv sync --frozen --no-install-project

# Copy your code and configs
COPY src/ src/
COPY configs/ configs/

# Run your training script
CMD ["uv", "run", "src/olivetti_faces/train.py"]




# Previous builtin
#FROM ghcr.io/astral-sh/uv:python3.12.3-alpine AS base

#COPY uv.lock uv.lock
#COPY pyproject.toml pyproject.toml

#RUN uv sync --frozen --no-install-project

#COPY src src/
#COPY README.md README.md
#COPY LICENSE LICENSE

#RUN uv sync --frozen

#ENTRYPOINT ["uv", "run", "src/olivetti_faces/train.py"]

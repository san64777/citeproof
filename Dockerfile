# citeproof - the verification app (binder + web UI). Ollama runs as a separate service
# (see docker-compose.yml). Built for a CUDA GPU: torch 2.12 from PyPI pulls the CUDA runtime
# libraries, and the host GPU is provided at run time via the NVIDIA Container Toolkit.
#
# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# git: MiniCheck (the primary entailment model) is a git dependency, not on PyPI, so it cannot live
# in the lockfile and is installed separately below. ca-certificates for HTTPS to PyPI/HF.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
# copy (not symlink) installed packages so the venv is self-contained; use the image's Python.
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# 1) Locked dependencies first, so this layer is cached unless pyproject/uv.lock change. The binder
#    extra brings torch + the CUDA libraries; the app extra brings FastAPI/uvicorn/ddgs.
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY src ./src
RUN uv sync --no-dev --extra binder --extra app

# 2) MiniCheck (git-only; cannot be locked). Installed into the same venv.
RUN uv pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"

# 3) The sentence tokenizer the binder needs at run time.
RUN uv run --no-sync python -c "import nltk; nltk.download('punkt_tab')"

# Model weights (HuggingFace cache) live on a mounted volume so they survive container rebuilds and
# are downloaded only once. The app binds all interfaces inside the container and reaches Ollama by
# its compose service name.
ENV HF_HOME=/models/hf \
    CITEPROOF_HOST=0.0.0.0 \
    OLLAMA_HOST=http://ollama:11434

EXPOSE 8417
# Run the venv's Python directly (never `uv run` without --no-sync: a sync would drop MiniCheck).
CMD [".venv/bin/python", "-m", "citeproof.app"]

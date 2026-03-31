# ==============================================================================
# GBM Multi-Modal Clustering Pipeline — Dockerfile
# ==============================================================================
# Multi-stage build:
#   Stage 1 (builder): Compile ANTsPy and other C extensions
#   Stage 2 (runtime): Lean image with only runtime dependencies
#
# Build:  docker compose build
# Run:    docker compose run --rm gbm-pipeline
# ==============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel AS builder

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies for compiling ANTsPy, SimpleITK, and other C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages in a virtual env so we can copy cleanly
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements first (Docker layer caching — deps change rarely)
COPY requirements.txt /tmp/requirements.txt

# Install dependencies with extended timeouts (ANTsPy compiles C++ ~15-30 min)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout 1200 --retries 3 \
    -r /tmp/requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Application setup ──────────────────────────────────────────────────────
WORKDIR /workspace

# Set Python path so `from src.xxx import ...` works
ENV PYTHONPATH=/workspace
# Deterministic hashing for reproducibility
ENV PYTHONHASHSEED=0
# Performance tuning defaults (can override in docker-compose)
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

# Copy project source code (Dataset is bind-mounted at runtime, not copied)
COPY --chown=appuser:appuser src/ /workspace/src/
COPY --chown=appuser:appuser scripts/ /workspace/scripts/
COPY --chown=appuser:appuser config/ /workspace/config/

# Create output directories
RUN mkdir -p /workspace/data /workspace/results/figures /workspace/results/tables \
    && chown -R appuser:appuser /workspace

# Copy entrypoint script
COPY --chown=appuser:appuser docker-entrypoint.sh /workspace/docker-entrypoint.sh
RUN chmod +x /workspace/docker-entrypoint.sh

# Switch to non-root user
USER appuser

# Health check: verify critical imports work
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import torch, radiomics, nibabel, sklearn; print('OK')" || exit 1

ENTRYPOINT ["/workspace/docker-entrypoint.sh"]
CMD ["python", "-m", "scripts.run_pipeline"]

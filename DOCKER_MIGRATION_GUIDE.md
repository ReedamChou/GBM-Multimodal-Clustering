# Docker Migration Guide — GBM Multi-Modal Clustering Pipeline

> **Environment**: Windows 11 → WSL2 (Ubuntu 22.04) → Docker Desktop → NVIDIA CUDA
> **GPU**: NVIDIA RTX 4050, 6 GB VRAM | **CUDA Driver**: 12.x

---

## Table of Contents

1. [Prerequisites & Environment Verification](#1-prerequisites--environment-verification)
2. [Common Failure Points & Solutions](#2-common-failure-points--solutions)
3. [Project Adaptation for Docker](#3-project-adaptation-for-docker)
4. [Step-by-Step Migration Process](#4-step-by-step-migration-process)
5. [Professional Best Practices](#5-professional-best-practices)
6. [Quick Reference Commands](#6-quick-reference-commands)

---

## 1. Prerequisites & Environment Verification

### 1.1 Required Software Stack

| Layer | Component | Minimum Version | Your Status |
|-------|-----------|----------------|-------------|
| **Host OS** | Windows 10/11 (21H2+) | Build 19044+ | ✅ |
| **GPU Driver** | NVIDIA Game Ready / Studio | 525.60+ (for CUDA 12.x) | ✅ |
| **WSL2** | Ubuntu 22.04 | WSL kernel 5.15+ | ✅ |
| **Docker** | Docker Desktop | 4.20+ with WSL2 backend | ✅ |
| **NVIDIA Container Toolkit** | nvidia-container-toolkit | 1.14+ | ✅ |
| **CUDA (inside container)** | nvidia/cuda base image | 12.2.x | ✅ |

### 1.2 Verification Commands (Run in WSL2 Terminal)

```bash
# 1. GPU driver visible from WSL
nvidia-smi
# Should show: RTX 4050, Driver 5xx.xx, CUDA 12.x

# 2. Docker engine running
docker info | grep -i runtime
# Should show: Runtimes: nvidia runc

# 3. GPU accessible inside container
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
# Should show GPU table

# 4. PyTorch GPU test
docker run --rm --gpus all pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime \
  python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
# Should print: CUDA available: True, Device: NVIDIA GeForce RTX 4050 ...

# 5. Docker Compose version
docker compose version
# Should show: v2.x
```

### 1.3 Critical Architecture Understanding

```
┌─────────────────────────────────────────────────┐
│  Windows 11 (Host)                              │
│  NVIDIA Driver 5xx.xx (manages physical GPU)    │
│                                                 │
│  ┌─────────────────────────────────────────┐    │
│  │  WSL2 (Ubuntu 22.04)                    │    │
│  │  /usr/lib/wsl/lib/nvidia-smi  (bridge)  │    │
│  │                                         │    │
│  │  ┌─────────────────────────────────┐    │    │
│  │  │  Docker Container               │    │    │
│  │  │  CUDA Toolkit 12.2 (runtime)    │    │    │
│  │  │  PyTorch 2.2 + cuDNN 8          │    │    │
│  │  │  Your ML Code + Dependencies    │    │    │
│  │  │  ──────────────────────────     │    │    │
│  │  │  /workspace (bind-mounted code) │    │    │
│  │  │  /data      (bind-mounted data) │    │    │
│  │  └─────────────────────────────────┘    │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

**Key insight**: You do NOT install CUDA drivers inside the container. The container
only has the CUDA runtime/toolkit. The host driver is shared via the NVIDIA Container
Toolkit.

---

## 2. Common Failure Points & Solutions

### 2.1 CUDA Version Mismatches

| Problem | Cause | Solution |
|---------|-------|----------|
| `CUDA error: no kernel image` | PyTorch compiled for different CUDA than container | Match PyTorch CUDA version to base image CUDA |
| `nvidia-smi` works but `torch.cuda.is_available()` returns False | CUDA toolkit/runtime mismatch | Use `pytorch/pytorch` official image which bundles correct CUDA |
| `CUDA driver version insufficient` | Host driver too old for container CUDA | Update host NVIDIA driver OR use older CUDA base image |

**Version Compatibility Matrix (your setup)**:

```
Host Driver: 5xx.xx → supports CUDA ≤ 12.x
Base Image:  CUDA 12.1 runtime
PyTorch:     2.2.0+cu121  ← must match base image CUDA
```

> **Rule**: Host driver CUDA version ≥ Container CUDA version. Always.

### 2.2 ANTsPy Installation Failures

ANTsPy is the **most fragile** dependency in this project. It compiles C++ code and
frequently fails in Docker if build tools are missing.

```dockerfile
# Required system packages for ANTsPy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

# Install ANTsPy with extended timeout (compilation takes 15–30 min)
RUN pip install --no-cache-dir antspyx>=0.4 --timeout 1200
```

**If ANTsPy pip install fails**, use conda inside Docker:
```dockerfile
# Alternative: conda-forge install
RUN conda install -c aramislab antspyx -y
```

### 2.3 Memory Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `Killed` / OOM during radiomics | Docker default memory limit | Set `deploy.resources.limits.memory: 16g` in compose |
| WSL2 consuming all RAM | WSL2 default: 50% of host RAM | Create `%USERPROFILE%\.wslconfig` (see below) |
| GPU OOM with PyTorch | 6GB VRAM limit | Limit batch sizes, use `torch.cuda.empty_cache()` |

**WSL2 Memory Configuration** (`C:\Users\Husain\.wslconfig`):
```ini
[wsl2]
memory=12GB
swap=4GB
processors=8
```

### 2.4 File System & Path Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Paths break in container | Windows `\` vs Linux `/` | All config paths use `/` — already handled by `pathlib.Path` |
| Slow file I/O on bind mounts | Cross-filesystem WSL2 ↔ Windows | Store data in WSL2 native fs (`/home/user/...`) |
| Permission denied on output dirs | Container user ≠ host user | Run as same UID or use `user:` in compose |

### 2.5 PyRadiomics / SimpleITK Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `SimpleITK` import fails | Missing shared libs | Install `libgl1-mesa-glx` in container |
| Radiomics extraction hangs | Parallel workers + Docker overhead | Reduce `n_jobs` to 2 inside container |

### 2.6 Networking & Timeout Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `pip install` timeout during build | Large packages + slow connection | Use `--timeout 600` and `--retries 3` |
| Cannot download atlases at runtime | No internet in container | Pre-download atlases in build OR mount them |

---

## 3. Project Adaptation for Docker

### 3.1 Docker-Ready Project Structure

```
Project Reedam/
├── Dockerfile                     ← NEW: multi-stage build
├── docker-compose.yml             ← NEW: GPU + volume orchestration
├── .dockerignore                  ← NEW: exclude Dataset/ from build context
├── docker-entrypoint.sh           ← NEW: container startup script
├── requirements.txt               ← UPDATED: pinned versions + extras
├── config/
│   └── config.yaml                ← UNCHANGED (paths already relative)
├── src/                           ← UNCHANGED
├── scripts/                       ← UNCHANGED
├── data/                          ← MOUNTED as volume
├── results/                       ← MOUNTED as volume
└── Dataset/                       ← MOUNTED as read-only volume
```

### 3.2 Base Image Selection

| Base Image | Size | Use Case |
|-----------|------|----------|
| `nvidia/cuda:12.1.1-devel-ubuntu22.04` | ~4GB | Need to compile CUDA extensions (e.g., ANTsPy) |
| `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel` | ~7GB | PyTorch + CUDA + compile tools (recommended for build stage) |
| `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` | ~4GB | PyTorch + CUDA runtime only (recommended for final stage) |

**Chosen strategy**: Multi-stage build:
- **Stage 1 (builder)**: `devel` image to compile ANTsPy and other C extensions
- **Stage 2 (runtime)**: `runtime` image with only compiled packages copied over

### 3.3 Volume Mount Strategy

```yaml
# Dataset = READ-ONLY (never modify source data)
# data/results = READ-WRITE (pipeline outputs)
volumes:
  - ./Dataset:/workspace/Dataset:ro     # MRI data (read-only!)
  - ./data:/workspace/data              # intermediate outputs
  - ./results:/workspace/results        # final outputs
  - ./config:/workspace/config:ro       # configuration
```

**Why not COPY the Dataset?**
- Dataset is ~50-100+ GB of NIfTI files
- Would bloat the Docker image enormously
- Bind-mount gives native speed and keeps images lean

### 3.4 GPU Utilization Best Practices for 6GB VRAM

```python
# In your code, guard GPU usage:
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Monitor VRAM usage
if torch.cuda.is_available():
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB / "
          f"{torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB")

# Clear cache between heavy operations
torch.cuda.empty_cache()
```

For this project, most computation (PyRadiomics, ANTsPy, scikit-learn) is CPU-bound.
GPU is only needed for optional PyTorch components (VAE, DeepSurv). Keep PyTorch batch
sizes small (≤32) and use `float32` to stay within 6GB VRAM.

---

## 4. Step-by-Step Migration Process

### Phase 1: Prepare (Before Docker)

```
□ Step 1: Pin all dependency versions in requirements.txt
□ Step 2: Verify project runs locally end-to-end (at least steps 1,3-6)
□ Step 3: Create .dockerignore to exclude Dataset/ and __pycache__/
□ Step 4: Ensure all config paths are relative (already done via pathlib)
```

### Phase 2: Build

```
□ Step 5: Create Dockerfile (multi-stage for ANTsPy compilation)
□ Step 6: Create docker-compose.yml with GPU + volume mounts
□ Step 7: Build the image:
          docker compose build
          (expect 15-30 min first time due to ANTsPy compilation)
□ Step 8: Verify build:
          docker compose run --rm gbm-pipeline python -c "
            import torch, radiomics, ants, nibabel, sklearn
            print('All imports OK')
            print(f'CUDA: {torch.cuda.is_available()}')
          "
```

### Phase 3: Test

```
□ Step 9:  Run data verification (fast, no GPU needed):
           docker compose run --rm gbm-pipeline python -m scripts.run_pipeline --step 1
□ Step 10: Run a single patient radiomics extraction to verify SimpleITK/PyRadiomics
□ Step 11: Run full pipeline steps 3-6 with --skip-registration:
           docker compose run --rm gbm-pipeline python -m scripts.run_pipeline --step 3-6 --skip-registration
□ Step 12: Verify GPU inside container:
           docker compose run --rm gbm-pipeline python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Phase 4: Production Run

```
□ Step 13: Run the full pipeline:
           docker compose run --rm gbm-pipeline python -m scripts.run_pipeline --step 1-10
□ Step 14: Check results/ directory for outputs
□ Step 15: Tag and push image to registry (optional, for reproducibility)
```

---

## 5. Professional Best Practices

### 5.1 Reproducibility

| Practice | Implementation |
|----------|---------------|
| **Pin ALL versions** | `torch==2.2.0`, not `torch>=2.1` |
| **Lock CUDA version** | Base image + PyTorch wheel must match |
| **Use `.dockerignore`** | Exclude data, caches, IDE files |
| **Tag images with git hash** | `docker build -t gbm-pipeline:$(git rev-parse --short HEAD) .` |
| **Set random seeds** | Already done in `helpers.py` — sets numpy, random, torch seeds |
| **Use `PYTHONHASHSEED=0`** | Deterministic dict ordering |

### 5.2 Stability

| Practice | Implementation |
|----------|---------------|
| **Health checks** | Verify GPU access on container start |
| **Graceful degradation** | Fall back to CPU if GPU unavailable |
| **Resource limits** | Cap memory/CPU in docker-compose |
| **Non-root user** | Run as `appuser` inside container |
| **Read-only data mounts** | `Dataset/:ro` prevents accidental modification |

### 5.3 Scalability

| Practice | Implementation |
|----------|---------------|
| **Multi-stage builds** | Small runtime image (~5GB vs ~10GB) |
| **Layer caching** | Install deps before copying code (Dockerfile ordering) |
| **Parameterized runs** | Override config via env vars or mounted config |
| **Compose profiles** | Separate profiles for dev vs full-run |

### 5.4 Debugging Inside Container

```bash
# Interactive shell
docker compose run --rm gbm-pipeline bash

# Run with increased verbosity
docker compose run --rm gbm-pipeline python -m scripts.run_pipeline --step 1 2>&1 | tee /workspace/results/pipeline.log

# Check GPU inside container
docker compose run --rm gbm-pipeline nvidia-smi

# Monitor resource usage
docker stats
```

### 5.5 Performance Tuning

```yaml
# In docker-compose.yml — tune for your hardware
environment:
  - OMP_NUM_THREADS=8        # OpenMP threads for numpy/scipy
  - MKL_NUM_THREADS=8        # Intel MKL threads
  - NUMEXPR_MAX_THREADS=8    # numexpr threads
  - JOBLIB_TEMP_FOLDER=/tmp  # fast temp storage for joblib
```

For **ANTsPy registration** (CPU-bound, ~40 hours sequential):
- Set `n_jobs: 4` in config (not 8, to avoid Docker overhead + memory pressure)
- Each registration uses ~2-4 GB RAM, so 4 parallel × 4 GB = 16 GB needed

For **PyRadiomics** (CPU-bound):
- Same guidance — `n_jobs: 4` balances speed and memory
- Docker overhead adds ~5-10% to wall time vs bare metal

---

## 6. Quick Reference Commands

```bash
# ── Build ──────────────────────────────────────────────
docker compose build                              # build image
docker compose build --no-cache                   # rebuild from scratch

# ── Run Pipeline ───────────────────────────────────────
docker compose run --rm gbm-pipeline              # full pipeline (default CMD)
docker compose run --rm gbm-pipeline \
  python -m scripts.run_pipeline --step 1         # specific step

# ── Interactive ────────────────────────────────────────
docker compose run --rm gbm-pipeline bash         # shell access
docker compose run --rm gbm-pipeline python       # Python REPL

# ── GPU Verification ──────────────────────────────────
docker compose run --rm gbm-pipeline nvidia-smi
docker compose run --rm gbm-pipeline python -c \
  "import torch; print(torch.cuda.is_available())"

# ── Monitoring ─────────────────────────────────────────
docker stats                                      # live resource usage
docker compose logs -f                            # follow logs

# ── Cleanup ────────────────────────────────────────────
docker compose down                               # stop containers
docker system prune -f                            # remove dangling images
docker volume prune -f                            # remove unused volumes
```

---

## Troubleshooting Cheat Sheet

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `docker: Error response from daemon: could not select device driver ""` | NVIDIA Container Toolkit not installed | `sudo apt install nvidia-container-toolkit && sudo systemctl restart docker` |
| `RuntimeError: CUDA out of memory` | 6GB VRAM exceeded | Reduce batch size or use CPU for that step |
| `Killed` during ANTsPy build | OOM during compilation | Increase WSL2 memory in `.wslconfig` to 12GB+ |
| `Permission denied` on `/workspace/results/` | UID mismatch | Use `user: "${UID}:${GID}"` in compose or `chmod 777 results/` |
| Pipeline hangs at registration | Too many parallel workers | Reduce `n_jobs` from 4 to 2 in `config.yaml` |
| `ModuleNotFoundError: No module named 'src'` | PYTHONPATH not set | Set `ENV PYTHONPATH=/workspace` in Dockerfile |
| Extremely slow file I/O | Data on Windows fs, not WSL2 native | Move Dataset to `/home/husain/Dataset` in WSL2 |
| `ImportError: libGL.so.1` | Missing OpenGL libs | `apt install libgl1-mesa-glx` in Dockerfile |

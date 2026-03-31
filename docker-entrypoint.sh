#!/bin/bash
# ==============================================================================
# docker-entrypoint.sh — Container startup script
# ==============================================================================
# Performs environment health checks before running the pipeline.
# Usage: Automatically invoked by Docker as ENTRYPOINT.
# ==============================================================================

set -e

echo "============================================================"
echo "  GBM Multi-Modal Clustering Pipeline"
echo "============================================================"

# ── GPU Check ────────────────────────────────────────────────────────────────
echo ""
echo "[CHECK] GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo "UNKNOWN")
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo "?")
    echo "  ✓ GPU detected: ${GPU_NAME} (${GPU_MEM} MiB)"
else
    echo "  ⚠ nvidia-smi not found — running in CPU-only mode"
fi

# ── Python + CUDA Check ─────────────────────────────────────────────────────
echo ""
echo "[CHECK] Python environment..."
python -c "
import sys
print(f'  Python:   {sys.version.split()[0]}')

try:
    import torch
    print(f'  PyTorch:  {torch.__version__}')
    if torch.cuda.is_available():
        print(f'  CUDA:     {torch.version.cuda} (device: {torch.cuda.get_device_name(0)})')
    else:
        print('  CUDA:     Not available (CPU mode)')
except ImportError:
    print('  PyTorch:  Not installed')

try:
    import radiomics
    print(f'  PyRadiomics: {radiomics.__version__}')
except ImportError:
    print('  PyRadiomics: Not installed')

try:
    import ants
    print(f'  ANTsPy:   OK')
except ImportError:
    print('  ANTsPy:   Not installed (registration will fail)')
"

# ── Data Mount Check ─────────────────────────────────────────────────────────
echo ""
echo "[CHECK] Data mounts..."
if [ -d "/workspace/Dataset" ]; then
    UCSF_COUNT=$(find /workspace/Dataset/UCSF/DATA-IMAGE-STRUCTURAL -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    UPENN_COUNT=$(find /workspace/Dataset/UPenn/DATA-IMAGE-STRUCTURAL -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "  ✓ Dataset mounted: UCSF=${UCSF_COUNT} patients, UPenn=${UPENN_COUNT} patients"
else
    echo "  ⚠ Dataset NOT mounted at /workspace/Dataset"
    echo "    Run with: docker compose run --rm gbm-pipeline"
fi

if [ -d "/workspace/config" ]; then
    echo "  ✓ Config mounted"
else
    echo "  ⚠ Config NOT mounted"
fi

echo ""
echo "============================================================"
echo "  Starting: $@"
echo "============================================================"
echo ""

# Execute the command passed to the container
exec "$@"

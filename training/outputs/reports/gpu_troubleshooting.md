# GPU Troubleshooting

The training pipeline requested CUDA, but this environment fell back to CPU because `nvidia-smi` could not communicate with the NVIDIA driver.

## What To Check

1. Confirm the machine actually has an NVIDIA GPU.
2. Run `nvidia-smi` in the same shell where you launch training.
3. If `nvidia-smi` fails, install or repair the NVIDIA driver for the host system.
4. If you are inside WSL, Docker, or a remote container, make sure GPU passthrough is enabled.
5. Verify that the Python environment can see the same CUDA-capable runtime as the host driver.
6. Rerun:

```bash
venv/bin/python training/scripts/run_ucsf_training.py
```

## Expected Healthy State

- `nvidia-smi` returns GPU information successfully
- `training/outputs/logs/gpu_status.json` shows `"selected_device": "cuda"`
- The training summary report shows `GPU available: True`

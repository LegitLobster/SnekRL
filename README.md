# SnekRL

This repo contains the Snake game plus multiple RL training workflows.

## Quick start

### Legacy workflow (stable)
```
C:\Users\Loppe\Documents\snekRL\.venv\Scripts\python.exe C:\Users\Loppe\Documents\snekRL\train_rl.py --live-plot --device cuda
```

### GPU-fast workflow (experimental)
```
C:\Users\Loppe\Documents\snekRL\.venv\Scripts\python.exe C:\Users\Loppe\Documents\snekRL\train_rl_gpu.py --live-plot
```

### Full-GPU env workflow (experimental)
```
C:\Users\Loppe\Documents\snekRL\.venv\Scripts\python.exe C:\Users\Loppe\Documents\snekRL\train_rl_gpu_full.py --live-plot
```

## Notes
- The legacy workflow is kept intact.
- The GPU-fast workflow is a separate script with aggressive defaults for speed (fast food respawn + heavier updates).
- The full-GPU env workflow removes Python per-env loops and long-starts for maximum sim throughput.
- You can override any defaults by passing args to `train_rl_gpu.py`.
- If you want to push changes, use the helper script:
```
.\git_snek.ps1 status
.\git_snek.ps1 add .
.\git_snek.ps1 commit -m "message"
.\git_snek.ps1 push
```

## Requirements
- Python 3.12
- PyTorch with CUDA
- Pygame

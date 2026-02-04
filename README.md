# SnekRL

This repo contains the Snake game plus a minimal RL training workflow (barebones DQN) targeting a fixed 12x12 board.

## Quick start

### Barebones DQN (12x12)
```
C:\Users\Loppe\Documents\snekRL\.venv\Scripts\python.exe C:\Users\Loppe\Documents\snekRL\train_simple.py --live-plot --device cuda
```

### GPU DQN (vectorized env)
```
C:\Users\Loppe\Documents\snekRL\.venv\Scripts\python.exe C:\Users\Loppe\Documents\snekRL\train_gpu.py --live-plot --device cuda
```

## Notes
- The minimal trainer is `train_simple.py` (no curriculum, no dueling, no long starts).
- The GPU trainer is `train_gpu.py` (vectorized env on GPU; logging schema matches the simple trainer).
- Checkpoints are separate: `checkpoint.pt` (simple) and `checkpoint_gpu.pt` (GPU).
- You can override any defaults by passing args to `train_simple.py` or `train_gpu.py`.
- If you want to push changes, use the helper script:
```
.\git_snek.ps1 status
.\git_snek.ps1 add .
.\git_snek.ps1 commit -m "message"
.\git_snek.ps1 push
```

## Requirements
- Python 3.12
- PyTorch (CPU or CUDA)
- Pygame

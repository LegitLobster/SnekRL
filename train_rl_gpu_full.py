"""Full-GPU env wrapper (keeps legacy train_rl.py unchanged)."""
import sys
from train_rl import main


def _ensure_arg(args, flag, value=None):
    if flag in args:
        return
    if value is None or value is True:
        args.append(flag)
    else:
        args.extend([flag, str(value)])


def run():
    args = sys.argv[1:]
    _ensure_arg(args, "--device", "cuda")
    _ensure_arg(args, "--full-gpu-env")
    _ensure_arg(args, "--fast-food")
    sys.argv = ["train_rl.py"] + args
    main()


if __name__ == "__main__":
    run()

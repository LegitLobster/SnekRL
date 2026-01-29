"""GPU-fast training preset (keeps legacy train_rl.py unchanged)."""
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
    _ensure_arg(args, "--fast-food")
    _ensure_arg(args, "--n-step", "1")

    _ensure_arg(args, "--n-envs", "256")
    _ensure_arg(args, "--batch-size", "32768")
    _ensure_arg(args, "--gradient-steps", "32")
    _ensure_arg(args, "--train-every", "1")
    _ensure_arg(args, "--buffer-size", "1000000")
    _ensure_arg(args, "--learning-starts", "20000")

    _ensure_arg(args, "--eps-start", "1.0")
    _ensure_arg(args, "--eps-reset", "0.6")
    _ensure_arg(args, "--eps-end", "0.05")
    _ensure_arg(args, "--eps-decay-steps", "200000")
    _ensure_arg(args, "--eps-decay-type", "exp")
    _ensure_arg(args, "--eps-exp-k", "6")

    _ensure_arg(args, "--success-episodes", "1")
    _ensure_arg(args, "--success-episodes-4x4", "1")
    _ensure_arg(args, "--eval-interval", "10000")

    sys.argv = ["train_rl.py"] + args
    main()


if __name__ == "__main__":
    run()

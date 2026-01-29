import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-starts", type=int, default=100_000)
    parser.add_argument("--train-every", type=int, default=2)
    parser.add_argument("--gradient-steps", type=int, default=8)
    parser.add_argument("--target-update", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eps-decay-steps", type=int, default=300_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-root", type=str, default="rl_out")
    parser.add_argument("--refresh", type=float, default=2.0)
    parser.add_argument("--stop-file", type=str, default="")
    parser.add_argument("--clean-logs", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    run1 = out_root / "zero_on"
    run2 = out_root / "zero_off"
    stop_path = Path(args.stop_file) if args.stop_file else (out_root / "stop.txt")
    if stop_path.exists():
        try:
            stop_path.unlink()
        except OSError:
            pass

    base_cmd = [
        sys.executable,
        str(Path(__file__).with_name("train_rl.py")),
        "--timesteps",
        str(args.timesteps),
        "--n-envs",
        str(args.n_envs),
        "--buffer-size",
        str(args.buffer_size),
        "--batch-size",
        str(args.batch_size),
        "--learning-starts",
        str(args.learning_starts),
        "--train-every",
        str(args.train_every),
        "--gradient-steps",
        str(args.gradient_steps),
        "--target-update",
        str(args.target_update),
        "--lr",
        str(args.lr),
        "--eps-decay-steps",
        str(args.eps_decay_steps),
        "--device",
        args.device,
    ]

    cmd1 = base_cmd + [
        "--out",
        str(run1),
        "--zero-out-on-death",
        "--stop-file",
        str(stop_path),
        "--clean-logs",
    ]
    cmd2 = base_cmd + [
        "--out",
        str(run2),
        "--no-zero-out-on-death",
        "--stop-file",
        str(stop_path),
        "--clean-logs",
    ]
    if args.resume:
        cmd1.append("--resume")
        cmd2.append("--resume")

    p1 = subprocess.Popen(cmd1)
    p2 = subprocess.Popen(cmd2)

    plot_cmd = [
        sys.executable,
        str(Path(__file__).with_name("plot_live.py")),
        "--log",
        str(run1 / "train_log.csv"),
        "--eval",
        str(run1 / "eval_log.csv"),
        "--log2",
        str(run2 / "train_log.csv"),
        "--eval2",
        str(run2 / "eval_log.csv"),
        "--refresh",
        str(args.refresh),
        "--stop-file",
        str(stop_path),
    ]

    plot_proc = subprocess.Popen(plot_cmd)

    p1.wait()
    p2.wait()
    plot_proc.terminate()


if __name__ == "__main__":
    main()

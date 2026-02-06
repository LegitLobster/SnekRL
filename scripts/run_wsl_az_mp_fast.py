import os
import subprocess
import sys
import time


def _win_to_wsl(path: str) -> str:
    path = os.path.abspath(path)
    drive, rest = os.path.splitdrive(path)
    if not drive:
        return path.replace("\\", "/")
    drive_letter = drive[0].lower()
    rest = rest.replace("\\", "/")
    if rest.startswith("/"):
        rest = rest[1:]
    return f"/mnt/{drive_letter}/{rest}"


def main() -> int:
    distro = os.environ.get("SNEK_WSL_DISTRO", "Ubuntu")
    repo_win = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo_wsl = _win_to_wsl(repo_win)
    venv_wsl = f"{repo_wsl}/.venv_wsl"
    venv_win = os.path.join(repo_win, ".venv_wsl")
    activate_win = os.path.join(venv_win, "bin", "activate")

    # Launch the Windows plotter against the shared CSV logs.
    try:
        plot_cmd = [
            sys.executable,
            os.path.join(repo_win, "plot_live.py"),
            "--log",
            os.path.join(repo_win, "rl_out_az_mp", "train_log.csv"),
            "--eval",
            os.path.join(repo_win, "rl_out_az_mp", "eval_log.csv"),
            "--refresh-ms",
            "1000",
            "--no-stream",
        ]
        subprocess.Popen(plot_cmd, cwd=repo_win)
    except Exception:
        pass

    if not os.path.exists(activate_win):
        print("WSL venv not found. Expected:", activate_win)
        print("Please run the WSL setup steps to create .venv_wsl.")
        return 0

    # Fast-start config: smaller sims/episodes, more training per sample.
    cmd = (
        f"source {venv_wsl}/bin/activate && "
        f"cd {repo_wsl} && "
        "python -u train_az_mp.py "
        "--clean-logs "
        "--timesteps 0 "
        "--grid 12 "
        "--mcts-sims 32 "
        "--c-puct 1.5 "
        "--dirichlet-alpha 0.3 "
        "--dirichlet-eps 0.25 "
        "--temperature 1.0 "
        "--temp-threshold 30 "
        "--max-episode-steps 200 "
        "--selfplay-batch 32 "
        "--workers 8 "
        "--infer-max-batch 2048 "
        "--infer-batch-wait-ms 5 "
        "--replay-size 50000 "
        "--replay-warmup 1000 "
        "--batch-size 512 "
        "--train-steps 16 "
        "--eval-interval 5000 "
        "--eval-episodes 3 "
        "--eval-max-steps 500 "
        "--log-interval 200 "
        "--checkpoint-interval 20000 "
        "--stop-when-solved "
        "--solve-evals 3 "
        "--solve-min-max-len 0 "
        "--amp "
        "--device cuda "
        "--plot-no-stream"
    )

    args = ["wsl.exe", "-d", distro, "--", "bash", "-lc", cmd]
    log_path = os.path.join(repo_win, "rl_out_az_mp", "error.log")
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="ascii") as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} INFO: WSL fast run started\n")
            log_file.flush()
            result = subprocess.run(args, stdout=log_file, stderr=log_file, text=True)
            if result.returncode != 0:
                log_file.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} ERROR: WSL fast command failed with exit code {result.returncode}\n"
                )
                log_file.flush()
    except Exception as exc:
        try:
            with open(log_path, "a", encoding="ascii") as log_file:
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} ERROR: WSL fast runner failed: {exc}\n")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    main()

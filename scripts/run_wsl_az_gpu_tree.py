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
            os.path.join(repo_win, "rl_out_az_gpu", "train_log.csv"),
            "--eval",
            os.path.join(repo_win, "rl_out_az_gpu", "eval_log.csv"),
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

    cmd = (
        f"source {venv_wsl}/bin/activate && "
        f"cd {repo_wsl} && "
        "python -u train_az_gpu_tree.py "
        "--clean-logs "
        "--timesteps 0 "
        "--grid 12 "
        "--mcts-sims 128 "
        "--mcts-max-nodes 1024 "
        "--mcts-max-depth 32 "
        "--c-puct 1.5 "
        "--dirichlet-alpha 0.3 "
        "--dirichlet-eps 0.25 "
        "--temperature 1.0 "
        "--temp-threshold 30 "
        "--max-episode-steps 1000 "
        "--selfplay-batch 256 "
        "--replay-size 50000 "
        "--replay-warmup 5000 "
        "--batch-size 256 "
        "--train-steps 4 "
        "--eval-interval 20000 "
        "--eval-episodes 5 "
        "--eval-max-steps 1000 "
        "--log-interval 200 "
        "--checkpoint-interval 50000 "
        "--stop-when-solved "
        "--solve-evals 3 "
        "--solve-min-max-len 0 "
        "--torch-compile "
        "--compile-mode reduce-overhead "
        "--amp "
        "--device cuda "
        "--plot-no-stream"
    )

    args = ["wsl.exe", "-d", distro, "--", "bash", "-lc", cmd]
    log_path = os.path.join(repo_win, "rl_out_az_gpu", "error.log")
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="ascii") as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} INFO: WSL run started\n")
            log_file.flush()
            result = subprocess.run(args, stdout=log_file, stderr=log_file, text=True)
            if result.returncode != 0:
                log_file.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} ERROR: WSL command failed with exit code {result.returncode}\n"
                )
                log_file.flush()
    except Exception as exc:
        try:
            with open(log_path, "a", encoding="ascii") as log_file:
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} ERROR: WSL runner failed: {exc}\n")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    main()

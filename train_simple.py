import argparse
import csv
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from snek_env import SnekConfig, SnekEnv


@dataclass
class TrainStats:
    ep_rewards: List[float]
    ep_lengths: List[int]
    ep_max_lens: List[int]
    best_train_max_len: int = 0
    best_eval_max_len: int = 0


class QNet(nn.Module):
    def __init__(self, in_ch: int, h: int, w: int, n_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * h * w, 256)
        self.out = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        x = F.relu(self.fc(x))
        return self.out(x)


class QNetLarge(nn.Module):
    def __init__(self, in_ch: int, h: int, w: int, n_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * h * w, 512)
        self.out = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        x = F.relu(self.fc(x))
        return self.out(x)


def build_qnet(model_size: str, in_ch: int, h: int, w: int, n_actions: int):
    if model_size == "large":
        return QNetLarge(in_ch, h, w, n_actions)
    return QNet(in_ch, h, w, n_actions)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.idx] = obs
        self.next_obs[self.idx] = next_obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    return obj


def _move_opt_state_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _save_checkpoint_async(state, path: Path, save_state: dict) -> bool:
    existing = save_state.get("thread")
    if existing is not None and existing.is_alive():
        return False

    def _worker():
        try:
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            torch.save(state, tmp_path)
            tmp_path.replace(path)
            save_state["last_error"] = None
        except Exception as exc:
            save_state["last_error"] = str(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    save_state["thread"] = thread
    thread.start()
    return True


def make_env(grid: int):
    cfg = SnekConfig(grid_w=grid, grid_h=grid)
    return SnekEnv(cfg)


def epsilon_by_step(step: int, start: float, end: float, decay_steps: int):
    if decay_steps <= 0:
        return end
    t = min(1.0, step / float(decay_steps))
    return start + t * (end - start)


def eval_policy(qnet, device, grid, episodes, max_steps):
    env = make_env(grid)
    qnet.eval()
    total_reward = 0.0
    max_len = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        ep_reward = 0.0
        ep_max_len = 0
        while not done and steps < max_steps:
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                q = qnet(obs_t)
                action = int(q.argmax(dim=1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_max_len = max(ep_max_len, info.get("length", 0))
            steps += 1
        total_reward += ep_reward
        max_len = max(max_len, ep_max_len)
    qnet.train()
    mean_reward = total_reward / max(1, episodes)
    return max_len, mean_reward


def ensure_csv(path: Path, header: List[str], clean: bool):
    if clean and path.exists():
        path.unlink()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="ascii") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def _log_error(path: Path, message: str):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="ascii") as f:
            f.write(f"{ts} ERROR: {message}\n")
    except Exception:
        pass


def write_row(path: Path, row: dict, header: List[str], error_log: Optional[Path] = None):
    try:
        with path.open("a", newline="", encoding="ascii") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writerow(row)
    except Exception as exc:
        if error_log is not None:
            _log_error(error_log, f"write_row failed for {path}: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=12)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update", type=int, default=10_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=200_000)
    parser.add_argument("--model-size", type=str, default="base", choices=["base", "large"])
    parser.add_argument("--eval-interval", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-max-steps", type=int, default=1_000)
    parser.add_argument("--stop-when-solved", action="store_true")
    parser.add_argument("--solve-evals", type=int, default=3)
    parser.add_argument("--solve-min-max-len", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=2_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clean-logs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--live-plot", action="store_true")
    parser.add_argument("--plot-refresh-ms", type=int, default=1000)
    parser.add_argument("--plot-no-stream", action="store_true")
    parser.add_argument("--stop-file", type=str, default="rl_out/stop.txt")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    envs = [make_env(args.grid) for _ in range(args.n_envs)]
    obs_shape = envs[0].observation_space.shape
    n_actions = envs[0].action_space.n

    qnet = build_qnet(args.model_size, obs_shape[0], obs_shape[1], obs_shape[2], n_actions).to(device)
    target = build_qnet(args.model_size, obs_shape[0], obs_shape[1], obs_shape[2], n_actions).to(device)
    target.load_state_dict(qnet.state_dict())
    optimizer = torch.optim.Adam(qnet.parameters(), lr=args.lr)

    buffer = ReplayBuffer(args.buffer_size, obs_shape)

    out_dir = Path("rl_out")
    ckpt_suffix = "" if args.model_size == "base" else f"_{args.model_size}"
    train_log = out_dir / "train_log.csv"
    eval_log = out_dir / "eval_log.csv"
    error_log = out_dir / "error.log"
    stop_path = Path(args.stop_file) if args.stop_file else None

    train_header = [
        "steps",
        "eps",
        "fps",
        "runtime_sec",
        "mean_reward",
        "mean_len",
        "mean_max_len",
        "best_train_max_len",
        "best_eval_max_len",
        "best_train_rate_per_min",
        "best_eval_rate_per_min",
        "loss",
        "buffer_size",
        "board_size",
        "death_wall_per_k",
        "death_self_per_k",
        "training_started",
    ]
    eval_header = ["steps", "max_len", "mean_eval_reward"]

    ensure_csv(train_log, train_header, args.clean_logs)
    ensure_csv(eval_log, eval_header, args.clean_logs)

    if args.resume:
        ckpt_path = out_dir / f"checkpoint{ckpt_suffix}.pt"
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                qnet.load_state_dict(ckpt.get("qnet", {}))
                target.load_state_dict(ckpt.get("target", {}))
                optimizer.load_state_dict(ckpt.get("opt", {}))
                _move_opt_state_to(optimizer, device)
                start_step = int(ckpt.get("step", 0))
            except Exception:
                start_step = 0
                try:
                    bad_path = ckpt_path.with_suffix(ckpt_path.suffix + ".bad")
                    ckpt_path.replace(bad_path)
                except Exception:
                    pass
                _log_error(error_log, "Failed to load checkpoint; renamed to .bad and starting fresh.")
        else:
            start_step = 0
    else:
        start_step = 0

    # When resuming, treat --timesteps as additional steps to run,
    # so we don't immediately exit if the checkpoint already passed the default limit.
    # If --timesteps <= 0, run until stopped.
    if args.timesteps <= 0:
        target_steps = math.inf
    elif args.resume and start_step > 0:
        target_steps = start_step + args.timesteps
    else:
        target_steps = args.timesteps

    if args.live_plot:
        try:
            import subprocess
            plot_cmd = [
                str(Path(__file__).parent / ".venv" / "Scripts" / "python.exe"),
                str(Path(__file__).parent / "plot_live.py"),
                "--log",
                str(train_log),
                "--eval",
                str(eval_log),
                "--refresh-ms",
                str(args.plot_refresh_ms),
            ]
            if args.plot_no_stream:
                plot_cmd.append("--no-stream")
            subprocess.Popen(plot_cmd, cwd=str(Path(__file__).parent))
        except Exception:
            pass

    stats = TrainStats(ep_rewards=[], ep_lengths=[], ep_max_lens=[])
    save_state = {"thread": None, "last_error": None}
    death_wall_window = 0
    death_self_window = 0
    death_steps_window = 0
    best_train_rate = 0.0
    best_eval_rate = 0.0

    obs = []
    for env in envs:
        o, _ = env.reset()
        obs.append(o)
    obs = np.stack(obs, axis=0)

    cur_ep_rewards = np.zeros(args.n_envs, dtype=np.float32)
    cur_ep_lengths = np.zeros(args.n_envs, dtype=np.int32)
    cur_ep_max_len = np.zeros(args.n_envs, dtype=np.int32)

    start_time = time.time()
    last_log = start_step
    last_eval = start_step
    last_ckpt = start_step
    global_step = start_step
    last_best_time = start_time
    last_best_len = 0
    last_log_time = None
    last_log_steps = start_step
    log_ready = False
    solved_streak = 0
    target_solve_len = args.solve_min_max_len if args.solve_min_max_len > 0 else args.grid * args.grid
    required_solve_evals = max(1, int(args.solve_evals))

    if stop_path is not None:
        try:
            if stop_path.exists():
                stop_path.unlink()
        except OSError as exc:
            _log_error(error_log, f"Failed to remove stop file: {exc}")

    while global_step < target_steps:
        eps = epsilon_by_step(global_step, args.eps_start, args.eps_end, args.eps_decay_steps)

        obs_t = torch.from_numpy(obs).to(device)
        with torch.no_grad():
            q = qnet(obs_t)
        greedy_actions = q.argmax(dim=1).cpu().numpy()
        random_actions = np.random.randint(0, n_actions, size=args.n_envs)
        use_random = np.random.rand(args.n_envs) < eps
        actions = np.where(use_random, random_actions, greedy_actions)

        next_obs = np.zeros_like(obs)
        for i, env in enumerate(envs):
            o2, r, terminated, truncated, info = env.step(int(actions[i]))
            done = terminated or truncated
            next_obs[i] = o2
            buffer.add(obs[i], actions[i], r, o2, done)

            cur_ep_rewards[i] += r
            cur_ep_lengths[i] += 1
            cur_ep_max_len[i] = max(cur_ep_max_len[i], int(info.get("length", 0)))

            if done:
                stats.ep_rewards.append(float(cur_ep_rewards[i]))
                stats.ep_lengths.append(int(cur_ep_lengths[i]))
                stats.ep_max_lens.append(int(cur_ep_max_len[i]))
                stats.best_train_max_len = max(stats.best_train_max_len, int(cur_ep_max_len[i]))
                if info.get("death") == "wall":
                    death_wall_window += 1
                if info.get("death") == "self":
                    death_self_window += 1
                cur_ep_rewards[i] = 0.0
                cur_ep_lengths[i] = 0
                cur_ep_max_len[i] = 0
                o2, _ = env.reset()
                next_obs[i] = o2
            death_steps_window += 1

        obs = next_obs
        global_step += args.n_envs

        if not log_ready and buffer.size >= args.buffer_size:
            log_ready = True
            last_log = global_step
            last_eval = global_step
            last_log_steps = global_step
            last_log_time = time.time()

        if buffer.size >= args.learning_starts and (global_step % args.train_every == 0):
            for _ in range(args.gradient_steps):
                b_obs, b_act, b_rew, b_next, b_done = buffer.sample(args.batch_size)
                b_obs_t = torch.from_numpy(b_obs).to(device)
                b_act_t = torch.from_numpy(b_act).to(device)
                b_rew_t = torch.from_numpy(b_rew).to(device)
                b_next_t = torch.from_numpy(b_next).to(device)
                b_done_t = torch.from_numpy(b_done).to(device)

                q_values = qnet(b_obs_t).gather(1, b_act_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target(b_next_t).max(1)[0]
                    target_q = b_rew_t + args.gamma * (1.0 - b_done_t) * next_q
                loss = F.smooth_l1_loss(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            loss = torch.tensor(0.0)

        if global_step % args.target_update == 0:
            target.load_state_dict(qnet.state_dict())

        training_started = buffer.size >= args.learning_starts
        if log_ready and training_started and (global_step - last_eval >= args.eval_interval):
            try:
                max_len, mean_eval_reward = eval_policy(
                    qnet,
                    device,
                    args.grid,
                    args.eval_episodes,
                    args.eval_max_steps,
                )
                stats.best_eval_max_len = max(stats.best_eval_max_len, int(max_len))
                write_row(
                    eval_log,
                    {"steps": global_step, "max_len": max_len, "mean_eval_reward": mean_eval_reward},
                    eval_header,
                    error_log=error_log,
                )
                if args.stop_when_solved:
                    if max_len >= target_solve_len:
                        solved_streak += 1
                    else:
                        solved_streak = 0
                    if solved_streak >= required_solve_evals:
                        break
            except Exception as exc:
                _log_error(error_log, f"eval_policy failed: {exc}")
            last_eval = global_step

        if global_step - last_ckpt >= args.checkpoint_interval:
            ckpt = {
                "qnet": qnet.state_dict(),
                "target": target.state_dict(),
                "opt": optimizer.state_dict(),
                "step": global_step,
            }
            try:
            _save_checkpoint_async(_to_cpu(ckpt), out_dir / f"checkpoint{ckpt_suffix}.pt", save_state)
            except Exception as exc:
                _log_error(error_log, f"save_checkpoint failed: {exc}")
            last_ckpt = global_step

        if log_ready and training_started and (global_step - last_log >= args.log_interval):
            now = time.time()
            runtime_sec = now - start_time
            if last_log_time is None:
                last_log_time = now
                last_log_steps = global_step
            dt = max(1e-6, now - last_log_time)
            dsteps = global_step - last_log_steps
            fps = dsteps / dt

            if stats.ep_rewards:
                mean_reward = float(np.mean(stats.ep_rewards))
                mean_len = float(np.mean(stats.ep_lengths))
                mean_max_len = float(np.mean(stats.ep_max_lens))
            else:
                mean_reward = 0.0
                mean_len = 0.0
                mean_max_len = 0.0

            if stats.best_train_max_len > last_best_len:
                dt_min = max(1e-6, (now - last_best_time) / 60.0)
                best_train_rate = (stats.best_train_max_len - last_best_len) / dt_min
                last_best_len = stats.best_train_max_len
                last_best_time = now
            if stats.best_eval_max_len > 0:
                best_eval_rate = stats.best_eval_max_len / max(1e-6, runtime_sec / 60.0)

            death_wall_per_k = (death_wall_window / max(1, death_steps_window)) * 1000.0
            death_self_per_k = (death_self_window / max(1, death_steps_window)) * 1000.0
            death_wall_window = 0
            death_self_window = 0
            death_steps_window = 0

            row = {
                "steps": global_step,
                "eps": eps,
                "fps": fps,
                "runtime_sec": runtime_sec,
                "mean_reward": mean_reward,
                "mean_len": mean_len,
                "mean_max_len": mean_max_len,
                "best_train_max_len": stats.best_train_max_len,
                "best_eval_max_len": stats.best_eval_max_len,
                "best_train_rate_per_min": best_train_rate,
                "best_eval_rate_per_min": best_eval_rate,
                "loss": float(loss.item()),
                "buffer_size": buffer.size,
                "board_size": f"{args.grid}x{args.grid}",
                "death_wall_per_k": death_wall_per_k,
                "death_self_per_k": death_self_per_k,
                "training_started": training_started,
            }
            write_row(train_log, row, train_header, error_log=error_log)

            stats.ep_rewards.clear()
            stats.ep_lengths.clear()
            stats.ep_max_lens.clear()
            last_log = global_step
            last_log_time = now
            last_log_steps = global_step

            if stop_path is not None:
                try:
                    if stop_path.exists():
                        break
                except OSError as exc:
                    _log_error(error_log, f"stop_file check failed: {exc}")

    ckpt = {
        "qnet": qnet.state_dict(),
        "target": target.state_dict(),
        "opt": optimizer.state_dict(),
        "step": global_step,
    }
    try:
        _save_checkpoint_async(_to_cpu(ckpt), out_dir / f"checkpoint{ckpt_suffix}.pt", save_state)
    except Exception as exc:
        _log_error(error_log, f"final checkpoint failed: {exc}")


if __name__ == "__main__":
    main()

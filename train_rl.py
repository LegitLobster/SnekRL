import argparse
import csv
import json
import random
import socket
import select
import subprocess
import sys
import time
import threading
import faulthandler
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from snek_env import SnekConfig, SnekEnv
from snek_env_torch import TorchSnekBatch, TorchSnekBatchFast


class QNet(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
        )
        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, n_actions)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class PrioritizedReplayBuffer:
    def __init__(self, capacity, obs_shape, device, alpha=0.6, eps=1e-6):
        self.capacity = capacity
        self.device = device
        self.idx = 0
        self.full = False
        self.alpha = alpha
        self.eps = eps

        self.obs = torch.empty((capacity, *obs_shape), device=device, dtype=torch.float32)
        self.next_obs = torch.empty((capacity, *obs_shape), device=device, dtype=torch.float32)
        self.actions = torch.empty((capacity,), device=device, dtype=torch.int64)
        self.rewards = torch.empty((capacity,), device=device, dtype=torch.float32)
        self.dones = torch.empty((capacity,), device=device, dtype=torch.float32)
        self.priorities = torch.zeros((capacity,), device=device, dtype=torch.float32)
        self.is_long = torch.zeros((capacity,), device=device, dtype=torch.bool)

    def add(self, obs, action, reward, next_obs, done, is_long=False):
        self.obs[self.idx].copy_(obs)
        self.next_obs[self.idx].copy_(next_obs)
        self.actions[self.idx] = action
        self.rewards[self.idx] = float(reward)
        self.dones[self.idx] = float(done)
        self.is_long[self.idx] = bool(is_long)
        max_prio = self.priorities.max() if (self.full or self.idx > 0) else torch.tensor(1.0, device=self.device)
        self.priorities[self.idx] = max_prio

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def add_batch(self, obs, actions, rewards, next_obs, dones, is_long):
        if obs.numel() == 0:
            return
        n = obs.shape[0]
        idxs = (torch.arange(n, device=self.device) + self.idx) % self.capacity
        self.obs[idxs].copy_(obs)
        self.next_obs[idxs].copy_(next_obs)
        self.actions[idxs] = actions.to(self.actions.dtype)
        self.rewards[idxs] = rewards.to(self.rewards.dtype)
        self.dones[idxs] = dones.to(self.dones.dtype)
        if torch.is_tensor(is_long):
            self.is_long[idxs] = is_long.to(self.is_long.dtype)
        else:
            self.is_long[idxs] = torch.as_tensor(is_long, device=self.device, dtype=self.is_long.dtype)
        max_prio = self.priorities.max() if (self.full or self.idx > 0) else torch.tensor(1.0, device=self.device)
        self.priorities[idxs] = max_prio

        new_idx = self.idx + n
        if new_idx >= self.capacity:
            self.full = True
        self.idx = new_idx % self.capacity

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch_size, beta=0.4):
        max_idx = self.capacity if self.full else self.idx
        if max_idx == 0:
            raise ValueError("Cannot sample from empty buffer.")
        prios = self.priorities[:max_idx].clamp_min(self.eps).pow(self.alpha)
        probs = prios / prios.sum()
        idxs = torch.multinomial(probs, batch_size, replacement=True)
        weights = (max_idx * probs[idxs]) ** (-beta)
        weights = weights / weights.max()
        weights_t = weights.to(torch.float32)
        return (
            self.obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_obs[idxs],
            self.dones[idxs],
            weights_t,
            idxs,
            self.is_long[idxs],
        )

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "idx": self.idx,
            "full": self.full,
            "obs": self.obs,
            "next_obs": self.next_obs,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "priorities": self.priorities.detach().cpu(),
            "is_long": self.is_long.detach().cpu(),
            "alpha": self.alpha,
            "eps": self.eps,
        }

    def load_state_dict(self, state):
        self.idx = int(state["idx"])
        self.full = bool(state["full"])
        self.obs.copy_(state["obs"])
        self.next_obs.copy_(state["next_obs"])
        self.actions.copy_(state["actions"])
        self.rewards.copy_(state["rewards"])
        self.dones.copy_(state["dones"])
        self.priorities = torch.as_tensor(state["priorities"], device=self.device, dtype=torch.float32)
        if "is_long" in state:
            self.is_long = torch.as_tensor(state["is_long"], device=self.device, dtype=torch.bool)
        self.alpha = float(state.get("alpha", self.alpha))
        self.eps = float(state.get("eps", self.eps))

    def update_priorities(self, idxs, priorities):
        if torch.is_tensor(idxs):
            idxs_t = idxs.to(self.device)
            prios_t = torch.as_tensor(priorities, device=self.device, dtype=torch.float32)
            self.priorities[idxs_t] = prios_t
            return
        for i, prio in zip(idxs, priorities):
            self.priorities[int(i)] = float(prio)

    def reset_priorities(self, value=1.0):
        max_idx = self.capacity if self.full else self.idx
        if max_idx > 0:
            self.priorities[:max_idx].fill_(float(value))


def _neighbors(pos, w, h, blocked):
    x, y = pos
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if nx < 0 or ny < 0 or nx >= w or ny >= h:
            continue
        if (nx, ny) in blocked:
            continue
        out.append((nx, ny))
    random.shuffle(out)
    return out


def _random_snake(config, min_len, max_len, max_tries):
    grid_cells = config.grid_w * config.grid_h
    min_len = max(3, min_len)
    max_len = min(max_len, grid_cells)
    if min_len > max_len:
        min_len = max_len
    for _ in range(max_tries):
        target_len = random.randint(min_len, max_len)
        start = (random.randrange(config.grid_w), random.randrange(config.grid_h))
        path = [start]  # tail -> head
        visited = {start}
        stack = [_neighbors(start, config.grid_w, config.grid_h, visited)]
        max_ops = target_len * 20
        ops = 0
        while len(path) < target_len and stack:
            ops += 1
            if ops > max_ops:
                break
            if not stack[-1]:
                stack.pop()
                removed = path.pop()
                visited.remove(removed)
                continue
            nxt = stack[-1].pop()
            if nxt in visited:
                continue
            path.append(nxt)
            visited.add(nxt)
            stack.append(_neighbors(nxt, config.grid_w, config.grid_h, visited))
        if len(path) == target_len:
            snake = list(reversed(path))
            if len(snake) > 1:
                hx, hy = snake[0]
                nx, ny = snake[1]
                direction = (hx - nx, hy - ny)
            else:
                direction = (1, 0)
            return snake, direction
    return None, None


def _clone_config(cfg):
    return SnekConfig(**vars(cfg))


def _sample_board_size(current_size, max_size, decay):
    if max_size <= current_size:
        return current_size
    sizes = list(range(current_size, max_size + 1))
    weights = [decay ** (s - current_size) for s in sizes]
    total = sum(weights)
    if total <= 0:
        return current_size
    r = random.random() * total
    acc = 0.0
    for s, w in zip(sizes, weights):
        acc += w
        if r <= acc:
            return s
    return sizes[-1]


def make_envs(n_envs, config, target_h, target_w, random_start):
    envs = [SnekEnv(_clone_config(config)) for _ in range(n_envs)]
    obs = []
    flags = []
    prob = random_start["prob"]
    min_len = random_start["min_len"]
    max_len = random_start["max_len"]
    max_tries = random_start["max_tries"]
    max_grid = random_start["max_grid"]
    size_decay = random_start["size_decay"]
    for env in envs:
        if prob > 0.0 and random.random() < prob:
            size = _sample_board_size(config.grid_w, max_grid, size_decay)
            env.config.grid_w = size
            env.config.grid_h = size
            grid_cells = size * size
            min_len_i = max(3, int(grid_cells * min_len))
            max_len_i = max(3, int(grid_cells * max_len))
            snake, direction = _random_snake(env.config, min_len_i, max_len_i, max_tries)
            if snake is not None:
                ob, _ = env.reset(
                    options={
                        "snake": snake,
                        "direction": direction,
                        "ensure_reachable_food": True,
                        "food_attempt_limit": 200,
                    }
                )
                flags.append(True)
            else:
                env.config.grid_w = config.grid_w
                env.config.grid_h = config.grid_h
                ob, _ = env.reset()
                flags.append(False)
        else:
            env.config.grid_w = config.grid_w
            env.config.grid_h = config.grid_h
            ob, _ = env.reset()
            flags.append(False)
        obs.append(pad_obs(ob, target_h, target_w))
    return envs, np.stack(obs, axis=0), np.array(flags, dtype=bool)


def pad_obs(obs, target_h, target_w):
    if obs.shape[1] == target_h and obs.shape[2] == target_w:
        return obs
    padded = np.zeros((obs.shape[0], target_h, target_w), dtype=obs.dtype)
    h = min(target_h, obs.shape[1])
    w = min(target_w, obs.shape[2])
    padded[:, :h, :w] = obs[:, :h, :w]
    return padded


def epsilon_by_step(step, eps_start, eps_end, decay_steps, decay_type="linear", exp_k=5.0):
    if step >= decay_steps:
        return eps_end
    if decay_type == "exp":
        frac = step / decay_steps
        # Fast early decay, slower tail.
        weight = np.exp(-exp_k * frac)
        return eps_end + (eps_start - eps_end) * weight
    frac = step / decay_steps
    return eps_start + (eps_end - eps_start) * frac


def select_actions(qnet, obs, eps, device):
    if torch.is_tensor(obs):
        obs_t = obs
        n = obs_t.shape[0]
        with torch.no_grad():
            q = qnet(obs_t)
        greedy = torch.argmax(q, dim=1)
        if eps >= 1.0:
            return torch.randint(0, 4, (n,), device=obs_t.device)
        if eps <= 0.0:
            return greedy
        rand_mask = torch.rand(n, device=obs_t.device) < eps
        random_actions = torch.randint(0, 4, (n,), device=obs_t.device)
        return torch.where(rand_mask, random_actions, greedy)

    if eps >= 1.0:
        return np.random.randint(0, 4, size=(obs.shape[0],))

    obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
    with torch.no_grad():
        q = qnet(obs_t)
        greedy = torch.argmax(q, dim=1).cpu().numpy()

    if eps <= 0.0:
        return greedy

    rand_mask = np.random.rand(obs.shape[0]) < eps
    random_actions = np.random.randint(0, 4, size=(obs.shape[0],))
    greedy[rand_mask] = random_actions[rand_mask]
    return greedy


def train_step(qnet, target_net, optimizer, batch, gamma):
    obs, actions, rewards, next_obs, dones, weights, idxs, is_long = batch
    q_values = qnet(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_actions = qnet(next_obs).argmax(1)
        next_q = target_net(next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = rewards + gamma * (1.0 - dones) * next_q

    td_error = target - q_values
    per_loss = F.smooth_l1_loss(q_values, target, reduction="none")
    loss = (weights * per_loss).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    long_mask = is_long.to(torch.bool)
    long_loss = None
    if long_mask.any():
        long_loss = (weights[long_mask] * per_loss[long_mask]).mean().item()
    return loss.item(), td_error.detach().abs().cpu().numpy(), idxs, long_loss


def evaluate(
    qnet,
    config,
    device,
    target_h,
    target_w,
    episodes=5,
    max_steps=2000,
    max_seconds=30.0,
    heartbeat_steps=200,
    status_cb=None,
    stop_file=None,
):
    env = SnekEnv(config)
    max_len = 0
    total_reward = 0.0
    episodes_done = 0
    total_steps = 0
    eps_steps = 0
    start_time = time.time()
    timed_out = False
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        while not done:
            if stop_file is not None and stop_file.exists():
                timed_out = True
                break
            obs_t = torch.as_tensor(pad_obs(obs, target_h, target_w), device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(qnet(obs_t), dim=1).item()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1
            total_steps += 1
            if heartbeat_steps and total_steps % heartbeat_steps == 0 and status_cb is not None:
                elapsed = time.time() - start_time
                status_cb(f"eval heartbeat steps={total_steps} elapsed={elapsed:.1f}s")
            if max_seconds and (time.time() - start_time) >= max_seconds:
                timed_out = True
                break
            if steps >= max_steps:
                break
            max_len = max(max_len, info.get("length", 0))
        if timed_out:
            break
        total_reward += ep_reward
        episodes_done += 1
    elapsed = time.time() - start_time
    denom = episodes_done if episodes_done > 0 else 1
    return total_reward / denom, max_len, total_steps, elapsed, timed_out


def spawn_gif(model_path, out_path, grid_size, target_h, target_w, device):
    gif_script = Path(__file__).with_name("make_gif.py")
    if not gif_script.exists():
        return
    subprocess.Popen(
        [
            sys.executable,
            str(gif_script),
            "--model",
            str(model_path),
            "--out",
            str(out_path),
            "--grid",
            str(grid_size),
            "--max-grid",
            str(max(target_h, target_w)),
            "--device",
            "cpu",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument(
        "--until-solved",
        action="store_true",
        default=False,
        help="Ignore timesteps and run until max_grid is solved.",
    )
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--learning-starts", type=int, default=50_000)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=16)
    parser.add_argument("--target-update", type=int, default=10_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-reset", type=float, default=0.6)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=200_000)
    parser.add_argument("--eps-decay-type", type=str, default="exp", choices=["linear", "exp"])
    parser.add_argument("--eps-exp-k", type=float, default=6.0)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--fast-food", action="store_true", help="Use fast GPU food respawn")
    parser.add_argument(
        "--full-gpu-env",
        action="store_true",
        help="Use fully GPU-native env (no Python per-env loops, no long-starts)",
    )
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-steps", type=int, default=1_000_000)
    parser.add_argument("--per-priority-eps", type=float, default=1e-6)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--random-start-prob", type=float, default=0.2)
    parser.add_argument("--random-start-min-frac", type=float, default=0.3)
    parser.add_argument("--random-start-max-frac", type=float, default=0.7)
    parser.add_argument("--random-start-max-tries", type=int, default=200)
    parser.add_argument("--random-start-size-decay", type=float, default=0.5)
    parser.add_argument(
        "--only-long-starts",
        action="store_true",
        default=False,
        help="Train only on randomized long starts (regular starts only if a long start fails to generate).",
    )
    parser.add_argument(
        "--only-regular-starts",
        action="store_true",
        default=False,
        help="Train only on regular starts (disables randomized long starts).",
    )
    parser.add_argument(
        "--eps-decay-scale-by-area",
        action="store_true",
        default=True,
        help="Scale epsilon decay steps by board area relative to min_grid.",
    )
    parser.add_argument(
        "--keep-buffer-on-curriculum",
        action="store_true",
        default=True,
        help="Keep replay buffer when the board size increases.",
    )
    parser.add_argument("--eval-interval", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--max-eval-steps", type=int, default=2000)
    parser.add_argument("--max-eval-seconds", type=float, default=30.0)
    parser.add_argument("--eval-heartbeat-steps", type=int, default=200)
    parser.add_argument("--success-episodes", type=int, default=2)
    parser.add_argument("--success-episodes-4x4", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    default_out = str(Path(__file__).with_name("rl_out"))
    parser.add_argument("--out", type=str, default=default_out)
    parser.add_argument(
        "--zero-out-on-death",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, subtract all accumulated food reward on death.",
    )
    parser.add_argument("--log-interval-sec", type=float, default=5.0)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-on-log", action="store_true", default=False)
    parser.add_argument("--gif-interval", type=int, default=0)
    parser.add_argument("--live-plot", action="store_true")
    parser.add_argument("--plot-refresh-ms", type=int, default=16)
    parser.add_argument("--plot-stream-port", type=int, default=8765)
    parser.add_argument("--plot-stream-interval-sec", type=float, default=0.016)
    parser.add_argument("--stop-file", type=str, default="")
    parser.add_argument("--clean-logs", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--curriculum", action="store_true", default=True)
    parser.add_argument("--min-grid", type=int, default=4)
    parser.add_argument("--max-grid", type=int, default=12)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    error_log = out_dir / "error.log"

    try:
        _main(args, out_dir, error_log)
    except Exception as exc:
        with error_log.open("a", encoding="ascii") as f:
            f.write(f"ERROR: {exc!r}\n")
        raise


def _main(args, out_dir: Path, error_log: Path):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    config = SnekConfig()
    config.zero_out_on_death = args.zero_out_on_death
    if args.curriculum:
        config.grid_w = args.min_grid
        config.grid_h = args.min_grid
    grid_size = config.grid_w * config.grid_h
    max_points = grid_size - 3
    target_h = args.max_grid if args.curriculum else config.grid_h
    target_w = args.max_grid if args.curriculum else config.grid_w
    base_grid = args.min_grid if args.curriculum else config.grid_w
    base_grid_size = base_grid * base_grid

    log_path = out_dir / "train_log.csv"
    eval_log_path = out_dir / "eval_log.csv"
    status_log_path = out_dir / "status.log"
    latest_ckpt = out_dir / "latest.pt"
    latest_full_ckpt = out_dir / "latest_full.pt"
    if args.clean_logs and not args.resume:
        for path in [log_path, eval_log_path, status_log_path, latest_ckpt, latest_full_ckpt, out_dir / "dqn_snek.pt"]:
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
        for gif in out_dir.glob("best_*_len_*_steps_*.gif"):
            try:
                gif.unlink()
            except OSError:
                pass
        for gif in out_dir.glob("win_*_steps_*.gif"):
            try:
                gif.unlink()
            except OSError:
                pass
    stop_file = Path(args.stop_file) if args.stop_file else (out_dir / "stop.txt")
    if stop_file.exists():
        try:
            stop_file.unlink()
        except OSError:
            pass
    if stop_file.exists():
        stop_file = out_dir / f"stop_{int(time.time())}.txt"

    def log_status(message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with status_log_path.open("a", encoding="ascii") as f:
            f.write(f"{timestamp} {message}\n")

    log_status(
        f"startup: entered _main out_dir={out_dir} "
        f"n_envs={args.n_envs} n_step={args.n_step} "
        f"learning_starts={args.learning_starts} train_every={args.train_every} "
        f"batch={args.batch_size} grad_steps={args.gradient_steps}"
    )
    error_log_path = out_dir / "error.log"
    try:
        error_log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    try:
        error_log_path.touch(exist_ok=True)
    except OSError:
        pass
    full_ckpt = None
    if args.resume and latest_full_ckpt.exists():
        full_ckpt = torch.load(latest_full_ckpt, map_location=device)
        cfg = full_ckpt.get("config", {})
        if "grid_w" in cfg:
            config.grid_w = int(cfg["grid_w"])
        if "grid_h" in cfg:
            config.grid_h = int(cfg["grid_h"])
        grid_size = config.grid_w * config.grid_h
        max_points = grid_size - 3

    def load_qnet_weights(model_state):
        if "advantage.weight" in model_state or "value.weight" in model_state:
            qnet.load_state_dict(model_state)
            target_net.load_state_dict(model_state)
            return
        # Back-compat: non-dueling head (fc.3.*) -> dueling head.
        qnet.load_state_dict(model_state, strict=False)
        target_net.load_state_dict(model_state, strict=False)
        if "fc.3.weight" in model_state:
            adv_w = model_state["fc.3.weight"]
            adv_b = model_state.get("fc.3.bias")
            if adv_w.shape == qnet.advantage.weight.data.shape:
                qnet.advantage.weight.data.copy_(adv_w)
                target_net.advantage.weight.data.copy_(adv_w)
                if adv_b is not None:
                    qnet.advantage.bias.data.copy_(adv_b)
                    target_net.advantage.bias.data.copy_(adv_b)
                value_w = adv_w.mean(dim=0, keepdim=True)
                qnet.value.weight.data.copy_(value_w)
                target_net.value.weight.data.copy_(value_w)
                if adv_b is not None:
                    value_b = adv_b.mean().unsqueeze(0)
                    qnet.value.bias.data.copy_(value_b)
                    target_net.value.bias.data.copy_(value_b)

    random_start = {
        "prob": args.random_start_prob,
        "min_len": args.random_start_min_frac,
        "max_len": args.random_start_max_frac,
        "max_tries": args.random_start_max_tries,
        "max_grid": args.max_grid if args.curriculum else config.grid_w,
        "size_decay": args.random_start_size_decay,
    }
    if args.only_regular_starts:
        random_start["prob"] = 0.0
    if args.only_long_starts:
        random_start["prob"] = 1.0
    if args.full_gpu_env:
        env = TorchSnekBatchFast(args.n_envs, config, target_h, device, random_start=random_start)
        env.enable_long_start = random_start["prob"] > 0.0
    else:
        env = TorchSnekBatch(args.n_envs, config, target_h, device, random_start, fast_food=args.fast_food)
        env.enable_long_start = random_start["prob"] > 0.0
    obs = env.reset_all()
    long_start_flags = list(env.long_start_flags)
    log_status("startup: env reset complete")

    qnet = QNet(n_actions=4).to(device)
    target_net = QNet(n_actions=4).to(device)
    target_net.load_state_dict(qnet.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(qnet.parameters(), lr=args.lr)
    buffer_capacity = args.buffer_size
    if full_ckpt is not None:
        buffer_capacity = int(full_ckpt.get("buffer", {}).get("capacity", buffer_capacity))
    buffer = PrioritizedReplayBuffer(
        buffer_capacity,
        obs.shape[1:],
        device=device,
        alpha=args.per_alpha,
        eps=args.per_priority_eps,
    )

    if full_ckpt is not None:
        model_state = full_ckpt.get("qnet", full_ckpt.get("model", qnet.state_dict()))
        load_qnet_weights(model_state)
        if "optimizer" in full_ckpt:
            optimizer.load_state_dict(full_ckpt["optimizer"])
        if "buffer" in full_ckpt:
            buffer.load_state_dict(full_ckpt["buffer"])
    elif args.resume and latest_ckpt.exists():
        state = torch.load(latest_ckpt, map_location=device)
        load_qnet_weights(state)

    episode_rewards = np.zeros(args.n_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.n_envs, dtype=np.int32)
    episode_max_len = np.zeros(args.n_envs, dtype=np.int32)
    recent_rewards = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)
    recent_max_lens = deque(maxlen=100)
    recent_rewards_long = deque(maxlen=100)
    recent_lengths_long = deque(maxlen=100)
    recent_max_lens_long = deque(maxlen=100)
    last_mean_reward = 0.0
    last_mean_len = 0.0
    last_mean_max_len = 0.0
    last_mean_reward_long = 0.0
    last_mean_len_long = 0.0
    last_mean_max_len_long = 0.0

    last_log_time = time.time()
    last_log_steps = 0
    start_time = time.time()
    last_heartbeat = time.time()
    perf_env_ms = 0.0
    perf_replay_ms = 0.0
    perf_train_ms = 0.0
    perf_count = 0
    training_started = False
    fps_started = False
    last_eval_step = 0
    consecutive_success = 0
    total_steps = 0
    eps_steps = 0
    eps_start_current = args.eps_start
    best_eval_max_len = 0
    best_train_max_len = 0
    win_gif_sizes = set()
    best_history = deque(maxlen=120)

    if full_ckpt is not None:
        total_steps = int(full_ckpt.get("total_steps", total_steps))
        eps_steps = int(full_ckpt.get("eps_steps", eps_steps))
        eps_start_current = float(full_ckpt.get("eps_start_current", eps_start_current))
        last_eval_step = int(full_ckpt.get("last_eval_step", last_eval_step))
        consecutive_success = int(full_ckpt.get("consecutive_success", consecutive_success))
        best_eval_max_len = int(full_ckpt.get("best_eval_max_len", best_eval_max_len))
        best_train_max_len = int(full_ckpt.get("best_train_max_len", best_train_max_len))
        win_gif_sizes = set(full_ckpt.get("win_gif_sizes", []))

        saved_rewards = full_ckpt.get("episode_rewards")
        saved_lengths = full_ckpt.get("episode_lengths")
        saved_max_len = full_ckpt.get("episode_max_len")
        if saved_rewards is not None:
            episode_rewards = np.array(saved_rewards, dtype=np.float32)
        if saved_lengths is not None:
            episode_lengths = np.array(saved_lengths, dtype=np.int32)
        if saved_max_len is not None:
            episode_max_len = np.array(saved_max_len, dtype=np.int32)

        recent_rewards = deque(full_ckpt.get("recent_rewards", []), maxlen=100)
        recent_lengths = deque(full_ckpt.get("recent_lengths", []), maxlen=100)
        recent_max_lens = deque(full_ckpt.get("recent_max_lens", []), maxlen=100)
        recent_rewards_long = deque(full_ckpt.get("recent_rewards_long", []), maxlen=100)
        recent_lengths_long = deque(full_ckpt.get("recent_lengths_long", []), maxlen=100)
        recent_max_lens_long = deque(full_ckpt.get("recent_max_lens_long", []), maxlen=100)
        if recent_rewards:
            last_mean_reward = float(np.mean(recent_rewards))
        if recent_lengths:
            last_mean_len = float(np.mean(recent_lengths))
        if recent_max_lens:
            last_mean_max_len = float(np.mean(recent_max_lens))
        if recent_rewards_long:
            last_mean_reward_long = float(np.mean(recent_rewards_long))
        if recent_lengths_long:
            last_mean_len_long = float(np.mean(recent_lengths_long))
        if recent_max_lens_long:
            last_mean_max_len_long = float(np.mean(recent_max_lens_long))

        rng_state = full_ckpt.get("rng", {})
        if "torch" in rng_state:
            torch_state = rng_state["torch"]
            if getattr(torch_state, "dtype", None) == torch.uint8:
                torch_state = torch_state.detach().to("cpu")
                torch.set_rng_state(torch_state)
        if device.type == "cuda" and "cuda" in rng_state:
            try:
                cuda_state = rng_state["cuda"]
                if isinstance(cuda_state, (list, tuple)):
                    cleaned = []
                    for s in cuda_state:
                        if getattr(s, "dtype", None) == torch.uint8:
                            cleaned.append(s.detach().to("cpu"))
                    if cleaned:
                        torch.cuda.set_rng_state_all(cleaned)
            except RuntimeError:
                pass
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "python" in rng_state:
            random.setstate(rng_state["python"])

    plot_proc = None
    plot_sock = None
    last_stream_time = 0.0
    last_metrics = {}
    last_loss_val = None
    last_loss_long_val = None
    stream_last_time = time.time()
    stream_last_steps = total_steps
    best_train_rate = 0.0
    best_eval_rate = 0.0
    last_stream_ok = 0.0
    last_stream_connect = 0.0
    if args.live_plot:
        plot_script = Path(__file__).with_name("plot_live.py")
        if plot_script.exists():
            log_status("startup: launching plot")
            plot_proc = subprocess.Popen(
                [
                    sys.executable,
                    str(plot_script),
                    "--log",
                    str(log_path),
                    "--eval",
                    str(eval_log_path),
                    "--stop-file",
                    str(stop_file),
                    "--refresh-ms",
                    str(args.plot_refresh_ms),
                    "--stream-port",
                    str(args.plot_stream_port),
                    "--display-window",
                    "0",
                    "--display-max-points",
                    "0",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                plot_sock = socket.create_connection(("127.0.0.1", args.plot_stream_port), timeout=0.5)
                plot_sock.setblocking(False)
                plot_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
                plot_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
            except OSError:
                plot_sock = None
            log_status(f"startup: plot launch done sock={'ok' if plot_sock else 'none'}")

    with log_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "steps",
                "eps",
                "fps",
                "runtime_sec",
                "mean_reward",
                "mean_len",
                "mean_max_len",
                "mean_reward_long",
                "mean_len_long",
                "mean_max_len_long",
                "mean_loss_long",
                "best_train_max_len",
                "best_eval_max_len",
                "best_train_rate_per_min",
                "best_eval_rate_per_min",
                "loss",
                "buffer_size",
                "board_size",
            ]
        )

    with eval_log_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["steps", "mean_eval_reward", "max_len", "target_len", "success_streak", "eval_steps", "eval_seconds", "timed_out"]
        )

    def save_checkpoint(full=False):
        torch.save(qnet.state_dict(), latest_ckpt)
        if not full:
            return
        ckpt = {
            "qnet": qnet.state_dict(),
            "target_net": target_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "buffer": buffer.state_dict(),
            "total_steps": total_steps,
            "eps_steps": eps_steps,
            "eps_start_current": eps_start_current,
            "last_eval_step": last_eval_step,
            "consecutive_success": consecutive_success,
            "best_eval_max_len": best_eval_max_len,
            "best_train_max_len": best_train_max_len,
            "win_gif_sizes": list(win_gif_sizes),
            "episode_rewards": episode_rewards.tolist(),
            "episode_lengths": episode_lengths.tolist(),
            "episode_max_len": episode_max_len.tolist(),
            "recent_rewards": list(recent_rewards),
            "recent_lengths": list(recent_lengths),
            "recent_max_lens": list(recent_max_lens),
            "recent_rewards_long": list(recent_rewards_long),
            "recent_lengths_long": list(recent_lengths_long),
            "recent_max_lens_long": list(recent_max_lens_long),
            "config": {
                "grid_w": config.grid_w,
                "grid_h": config.grid_h,
                "zero_out_on_death": config.zero_out_on_death,
            },
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "per": {
                "alpha": args.per_alpha,
                "beta_start": args.per_beta_start,
                "beta_steps": args.per_beta_steps,
                "priority_eps": args.per_priority_eps,
                "n_step": args.n_step,
                "target_tau": args.target_tau,
            },
        }
        if device.type == "cuda":
            ckpt["rng"]["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(ckpt, latest_full_ckpt)

    nstep_buffers = [deque(maxlen=max(1, args.n_step)) for _ in range(args.n_envs)] if args.n_step > 1 else []

    def reset_env_with_random(env):
        if random_start["prob"] > 0.0 and random.random() < random_start["prob"]:
            size = _sample_board_size(config.grid_w, random_start["max_grid"], random_start["size_decay"])
            env.config.grid_w = size
            env.config.grid_h = size
            grid_cells = size * size
            min_len_i = max(3, int(grid_cells * random_start["min_len"]))
            max_len_i = max(3, int(grid_cells * random_start["max_len"]))
            snake, direction = _random_snake(env.config, min_len_i, max_len_i, random_start["max_tries"])
            if snake is not None:
                ob, _ = env.reset(
                    options={
                        "snake": snake,
                        "direction": direction,
                        "ensure_reachable_food": True,
                        "food_attempt_limit": 200,
                    }
                )
                return ob, True
        env.config.grid_w = config.grid_w
        env.config.grid_h = config.grid_h
        ob, _ = env.reset()
        return ob, False

    def make_n_step_transition(buffer_seq):
        reward_sum = 0.0
        gamma_acc = 1.0
        next_obs = buffer_seq[-1][3]
        done = False
        for (reward, next_obs_i, done_i) in [(t[2], t[3], t[4]) for t in buffer_seq]:
            reward_sum += gamma_acc * reward
            gamma_acc *= args.gamma
            next_obs = next_obs_i
            if done_i:
                done = True
                break
        obs0, action0 = buffer_seq[0][0], buffer_seq[0][1]
        return obs0, action0, reward_sum, next_obs, done

    log_status("startup: entering main loop")
    last_progress = {"time": time.time(), "steps": total_steps}

    def _stall_watchdog():
        while True:
            time.sleep(5.0)
            now = time.time()
            if now - last_progress["time"] >= 10.0:
                try:
                    log_status(f"stall: no loop progress for {now - last_progress['time']:.1f}s steps={last_progress['steps']}")
                except OSError:
                    pass
                try:
                    with error_log_path.open("a", encoding="ascii") as ef:
                        ef.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} stall detected\n")
                        faulthandler.dump_traceback(file=ef)
                        ef.write("\n")
                except OSError:
                    pass

    threading.Thread(target=_stall_watchdog, daemon=True).start()
    first_loop = True
    while True:
        last_progress["time"] = time.time()
        last_progress["steps"] = total_steps
        if first_loop:
            log_status("loop: first iteration")
            first_loop = False
        now = time.time()
        if now - last_heartbeat >= 5.0:
            log_status(
                f"heartbeat: steps={total_steps} buffer={len(buffer)} training_started={training_started} "
                f"eps_steps={eps_steps}"
            )
            last_heartbeat = now
        if stop_file.exists():
            print("Stop file detected, exiting gracefully.")
            break
        if not args.until_solved and total_steps >= args.timesteps:
            break
        decay_steps = args.eps_decay_steps
        if base_grid_size > 0:
            decay_steps = int(args.eps_decay_steps * (grid_size / base_grid_size))
        eps = epsilon_by_step(
            eps_steps,
            eps_start_current,
            args.eps_end,
            decay_steps,
            args.eps_decay_type,
            args.eps_exp_k,
        )
        actions = select_actions(qnet, obs, eps, device)
        t_env = time.perf_counter()
        next_obs, rewards_t, dones_t, lengths_t = env.step(actions)
        perf_env_ms += (time.perf_counter() - t_env) * 1000.0

        rewards = rewards_t.detach().cpu().numpy()
        dones = dones_t.detach().cpu().numpy().astype(bool)
        lengths = lengths_t.detach().cpu().numpy()
        actions_cpu = actions.detach().cpu().numpy()

        episode_rewards += rewards
        episode_lengths += 1
        episode_max_len = np.maximum(episode_max_len, lengths)

        done_indices = np.where(dones)[0]

        t_replay = time.perf_counter()
        if args.n_step <= 1:
            buffer.add_batch(
                obs,
                actions,
                rewards_t,
                next_obs,
                dones_t,
                torch.as_tensor(long_start_flags, device=device, dtype=torch.bool),
            )
        else:
            for i in range(args.n_envs):
                obs_i = obs[i]
                next_obs_i = next_obs[i]
                reward = float(rewards[i])
                done = bool(dones[i])

                nstep_buffers[i].append((obs_i, int(actions_cpu[i]), reward, next_obs_i, done))
                if len(nstep_buffers[i]) >= args.n_step:
                    obs0, action0, reward_n, next_obs_n, done_n = make_n_step_transition(nstep_buffers[i])
                    buffer.add(
                        obs0,
                        action0,
                        reward_n,
                        next_obs_n,
                        float(done_n),
                        is_long=long_start_flags[i],
                    )
                if done:
                    while nstep_buffers[i]:
                        obs0, action0, reward_n, next_obs_n, done_n = make_n_step_transition(nstep_buffers[i])
                        buffer.add(
                            obs0,
                            action0,
                            reward_n,
                            next_obs_n,
                            float(done_n),
                            is_long=long_start_flags[i],
                        )
                        nstep_buffers[i].popleft()
        perf_replay_ms += (time.perf_counter() - t_replay) * 1000.0

        for i in done_indices:
            if long_start_flags[i]:
                recent_rewards_long.append(episode_rewards[i])
                recent_lengths_long.append(episode_lengths[i])
                recent_max_lens_long.append(episode_max_len[i])
            else:
                recent_rewards.append(episode_rewards[i])
                recent_lengths.append(episode_lengths[i])
                recent_max_lens.append(episode_max_len[i])
            episode_rewards[i] = 0.0
            episode_lengths[i] = 0
            episode_max_len[i] = 0

        if done_indices.size:
            done_mask = torch.as_tensor(dones, device=device)
            env.reset_mask(done_mask)
            long_start_flags = list(env.long_start_flags)
            obs_reset = env.get_obs()
            next_obs[done_mask] = obs_reset[done_mask]

        obs = next_obs
        total_steps += args.n_envs
        if training_started:
            eps_steps += args.n_envs

        loss_val = None
        loss_long_val = None
        if len(buffer) >= args.learning_starts and (total_steps // args.n_envs) % args.train_every == 0:
            if not training_started:
                training_started = True
                env.enable_long_start = True
            beta = min(
                1.0,
                args.per_beta_start
                + (1.0 - args.per_beta_start) * (total_steps / max(1, args.per_beta_steps)),
            )
            t_train = time.perf_counter()
            for _ in range(args.gradient_steps):
                batch = buffer.sample(args.batch_size, beta=beta)
                loss_val, td_errs, idxs, long_loss = train_step(qnet, target_net, optimizer, batch, args.gamma)
                last_loss_val = loss_val
                if long_loss is not None:
                    loss_long_val = long_loss
                    last_loss_long_val = long_loss
                buffer.update_priorities(idxs, td_errs + buffer.eps)
            perf_train_ms += (time.perf_counter() - t_train) * 1000.0
            if not fps_started:
                # Start FPS tracking only after the first training update completes.
                fps_started = True
                last_log_time = time.time()
                last_log_steps = total_steps
                eps_steps = 0
        perf_count += 1

        if args.target_tau > 0.0:
            with torch.no_grad():
                for tgt, src in zip(target_net.parameters(), qnet.parameters()):
                    tgt.data.mul_(1.0 - args.target_tau).add_(src.data, alpha=args.target_tau)
        elif total_steps % args.target_update < args.n_envs:
            target_net.load_state_dict(qnet.state_dict())

        now = time.time()
        if now - last_log_time >= args.log_interval_sec:
            fps = (total_steps - last_log_steps) / max(1e-6, (now - last_log_time))
            if recent_rewards:
                mean_reward = float(np.mean(recent_rewards))
                last_mean_reward = mean_reward
            else:
                mean_reward = last_mean_reward
            if recent_lengths:
                mean_len = float(np.mean(recent_lengths))
                last_mean_len = mean_len
            else:
                mean_len = last_mean_len
            if recent_max_lens:
                mean_max_len = float(np.mean(recent_max_lens))
                last_mean_max_len = mean_max_len
            else:
                mean_max_len = last_mean_max_len
            if recent_max_lens:
                best_train_max_len = max(best_train_max_len, max(recent_max_lens))
            if recent_rewards_long:
                mean_reward_long = float(np.mean(recent_rewards_long))
                last_mean_reward_long = mean_reward_long
            else:
                mean_reward_long = last_mean_reward_long
            if recent_lengths_long:
                mean_len_long = float(np.mean(recent_lengths_long))
                last_mean_len_long = mean_len_long
            else:
                mean_len_long = last_mean_len_long
            if recent_max_lens_long:
                mean_max_len_long = float(np.mean(recent_max_lens_long))
                last_mean_max_len_long = mean_max_len_long
            else:
                mean_max_len_long = last_mean_max_len_long
            mean_loss_long = loss_long_val if loss_long_val is not None else 0.0
            runtime_sec = now - start_time
            best_history.append((runtime_sec, best_train_max_len, best_eval_max_len))
            best_train_rate = 0.0
            best_eval_rate = 0.0
            if len(best_history) >= 2:
                t0, bt0, be0 = best_history[0]
                t1, bt1, be1 = best_history[-1]
                dt_min = max(1e-6, (t1 - t0) / 60.0)
                best_train_rate = (bt1 - bt0) / dt_min
                best_eval_rate = (be1 - be0) / dt_min
            loss_str = f"{loss_val:.4f}" if loss_val is not None else "n/a"
            avg_env = perf_env_ms / max(1, perf_count)
            avg_replay = perf_replay_ms / max(1, perf_count)
            avg_train = perf_train_ms / max(1, perf_count)
            print(
                f"steps={total_steps} eps={eps:.3f} fps={fps:.0f} "
                f"mean_reward={mean_reward:.2f} mean_len={mean_len:.1f} "
                f"mean_max_len={mean_max_len:.1f} best_train_max_len={best_train_max_len} "
                f"best_eval_max_len={best_eval_max_len} "
                f"env_ms={avg_env:.2f} replay_ms={avg_replay:.2f} train_ms={avg_train:.2f} "
                f"loss={loss_str} buffer={len(buffer)}"
            )
            log_status(
                f"train steps={total_steps} fps={fps:.0f} mean_reward={mean_reward:.2f} "
                f"mean_len={mean_len:.1f} mean_max_len={mean_max_len:.1f} "
                f"mean_reward_long={mean_reward_long:.2f} mean_len_long={mean_len_long:.1f} "
                f"mean_max_len_long={mean_max_len_long:.1f} mean_loss_long={mean_loss_long:.4f} "
                f"best_train_rate={best_train_rate:.2f}/min best_eval_rate={best_eval_rate:.2f}/min "
                f"best_train_max_len={best_train_max_len} best_eval_max_len={best_eval_max_len} "
                f"env_ms={avg_env:.2f} replay_ms={avg_replay:.2f} train_ms={avg_train:.2f} "
                f"loss={loss_str} buffer={len(buffer)} grid={config.grid_w}x{config.grid_h}"
            )
            with log_path.open("a", newline="", encoding="ascii") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        total_steps,
                        f"{eps:.4f}",
                        f"{fps:.2f}",
                        f"{runtime_sec:.2f}",
                        f"{mean_reward:.4f}",
                        f"{mean_len:.2f}",
                        f"{mean_max_len:.2f}",
                        f"{mean_reward_long:.4f}",
                        f"{mean_len_long:.2f}",
                        f"{mean_max_len_long:.2f}",
                        f"{mean_loss_long:.6f}",
                        best_train_max_len,
                        best_eval_max_len,
                        f"{best_train_rate:.4f}",
                        f"{best_eval_rate:.4f}",
                        f"{loss_val:.6f}" if loss_val is not None else "",
                        len(buffer),
                        f"{config.grid_w}x{config.grid_h}",
                    ]
                )
            last_metrics = {
                "steps": int(total_steps),
                "eps": float(f"{eps:.4f}"),
                "fps": float(f"{fps:.2f}"),
                "runtime_sec": float(f"{runtime_sec:.2f}"),
                "mean_reward": float(f"{mean_reward:.4f}"),
                "mean_len": float(f"{mean_len:.2f}"),
                "mean_max_len": float(f"{mean_max_len:.2f}"),
                "mean_reward_long": float(f"{mean_reward_long:.4f}"),
                "mean_len_long": float(f"{mean_len_long:.2f}"),
                "mean_max_len_long": float(f"{mean_max_len_long:.2f}"),
                "mean_loss_long": float(f"{mean_loss_long:.6f}"),
                "best_train_max_len": int(best_train_max_len),
                "best_eval_max_len": int(best_eval_max_len),
                "best_train_rate_per_min": float(f"{best_train_rate:.4f}"),
                "best_eval_rate_per_min": float(f"{best_eval_rate:.4f}"),
                "loss": float(f"{loss_val:.6f}") if loss_val is not None else None,
                "buffer_size": int(len(buffer)),
                "board_size": f"{config.grid_w}x{config.grid_h}",
            }
            if plot_sock is not None:
                try:
                    _, writable, _ = select.select([], [plot_sock], [], 0)
                    if writable:
                        plot_sock.sendall((json.dumps(last_metrics) + "\n").encode("ascii"))
                        last_stream_ok = now
                except (BlockingIOError, InterruptedError):
                    pass
                except OSError:
                    plot_sock = None
            if args.checkpoint_on_log:
                save_checkpoint(full=False)
            last_log_time = now
            last_log_steps = total_steps
            last_stream_time = now
            perf_env_ms = 0.0
            perf_replay_ms = 0.0
            perf_train_ms = 0.0
            perf_count = 0

        if plot_sock is None and args.live_plot and (now - last_stream_connect) >= 1.0:
            try:
                plot_sock = socket.create_connection(("127.0.0.1", args.plot_stream_port), timeout=0.5)
                plot_sock.setblocking(False)
            except OSError:
                plot_sock = None
            last_stream_connect = now

        if plot_sock is not None and (now - last_stream_time) >= args.plot_stream_interval_sec:
            if recent_rewards:
                mean_reward_live = float(np.mean(recent_rewards))
                last_mean_reward = mean_reward_live
            else:
                mean_reward_live = last_mean_reward
            if recent_lengths:
                mean_len_live = float(np.mean(recent_lengths))
                last_mean_len = mean_len_live
            else:
                mean_len_live = last_mean_len
            if recent_max_lens:
                mean_max_len_live = float(np.mean(recent_max_lens))
                last_mean_max_len = mean_max_len_live
            else:
                mean_max_len_live = last_mean_max_len
            if recent_rewards_long:
                mean_reward_long_live = float(np.mean(recent_rewards_long))
                last_mean_reward_long = mean_reward_long_live
            else:
                mean_reward_long_live = last_mean_reward_long
            if recent_lengths_long:
                mean_len_long_live = float(np.mean(recent_lengths_long))
                last_mean_len_long = mean_len_long_live
            else:
                mean_len_long_live = last_mean_len_long
            if recent_max_lens_long:
                mean_max_len_long_live = float(np.mean(recent_max_lens_long))
                last_mean_max_len_long = mean_max_len_long_live
            else:
                mean_max_len_long_live = last_mean_max_len_long

            current_reward = float(np.mean(episode_rewards)) if episode_rewards.size else 0.0
            current_len = float(np.mean(episode_lengths)) if episode_lengths.size else 0.0
            current_max_len = float(np.mean(episode_max_len)) if episode_max_len.size else 0.0
            if any(long_start_flags):
                long_mask = np.asarray(long_start_flags, dtype=bool)
                current_reward_long = float(np.mean(episode_rewards[long_mask]))
                current_len_long = float(np.mean(episode_lengths[long_mask]))
                current_max_len_long = float(np.mean(episode_max_len[long_mask]))
            else:
                current_reward_long = 0.0
                current_len_long = 0.0
                current_max_len_long = 0.0
            mean_loss_long_live = float(last_loss_long_val) if last_loss_long_val is not None else 0.0
            loss_live = float(last_loss_val) if last_loss_val is not None else None
            if recent_max_lens:
                best_train_max_len = max(best_train_max_len, max(recent_max_lens))
            runtime_sec = now - start_time
            best_history.append((runtime_sec, best_train_max_len, best_eval_max_len))
            if len(best_history) >= 2:
                t0, bt0, be0 = best_history[0]
                t1, bt1, be1 = best_history[-1]
                dt_min = max(1e-6, (t1 - t0) / 60.0)
                best_train_rate = (bt1 - bt0) / dt_min
                best_eval_rate = (be1 - be0) / dt_min
            stream_fps = 0.0
            if fps_started:
                dt = now - stream_last_time
                if dt > 1e-6:
                    stream_fps = (total_steps - stream_last_steps) / dt
                    stream_last_time = now
                    stream_last_steps = total_steps
            stream_payload = {
                "steps": int(total_steps),
                "eps": float(f"{eps:.4f}"),
                "fps": float(f"{stream_fps:.2f}"),
                "runtime_sec": float(f"{runtime_sec:.2f}"),
                "training_started": bool(training_started),
                "mean_reward": float(f"{mean_reward_live:.4f}"),
                "mean_len": float(f"{mean_len_live:.2f}"),
                "mean_max_len": float(f"{mean_max_len_live:.2f}"),
                "mean_reward_long": float(f"{mean_reward_long_live:.4f}"),
                "mean_len_long": float(f"{mean_len_long_live:.2f}"),
                "mean_max_len_long": float(f"{mean_max_len_long_live:.2f}"),
                "current_reward": float(f"{current_reward:.4f}"),
                "current_len": float(f"{current_len:.2f}"),
                "current_max_len": float(f"{current_max_len:.2f}"),
                "current_reward_long": float(f"{current_reward_long:.4f}"),
                "current_len_long": float(f"{current_len_long:.2f}"),
                "current_max_len_long": float(f"{current_max_len_long:.2f}"),
                "mean_loss_long": float(f"{mean_loss_long_live:.6f}"),
                "best_train_max_len": int(best_train_max_len),
                "best_eval_max_len": int(best_eval_max_len),
                "best_train_rate_per_min": float(f"{best_train_rate:.4f}"),
                "best_eval_rate_per_min": float(f"{best_eval_rate:.4f}"),
                "loss": float(f"{loss_live:.6f}") if loss_live is not None else None,
                "buffer_size": int(len(buffer)),
                "board_size": f"{config.grid_w}x{config.grid_h}",
            }
            try:
                _, writable, _ = select.select([], [plot_sock], [], 0)
                if writable:
                    plot_sock.sendall((json.dumps(stream_payload) + "\n").encode("ascii"))
                    last_stream_time = now
                    last_stream_ok = now
            except (BlockingIOError, InterruptedError):
                pass
            except OSError:
                plot_sock = None

        if plot_sock is not None and last_stream_ok and (now - last_stream_ok) > 2.0:
            try:
                plot_sock.close()
            except OSError:
                pass
            plot_sock = None


        if total_steps - last_eval_step >= args.eval_interval:
            last_eval_step = total_steps
            log_status(f"eval start steps={total_steps} grid={config.grid_w}x{config.grid_h}")
            mean_eval_reward, max_len, eval_steps, eval_seconds, timed_out = evaluate(
                qnet,
                config,
                device,
                target_h,
                target_w,
                args.eval_episodes,
                args.max_eval_steps,
                args.max_eval_seconds,
                args.eval_heartbeat_steps,
                log_status,
                stop_file,
            )
            log_status(
                f"eval end steps={total_steps} eval_steps={eval_steps} "
                f"eval_seconds={eval_seconds:.1f} max_len={max_len} timed_out={timed_out}"
            )
            required_success = args.success_episodes
            if config.grid_w == 4 and config.grid_h == 4:
                required_success = args.success_episodes_4x4
            if timed_out:
                consecutive_success = 0
            elif max_len >= grid_size:
                consecutive_success += 1
            else:
                consecutive_success = 0
            print(
                f"eval: mean_reward={mean_eval_reward:.2f} max_len={max_len} target_len={grid_size} "
                f"success_streak={consecutive_success}/{required_success}"
            )
            with eval_log_path.open("a", newline="", encoding="ascii") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        total_steps,
                        f"{mean_eval_reward:.4f}",
                        max_len,
                        grid_size,
                        consecutive_success,
                        eval_steps,
                        f"{eval_seconds:.2f}",
                        int(timed_out),
                    ]
                )
            save_checkpoint(full=False)
            if max_len > best_eval_max_len:
                best_eval_max_len = max_len
                best_model_path = out_dir / f"best_eval_len_{best_eval_max_len}_steps_{total_steps}.pt"
                torch.save(qnet.state_dict(), best_model_path)
                spawn_gif(
                    best_model_path,
                    out_dir / f"best_eval_len_{best_eval_max_len}_steps_{total_steps}.gif",
                    config.grid_w,
                    target_h,
                    target_w,
                    device,
                )
            if max_len >= grid_size and config.grid_w not in win_gif_sizes and consecutive_success + 1 >= required_success:
                win_gif_sizes.add(config.grid_w)
                win_model_path = out_dir / f"win_{config.grid_w}x{config.grid_h}_steps_{total_steps}.pt"
                torch.save(qnet.state_dict(), win_model_path)
                spawn_gif(
                    win_model_path,
                    out_dir / f"win_{config.grid_w}x{config.grid_h}_steps_{total_steps}.gif",
                    config.grid_w,
                    target_h,
                    target_w,
                    device,
                )
            if args.curriculum and consecutive_success >= required_success:
                if config.grid_w < args.max_grid:
                    config.grid_w += 1
                    config.grid_h += 1
                    grid_size = config.grid_w * config.grid_h
                    max_points = grid_size - 3
                    best_eval_max_len = 0
                    best_train_max_len = 0
                    consecutive_success = 0
                    eps_steps = 0
                    eps_start_current = args.eps_reset
                    grid_size = config.grid_w * config.grid_h
                    random_start = {
                        "prob": args.random_start_prob,
                        "min_len": args.random_start_min_frac,
                        "max_len": args.random_start_max_frac,
                        "max_tries": args.random_start_max_tries,
                        "max_grid": args.max_grid if args.curriculum else config.grid_w,
                        "size_decay": args.random_start_size_decay,
                    }
                    if args.only_regular_starts:
                        random_start["prob"] = 0.0
                    if args.only_long_starts:
                        random_start["prob"] = 1.0
                    env.update_config(config, random_start)
                    env.enable_long_start = random_start["prob"] > 0.0
                    obs = env.reset_all()
                    long_start_flags = list(env.long_start_flags)
                    if args.keep_buffer_on_curriculum:
                        buffer.reset_priorities(1.0)
                    else:
                        buffer = PrioritizedReplayBuffer(
                            args.buffer_size,
                            obs.shape[1:],
                            device=device,
                            alpha=args.per_alpha,
                            eps=args.per_priority_eps,
                        )
                    nstep_buffers = [deque(maxlen=max(1, args.n_step)) for _ in range(args.n_envs)] if args.n_step > 1 else []
                    episode_rewards = np.zeros(args.n_envs, dtype=np.float32)
                    episode_lengths = np.zeros(args.n_envs, dtype=np.int32)
                    episode_max_len = np.zeros(args.n_envs, dtype=np.int32)
                    recent_rewards.clear()
                    recent_lengths.clear()
                    recent_max_lens.clear()
                    recent_rewards_long.clear()
                    recent_lengths_long.clear()
                    recent_max_lens_long.clear()
                else:
                    log_status("solved: max_grid reached with required streak")
                    print("Solved: reached max grid with required streak.")
                    save_checkpoint(full=True)
                    break
            elif not args.curriculum and consecutive_success >= required_success:
                log_status("solved: required streak reached")
                print("Solved: required streak reached.")
                save_checkpoint(full=True)
                break


        if args.checkpoint_interval > 0 and total_steps % args.checkpoint_interval < args.n_envs:
            save_checkpoint(full=True)


    model_path = out_dir / "dqn_snek.pt"
    torch.save(qnet.state_dict(), model_path)
    save_checkpoint(full=True)
    if plot_proc is not None:
        plot_proc.terminate()
    if plot_sock is not None:
        try:
            plot_sock.close()
        except OSError:
            pass

    print(f"max points possible: {max_points}")


if __name__ == "__main__":
    main()

import argparse
import csv
import math
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from az_game import SnekState, outcome_value
from snek_env import SnekConfig


@dataclass
class TrainStats:
    ep_rewards: List[float]
    ep_lengths: List[int]
    ep_max_lens: List[int]
    best_train_max_len: int = 0
    best_eval_max_len: int = 0


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + x)


class AZNet(nn.Module):
    def __init__(self, in_ch: int, h: int, w: int, n_actions: int, channels: int, blocks: int):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, channels, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * h * w, n_actions)

        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(1 * h * w, channels)
        self.value_fc2 = nn.Linear(channels, 1)

    def forward(self, x):
        x = F.relu(self.conv_in(x))
        x = self.res_blocks(x)

        p = F.relu(self.policy_conv(x))
        p = p.flatten(1)
        p = self.policy_fc(p)

        v = F.relu(self.value_conv(x))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


def build_model(model_size: str, in_ch: int, h: int, w: int, n_actions: int) -> AZNet:
    if model_size == "large":
        return AZNet(in_ch, h, w, n_actions, channels=128, blocks=4)
    return AZNet(in_ch, h, w, n_actions, channels=64, blocks=2)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.data)

    def add_many(self, items: List[Tuple[np.ndarray, np.ndarray, float]]):
        for item in items:
            self.data.append(item)

    def sample(self, batch_size: int, rng: np.random.Generator):
        n = len(self.data)
        if n <= 0:
            raise ValueError("replay buffer is empty")
        replace = n < batch_size
        idx = rng.choice(n, size=batch_size, replace=replace)
        obs = np.stack([self.data[i][0] for i in idx], axis=0)
        policy = np.stack([self.data[i][1] for i in idx], axis=0)
        value = np.array([self.data[i][2] for i in idx], dtype=np.float32)
        return obs, policy, value


class MCTSNode:
    def __init__(self, prior: float):
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, priors: np.ndarray, legal_actions: List[int]):
        for action in legal_actions:
            self.children[action] = MCTSNode(prior=float(priors[action]))


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


def masked_softmax(logits: torch.Tensor, legal_actions: List[int]) -> torch.Tensor:
    mask = torch.full_like(logits, -1e9)
    mask[legal_actions] = 0.0
    return F.softmax(logits + mask, dim=-1)


def policy_value(model: nn.Module, obs: np.ndarray, device: torch.device, legal_actions: List[int]):
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, value = model(obs_t)
    logits = logits.squeeze(0)
    probs = masked_softmax(logits, legal_actions)
    return probs.cpu().numpy(), float(value.item())


def select_child(node: MCTSNode, c_puct: float):
    best_score = -1e9
    best_action = None
    best_child = None
    sqrt_visits = math.sqrt(max(1, node.visit_count))
    for action, child in node.children.items():
        u = c_puct * child.prior * sqrt_visits / (1 + child.visit_count)
        score = child.value + u
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    return best_action, best_child


def add_dirichlet_noise(node: MCTSNode, alpha: float, eps: float, rng: np.random.Generator):
    actions = list(node.children.keys())
    if not actions:
        return
    noise = rng.dirichlet([alpha] * len(actions))
    for action, n in zip(actions, noise):
        child = node.children[action]
        child.prior = child.prior * (1.0 - eps) + n * eps


def run_mcts(
    root_state: SnekState,
    model: nn.Module,
    device: torch.device,
    sims: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    rng: np.random.Generator,
    add_noise: bool,
) -> np.ndarray:
    root = MCTSNode(0.0)
    legal = root_state.legal_actions()
    policy, _value = policy_value(model, root_state.to_obs(), device, legal)
    root.expand(policy, legal)
    if add_noise:
        add_dirichlet_noise(root, dirichlet_alpha, dirichlet_eps, rng)

    for _ in range(sims):
        node = root
        state = root_state.clone()
        search_path = [node]
        done = False

        while node.children:
            action, child = select_child(node, c_puct)
            if child is None:
                break
            _reward, done, _info = state.step(action)
            node = child
            search_path.append(node)
            if done:
                break

        if done:
            value = outcome_value(state)
        else:
            legal = state.legal_actions()
            policy, value = policy_value(model, state.to_obs(), device, legal)
            node.expand(policy, legal)

        for n in search_path:
            n.visit_count += 1
            n.value_sum += value

    counts = np.zeros(4, dtype=np.float32)
    for action, child in root.children.items():
        counts[action] = float(child.visit_count)
    return counts


def select_action_from_policy(policy: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    if temperature <= 0.0:
        return int(policy.argmax())
    adj = np.power(policy, 1.0 / max(1e-6, temperature))
    adj_sum = float(adj.sum())
    if adj_sum <= 0.0:
        return int(policy.argmax())
    probs = adj / adj_sum
    return int(rng.choice(len(policy), p=probs))


def self_play_episode(
    model: nn.Module,
    device: torch.device,
    config: SnekConfig,
    sims: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    temperature: float,
    temp_threshold: int,
    rng: np.random.Generator,
    seed: int,
    max_steps: int,
):
    state = SnekState.new(config, seed=seed)
    episode = []
    ep_reward = 0.0
    ep_len = 0
    ep_max_len = state.length
    death_type = None

    done = False
    while not done and ep_len < max_steps:
        counts = run_mcts(
            state,
            model,
            device,
            sims,
            c_puct,
            dirichlet_alpha,
            dirichlet_eps,
            rng,
            add_noise=True,
        )
        total = float(counts.sum())
        if total > 0:
            policy = counts / total
        else:
            policy = np.full((4,), 0.25, dtype=np.float32)

        episode.append((state.to_obs(), policy))
        temp = temperature if ep_len < temp_threshold else 0.0
        action = select_action_from_policy(policy, temp, rng)
        reward, done, info = state.step(action)
        ep_reward += reward
        ep_len += 1
        ep_max_len = max(ep_max_len, int(info.get("length", state.length)))
        if done:
            death_type = info.get("death")

    outcome = outcome_value(state)
    examples = [(obs, policy, outcome) for obs, policy in episode]
    return examples, ep_reward, ep_len, ep_max_len, death_type


def eval_episode(
    model: nn.Module,
    device: torch.device,
    config: SnekConfig,
    sims: int,
    c_puct: float,
    rng: np.random.Generator,
    seed: int,
    max_steps: int,
):
    state = SnekState.new(config, seed=seed)
    done = False
    steps = 0
    ep_reward = 0.0
    ep_max_len = state.length

    while not done and steps < max_steps:
        counts = run_mcts(
            state,
            model,
            device,
            sims,
            c_puct,
            dirichlet_alpha=0.0,
            dirichlet_eps=0.0,
            rng=rng,
            add_noise=False,
        )
        if counts.sum() <= 0:
            action = 0
        else:
            action = int(counts.argmax())
        reward, done, info = state.step(action)
        ep_reward += reward
        steps += 1
        ep_max_len = max(ep_max_len, int(info.get("length", state.length)))

    return ep_max_len, ep_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=12)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--mcts-sims", type=int, default=64)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-eps", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temp-threshold", type=int, default=30)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--replay-warmup", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--model-size", type=str, default="base", choices=["base", "large"])
    parser.add_argument("--eval-interval", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-max-steps", type=int, default=1_000)
    parser.add_argument("--eval-sims", type=int, default=0)
    parser.add_argument("--stop-when-solved", action="store_true")
    parser.add_argument("--solve-evals", type=int, default=3)
    parser.add_argument("--solve-min-max-len", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=2_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clean-logs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--out-dir", type=str, default="rl_out_az")
    parser.add_argument("--live-plot", action="store_true")
    parser.add_argument("--plot-refresh-ms", type=int, default=1000)
    parser.add_argument("--plot-no-stream", action="store_true")
    parser.add_argument("--stop-file", type=str, default="rl_out_az/stop.txt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    np_rng = np.random.default_rng(args.seed if args.seed > 0 else None)
    py_rng = random.Random(args.seed if args.seed > 0 else None)

    config = SnekConfig(grid_w=args.grid, grid_h=args.grid)

    model = build_model(args.model_size, 4, args.grid, args.grid, 4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    replay = ReplayBuffer(args.replay_size)

    out_dir = Path(args.out_dir)
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

    start_step = 0
    best_eval_max_len = 0
    if args.resume:
        ckpt_path = out_dir / f"checkpoint_az{ckpt_suffix}.pt"
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(ckpt.get("model", {}))
                optimizer.load_state_dict(ckpt.get("opt", {}))
                start_step = int(ckpt.get("step", 0))
                best_eval_max_len = int(ckpt.get("best_eval_max_len", 0))
                if "np_rng" in ckpt:
                    np_rng = np.random.default_rng()
                    np_rng.bit_generator.state = ckpt["np_rng"]
                if "py_rng" in ckpt:
                    py_rng.setstate(ckpt["py_rng"])
            except Exception:
                start_step = 0
                try:
                    bad_path = ckpt_path.with_suffix(ckpt_path.suffix + ".bad")
                    ckpt_path.replace(bad_path)
                except Exception:
                    pass
                _log_error(error_log, "Failed to load checkpoint; renamed to .bad and starting fresh.")

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
        model.eval()
        seed = py_rng.randrange(1 << 30)
        examples, ep_reward, ep_len, ep_max_len, death_type = self_play_episode(
            model,
            device,
            config,
            args.mcts_sims,
            args.c_puct,
            args.dirichlet_alpha,
            args.dirichlet_eps,
            args.temperature,
            args.temp_threshold,
            np_rng,
            seed,
            args.max_episode_steps,
        )

        if ep_len <= 0:
            continue

        replay.add_many(examples)
        global_step += ep_len
        death_steps_window += ep_len

        stats.ep_rewards.append(float(ep_reward))
        stats.ep_lengths.append(int(ep_len))
        stats.ep_max_lens.append(int(ep_max_len))
        stats.best_train_max_len = max(stats.best_train_max_len, int(ep_max_len))
        if death_type == "wall":
            death_wall_window += 1
        elif death_type == "self":
            death_self_window += 1

        if not log_ready and len(replay) >= args.replay_size:
            log_ready = True
            last_log = global_step
            last_eval = global_step
            last_log_steps = global_step
            last_log_time = time.time()

        training_started = len(replay) >= args.replay_warmup
        loss_val = 0.0
        if training_started:
            model.train()
            for _ in range(args.train_steps):
                if len(replay) < args.batch_size:
                    break
                b_obs, b_policy, b_value = replay.sample(args.batch_size, np_rng)
                obs_t = torch.from_numpy(b_obs).to(device)
                policy_t = torch.from_numpy(b_policy).to(device)
                value_t = torch.from_numpy(b_value).to(device)

                logits, value_pred = model(obs_t)
                log_probs = F.log_softmax(logits, dim=1)
                policy_loss = -(policy_t * log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(value_pred.squeeze(1), value_t)
                loss = policy_loss + args.value_loss_weight * value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val = float(loss.item())
        else:
            loss_val = 0.0

        if log_ready and training_started and (global_step - last_eval >= args.eval_interval):
            try:
                eval_sims = args.eval_sims if args.eval_sims > 0 else args.mcts_sims
                eval_max_len = 0
                total_eval_reward = 0.0
                model.eval()
                for _ in range(args.eval_episodes):
                    seed = py_rng.randrange(1 << 30)
                    ep_max_len, ep_reward = eval_episode(
                        model,
                        device,
                        config,
                        eval_sims,
                        args.c_puct,
                        np_rng,
                        seed,
                        args.eval_max_steps,
                    )
                    eval_max_len = max(eval_max_len, int(ep_max_len))
                    total_eval_reward += float(ep_reward)
                mean_eval_reward = total_eval_reward / max(1, args.eval_episodes)
                best_eval_max_len = max(best_eval_max_len, int(eval_max_len))
                stats.best_eval_max_len = best_eval_max_len

                write_row(
                    eval_log,
                    {"steps": global_step, "max_len": eval_max_len, "mean_eval_reward": mean_eval_reward},
                    eval_header,
                    error_log=error_log,
                )
                if args.stop_when_solved:
                    if eval_max_len >= target_solve_len:
                        solved_streak += 1
                    else:
                        solved_streak = 0
                    if solved_streak >= required_solve_evals:
                        break
            except Exception as exc:
                _log_error(error_log, f"eval_episode failed: {exc}")
            last_eval = global_step

        if global_step - last_ckpt >= args.checkpoint_interval:
            ckpt = {
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "step": global_step,
                "best_eval_max_len": best_eval_max_len,
                "np_rng": np_rng.bit_generator.state,
                "py_rng": py_rng.getstate(),
            }
            try:
                _save_checkpoint_async(_to_cpu(ckpt), out_dir / f"checkpoint_az{ckpt_suffix}.pt", save_state)
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
            if best_eval_max_len > 0:
                best_eval_rate = best_eval_max_len / max(1e-6, runtime_sec / 60.0)

            death_wall_per_k = (death_wall_window / max(1, death_steps_window)) * 1000.0
            death_self_per_k = (death_self_window / max(1, death_steps_window)) * 1000.0
            death_wall_window = 0
            death_self_window = 0
            death_steps_window = 0

            row = {
                "steps": global_step,
                "eps": args.temperature,
                "fps": fps,
                "runtime_sec": runtime_sec,
                "mean_reward": mean_reward,
                "mean_len": mean_len,
                "mean_max_len": mean_max_len,
                "best_train_max_len": stats.best_train_max_len,
                "best_eval_max_len": best_eval_max_len,
                "best_train_rate_per_min": best_train_rate,
                "best_eval_rate_per_min": best_eval_rate,
                "loss": loss_val,
                "buffer_size": len(replay),
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
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "step": global_step,
        "best_eval_max_len": best_eval_max_len,
        "np_rng": np_rng.bit_generator.state,
        "py_rng": py_rng.getstate(),
    }
    try:
        _save_checkpoint_async(_to_cpu(ckpt), out_dir / f"checkpoint_az{ckpt_suffix}.pt", save_state)
    except Exception as exc:
        _log_error(error_log, f"final checkpoint failed: {exc}")


if __name__ == "__main__":
    main()

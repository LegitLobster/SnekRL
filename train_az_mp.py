import argparse
import csv
import math
import os
import random
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from multiprocessing.connection import wait as mp_wait
from pathlib import Path
from queue import Empty
from typing import List, Optional, Tuple

import multiprocessing as mp

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


def _write_status(status_path: Path, payload: dict):
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        status_path.parent.mkdir(parents=True, exist_ok=True)
        with status_path.open("a", encoding="ascii") as f:
            f.write(
                f"{ts} steps={payload.get('steps', 0)} replay={payload.get('replay', 0)} "
                f"training_started={payload.get('training_started', False)} "
                f"last_ep_len={payload.get('last_ep_len', 0)} "
                f"last_ep_reward={payload.get('last_ep_reward', 0.0):.3f} "
                f"last_ep_max_len={payload.get('last_ep_max_len', 0)}\n"
            )
    except Exception:
        pass


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


def policy_value_batch(
    model: nn.Module,
    obs_batch: np.ndarray,
    device: torch.device,
    legal_actions_list: List[List[int]],
):
    obs_t = torch.from_numpy(obs_batch).to(device)
    with torch.no_grad():
        logits, values = model(obs_t)
    mask = torch.full_like(logits, -1e9)
    for i, legal in enumerate(legal_actions_list):
        mask[i, legal] = 0.0
    probs = F.softmax(logits + mask, dim=1)
    return probs.detach().cpu().numpy(), values.squeeze(1).detach().cpu().numpy()


def policy_value_batch_torch(
    model: nn.Module,
    obs_batch: np.ndarray,
    device: torch.device,
    legal_actions_list: List[List[int]],
    amp_enabled: bool,
):
    obs_t = torch.from_numpy(obs_batch).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits, values = model(obs_t)
    mask = torch.full_like(logits, -1e9)
    for i, legal in enumerate(legal_actions_list):
        mask[i, legal] = 0.0
    probs = F.softmax(logits + mask, dim=1)
    return probs.detach().cpu().numpy(), values.squeeze(1).float().detach().cpu().numpy()


class InferenceServer(threading.Thread):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        conns: List,
        model_lock: threading.Lock,
        stop_event: mp.Event,
        max_batch: int,
        batch_wait_ms: int,
        amp_enabled: bool,
    ):
        super().__init__(daemon=True)
        self.model = model
        self.device = device
        self.conns = conns
        self.model_lock = model_lock
        self.stop_event = stop_event
        self.max_batch = max(1, int(max_batch))
        self.batch_wait_ms = max(0, int(batch_wait_ms))
        self.amp_enabled = amp_enabled

    def run(self):
        while not self.stop_event.is_set():
            if not self.conns:
                time.sleep(0.01)
                continue
            ready = mp_wait(self.conns, timeout=0.01)
            if not ready:
                continue

            pending = []
            total = 0
            for conn in ready:
                try:
                    obs_batch, legal_actions_list = conn.recv()
                except EOFError:
                    try:
                        self.conns.remove(conn)
                    except ValueError:
                        pass
                    continue
                pending.append((conn, obs_batch, legal_actions_list))
                total += obs_batch.shape[0]

            t0 = time.time()
            while total < self.max_batch and (time.time() - t0) * 1000.0 < self.batch_wait_ms:
                more = mp_wait(self.conns, timeout=0)
                if not more:
                    break
                for conn in more:
                    try:
                        obs_batch, legal_actions_list = conn.recv()
                    except EOFError:
                        try:
                            self.conns.remove(conn)
                        except ValueError:
                            pass
                        continue
                    pending.append((conn, obs_batch, legal_actions_list))
                    total += obs_batch.shape[0]

            if not pending:
                continue

            obs_cat = np.concatenate([p[1] for p in pending], axis=0)
            legal_cat: List[List[int]] = []
            splits = []
            for _conn, obs_batch, legal_list in pending:
                splits.append(len(legal_list))
                legal_cat.extend(legal_list)

            with self.model_lock:
                self.model.eval()
                priors_all, values_all = policy_value_batch_torch(
                    self.model, obs_cat, self.device, legal_cat, self.amp_enabled
                )

            offset = 0
            for (conn, _obs_batch, _legal_list), count in zip(pending, splits):
                priors = priors_all[offset : offset + count]
                values = values_all[offset : offset + count]
                offset += count
                try:
                    conn.send((priors, values))
                except (BrokenPipeError, EOFError):
                    try:
                        self.conns.remove(conn)
                    except ValueError:
                        pass


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


def run_mcts_batch(
    root_states: List[SnekState],
    model: nn.Module,
    device: torch.device,
    sims: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    rng: np.random.Generator,
    add_noise: bool,
) -> np.ndarray:
    n_roots = len(root_states)
    if n_roots == 0:
        return np.zeros((0, 4), dtype=np.float32)

    roots = []
    root_obs = np.stack([state.to_obs() for state in root_states], axis=0)
    root_legal = [state.legal_actions() for state in root_states]
    root_priors, _root_values = policy_value_batch(model, root_obs, device, root_legal)
    for i in range(n_roots):
        root = MCTSNode(0.0)
        root.expand(root_priors[i], root_legal[i])
        if add_noise:
            add_dirichlet_noise(root, dirichlet_alpha, dirichlet_eps, rng)
        roots.append(root)

    for _ in range(sims):
        paths = []
        leaf_states = []
        leaf_meta = []
        for i, root in enumerate(roots):
            node = root
            state = root_states[i].clone()
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
                paths.append((search_path, value, None))
            else:
                leaf_idx = len(leaf_states)
                leaf_states.append(state)
                leaf_meta.append((node, state.legal_actions()))
                paths.append((search_path, None, leaf_idx))

        if leaf_states:
            obs_batch = np.stack([state.to_obs() for state in leaf_states], axis=0)
            priors_batch, values_batch = policy_value_batch(model, obs_batch, device, [m[1] for m in leaf_meta])
        else:
            priors_batch = None
            values_batch = None

        for search_path, value, leaf_idx in paths:
            if value is None and leaf_idx is not None:
                node, legal = leaf_meta[leaf_idx]
                node.expand(priors_batch[leaf_idx], legal)
                value = float(values_batch[leaf_idx])
            for n in search_path:
                n.visit_count += 1
                n.value_sum += value

    counts = np.zeros((n_roots, 4), dtype=np.float32)
    for i, root in enumerate(roots):
        for action, child in root.children.items():
            counts[i, action] = float(child.visit_count)
    return counts


def run_mcts_batch_remote(
    root_states: List[SnekState],
    infer_fn,
    sims: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    rng: np.random.Generator,
    add_noise: bool,
) -> np.ndarray:
    n_roots = len(root_states)
    if n_roots == 0:
        return np.zeros((0, 4), dtype=np.float32)

    roots = []
    root_obs = np.stack([state.to_obs() for state in root_states], axis=0)
    root_legal = [state.legal_actions() for state in root_states]
    root_priors, _root_values = infer_fn(root_obs, root_legal)
    for i in range(n_roots):
        root = MCTSNode(0.0)
        root.expand(root_priors[i], root_legal[i])
        if add_noise:
            add_dirichlet_noise(root, dirichlet_alpha, dirichlet_eps, rng)
        roots.append(root)

    for _ in range(sims):
        paths = []
        leaf_states = []
        leaf_meta = []
        for i, root in enumerate(roots):
            node = root
            state = root_states[i].clone()
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
                paths.append((search_path, value, None))
            else:
                leaf_idx = len(leaf_states)
                leaf_states.append(state)
                leaf_meta.append((node, state.legal_actions()))
                paths.append((search_path, None, leaf_idx))

        if leaf_states:
            obs_batch = np.stack([state.to_obs() for state in leaf_states], axis=0)
            priors_batch, values_batch = infer_fn(obs_batch, [m[1] for m in leaf_meta])
        else:
            priors_batch = None
            values_batch = None

        for search_path, value, leaf_idx in paths:
            if value is None and leaf_idx is not None:
                node, legal = leaf_meta[leaf_idx]
                node.expand(priors_batch[leaf_idx], legal)
                value = float(values_batch[leaf_idx])
            for n in search_path:
                n.visit_count += 1
                n.value_sum += value

    counts = np.zeros((n_roots, 4), dtype=np.float32)
    for i, root in enumerate(roots):
        for action, child in root.children.items():
            counts[i, action] = float(child.visit_count)
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
    status_hook=None,
    status_interval: int = 10,
):
    state = SnekState.new(config, seed=seed)
    episode = []
    ep_reward = 0.0
    ep_len = 0
    ep_max_len = state.length
    death_type = None
    last_status_step = 0

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
        if status_hook is not None and (ep_len - last_status_step) >= status_interval:
            status_hook(ep_len, ep_reward, ep_max_len)
            last_status_step = ep_len

    outcome = outcome_value(state)
    examples = [(obs, policy, outcome) for obs, policy in episode]
    return examples, ep_reward, ep_len, ep_max_len, death_type


def self_play_batch(
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
    batch_size: int,
    status_hook=None,
    status_interval: int = 10,
    progress_hook=None,
):
    states = [SnekState.new(config, seed=seed + i * 9973) for i in range(batch_size)]
    episodes = [[] for _ in range(batch_size)]
    ep_rewards = [0.0 for _ in range(batch_size)]
    ep_lengths = [0 for _ in range(batch_size)]
    ep_max_lens = [state.length for state in states]
    death_types = [None for _ in range(batch_size)]
    active = [True for _ in range(batch_size)]
    total_steps = 0
    steps_since_status = 0

    while any(active):
        active_idx = [i for i, a in enumerate(active) if a]
        active_states = [states[i] for i in active_idx]
        counts_batch = run_mcts_batch(
            active_states,
            model,
            device,
            sims,
            c_puct,
            dirichlet_alpha,
            dirichlet_eps,
            rng,
            add_noise=True,
        )

        policies = []
        for j, i in enumerate(active_idx):
            counts = counts_batch[j]
            total = float(counts.sum())
            if total > 0:
                policy = counts / total
            else:
                policy = np.full((4,), 0.25, dtype=np.float32)
            policies.append(policy)
            episodes[i].append((states[i].to_obs(), policy))

        for j, i in enumerate(active_idx):
            temp = temperature if ep_lengths[i] < temp_threshold else 0.0
            action = select_action_from_policy(policies[j], temp, rng)
            reward, done, info = states[i].step(action)
            ep_rewards[i] += reward
            ep_lengths[i] += 1
            ep_max_lens[i] = max(ep_max_lens[i], int(info.get("length", states[i].length)))
            total_steps += 1
            steps_since_status += 1

            if done or ep_lengths[i] >= max_steps:
                active[i] = False
                if done:
                    death_types[i] = info.get("death")

        if status_hook is not None and steps_since_status >= status_interval:
            last_reward = ep_rewards[active_idx[0]] if active_idx else 0.0
            last_max_len = max(ep_max_lens) if ep_max_lens else 0
            status_hook(total_steps, last_reward, last_max_len)
            if progress_hook is not None:
                progress_hook(total_steps, ep_rewards, ep_lengths, ep_max_lens, death_types)
            steps_since_status = 0

    examples_all = []
    for i, episode in enumerate(episodes):
        if not episode:
            continue
        outcome = outcome_value(states[i])
        for obs, policy in episode:
            examples_all.append((obs, policy, outcome))

    return examples_all, ep_rewards, ep_lengths, ep_max_lens, death_types, total_steps


def self_play_batch_remote(
    infer_fn,
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
    batch_size: int,
):
    states = [SnekState.new(config, seed=seed + i * 9973) for i in range(batch_size)]
    episodes = [[] for _ in range(batch_size)]
    ep_rewards = [0.0 for _ in range(batch_size)]
    ep_lengths = [0 for _ in range(batch_size)]
    ep_max_lens = [state.length for state in states]
    death_types = [None for _ in range(batch_size)]
    active = [True for _ in range(batch_size)]
    total_steps = 0

    while any(active):
        active_idx = [i for i, a in enumerate(active) if a]
        active_states = [states[i] for i in active_idx]
        counts_batch = run_mcts_batch_remote(
            active_states,
            infer_fn,
            sims,
            c_puct,
            dirichlet_alpha,
            dirichlet_eps,
            rng,
            add_noise=True,
        )

        policies = []
        for j, i in enumerate(active_idx):
            counts = counts_batch[j]
            total = float(counts.sum())
            if total > 0:
                policy = counts / total
            else:
                policy = np.full((4,), 0.25, dtype=np.float32)
            policies.append(policy)
            episodes[i].append((states[i].to_obs(), policy))

        for j, i in enumerate(active_idx):
            temp = temperature if ep_lengths[i] < temp_threshold else 0.0
            action = select_action_from_policy(policies[j], temp, rng)
            reward, done, info = states[i].step(action)
            ep_rewards[i] += reward
            ep_lengths[i] += 1
            ep_max_lens[i] = max(ep_max_lens[i], int(info.get("length", states[i].length)))
            total_steps += 1

            if done or ep_lengths[i] >= max_steps:
                death_types[i] = info.get("death")
                active[i] = False

    examples_all = []
    for i, episode in enumerate(episodes):
        if not episode:
            continue
        outcome = outcome_value(states[i])
        for obs, policy in episode:
            examples_all.append((obs, policy, outcome))

    return examples_all, ep_rewards, ep_lengths, ep_max_lens, death_types, total_steps


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


def worker_main(conn, result_queue, args, config: SnekConfig, worker_id: int, stop_event: mp.Event):
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    seed_base = args.seed if args.seed > 0 else None
    np_rng = np.random.default_rng(seed_base if seed_base is not None else None)
    py_rng = random.Random(seed_base if seed_base is not None else None)
    py_rng.seed((worker_id + 1) * 10007 if seed_base is None else seed_base + worker_id * 9973)

    per_worker_batch = max(1, int(math.ceil(args.selfplay_batch / max(1, args.workers))))

    def infer_fn(obs_batch: np.ndarray, legal_actions_list: List[List[int]]):
        conn.send((obs_batch, legal_actions_list))
        return conn.recv()

    while not stop_event.is_set():
        try:
            seed = py_rng.randrange(1 << 30)
            (
                examples_all,
                ep_rewards,
                ep_lengths,
                ep_max_lens,
                death_types,
                steps_taken,
            ) = self_play_batch_remote(
                infer_fn,
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
                per_worker_batch,
            )
            if steps_taken <= 0:
                continue

            last_ep_len = int(ep_lengths[-1]) if ep_lengths else 0
            last_ep_reward = float(ep_rewards[-1]) if ep_rewards else 0.0
            last_ep_max_len = int(ep_max_lens[-1]) if ep_max_lens else 0
            result = {
                "examples": examples_all,
                "ep_rewards": ep_rewards,
                "ep_lengths": ep_lengths,
                "ep_max_lens": ep_max_lens,
                "death_types": death_types,
                "steps": steps_taken,
                "last_ep_len": last_ep_len,
                "last_ep_reward": last_ep_reward,
                "last_ep_max_len": last_ep_max_len,
            }
            result_queue.put(result)
        except Exception as exc:
            try:
                result_queue.put({"error": f"worker {worker_id}: {exc}"})
            except Exception:
                pass
            break


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
    parser.add_argument("--selfplay-batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--infer-max-batch", type=int, default=2048)
    parser.add_argument("--infer-batch-wait-ms", type=int, default=2)
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
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="max-autotune", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--clean-logs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--out-dir", type=str, default="rl_out_az_mp")
    parser.add_argument("--live-plot", action="store_true")
    parser.add_argument("--plot-refresh-ms", type=int, default=1000)
    parser.add_argument("--plot-no-stream", action="store_true")
    parser.add_argument("--stop-file", type=str, default="rl_out_az_mp/stop.txt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    np_rng = np.random.default_rng(args.seed if args.seed > 0 else None)
    py_rng = random.Random(args.seed if args.seed > 0 else None)

    config = SnekConfig(grid_w=args.grid, grid_h=args.grid)

    model = build_model(args.model_size, 4, args.grid, args.grid, 4).to(device)
    if args.torch_compile and hasattr(torch, "compile"):
        try:
            import importlib.util

            if importlib.util.find_spec("triton") is None:
                _log_error(Path(args.out_dir) / "error.log", "torch_compile disabled: triton not installed")
            else:
                try:
                    import torch._dynamo as dynamo

                    dynamo.config.suppress_errors = True
                except Exception:
                    pass
                model = torch.compile(model, mode=args.compile_mode)
        except Exception as exc:
            _log_error(Path(args.out_dir) / "error.log", f"torch_compile failed: {exc}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    replay = ReplayBuffer(args.replay_size)

    out_dir = Path(args.out_dir)
    ckpt_suffix = "" if args.model_size == "base" else f"_{args.model_size}"
    train_log = out_dir / "train_log.csv"
    eval_log = out_dir / "eval_log.csv"
    error_log = out_dir / "error.log"
    status_log = out_dir / "status.log"
    stop_path = Path(args.stop_file) if args.stop_file else None

    def _excepthook(exc_type, exc, tb):
        _log_error(error_log, "".join(traceback.format_exception(exc_type, exc, tb)))

    sys.excepthook = _excepthook

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
    if args.clean_logs and status_log.exists():
        try:
            status_log.unlink()
        except OSError:
            pass

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
    last_log_time = time.time()
    last_log_steps = start_step
    log_ready = True
    last_status_time = 0.0
    solved_streak = 0
    target_solve_len = args.solve_min_max_len if args.solve_min_max_len > 0 else args.grid * args.grid
    required_solve_evals = max(1, int(args.solve_evals))

    if stop_path is not None:
        try:
            if stop_path.exists():
                stop_path.unlink()
        except OSError as exc:
            _log_error(error_log, f"Failed to remove stop file: {exc}")

    status_state = {
        "steps": start_step,
        "replay": 0,
        "training_started": False,
        "last_ep_len": 0,
        "last_ep_reward": 0.0,
        "last_ep_max_len": 0,
    }
    status_lock = threading.Lock()

    def status_update(step_progress: int, ep_reward: float, ep_max_len: int):
        with status_lock:
            status_state["steps"] = global_step + step_progress
            status_state["replay"] = len(replay)
            status_state["training_started"] = len(replay) >= args.replay_warmup
            status_state["last_ep_len"] = step_progress
            status_state["last_ep_reward"] = ep_reward
            status_state["last_ep_max_len"] = ep_max_len

    def heartbeat_worker():
        while True:
            time.sleep(10.0)
            with status_lock:
                payload = dict(status_state)
            _write_status(status_log, payload)

    threading.Thread(target=heartbeat_worker, daemon=True).start()
    _write_status(status_log, status_state)

    def _mean_vals(vals):
        if isinstance(vals, list):
            return float(np.mean(vals)) if vals else 0.0
        return float(np.mean(vals)) if vals is not None else 0.0

    def _max_vals(vals):
        if isinstance(vals, list):
            return float(max(vals)) if vals else 0.0
        return float(np.max(vals)) if vals is not None else 0.0

    first_progress_log = True

    def log_progress(step_progress, ep_rewards, ep_lengths, ep_max_lens, death_types):
        nonlocal last_log, last_log_time, last_log_steps, best_train_rate, last_best_len, last_best_time, best_eval_rate, first_progress_log
        progress_steps = global_step + int(step_progress)
        if progress_steps <= last_log and not first_progress_log:
            return
        now = time.time()
        runtime_sec = now - start_time
        dt = max(1e-6, now - (last_log_time or now))
        dsteps = progress_steps - last_log_steps
        fps = dsteps / dt if dsteps > 0 else 0.0

        mean_reward = _mean_vals(ep_rewards)
        mean_len = _mean_vals(ep_lengths)
        mean_max_len = _mean_vals(ep_max_lens)
        cur_best_len = int(max(stats.best_train_max_len, _max_vals(ep_max_lens)))

        if cur_best_len > last_best_len:
            dt_min = max(1e-6, (now - last_best_time) / 60.0)
            best_train_rate = (cur_best_len - last_best_len) / dt_min
            last_best_len = cur_best_len
            last_best_time = now
        if best_eval_max_len > 0:
            best_eval_rate = best_eval_max_len / max(1e-6, runtime_sec / 60.0)

        row = {
            "steps": progress_steps,
            "eps": 0.0,
            "fps": fps,
            "runtime_sec": runtime_sec,
            "mean_reward": mean_reward,
            "mean_len": mean_len,
            "mean_max_len": mean_max_len,
            "best_train_max_len": cur_best_len,
            "best_eval_max_len": best_eval_max_len,
            "best_train_rate_per_min": best_train_rate,
            "best_eval_rate_per_min": best_eval_rate,
            "loss": 0.0,
            "buffer_size": len(replay),
            "board_size": args.grid,
            "death_wall_per_k": 0.0,
            "death_self_per_k": 0.0,
            "training_started": len(replay) >= args.replay_warmup,
        }
        write_row(train_log, row, train_header, error_log=error_log)
        last_log = progress_steps
        last_log_steps = progress_steps
        last_log_time = now
        first_progress_log = False

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=max(2, args.workers * 2))
    stop_event = ctx.Event()
    conns = []
    workers = []
    for i in range(max(1, args.workers)):
        conn_main, conn_worker = ctx.Pipe()
        proc = ctx.Process(target=worker_main, args=(conn_worker, result_queue, args, config, i, stop_event))
        proc.daemon = True
        proc.start()
        conns.append(conn_main)
        workers.append(proc)

    model_lock = threading.Lock()
    infer_server = InferenceServer(
        model,
        device,
        conns,
        model_lock,
        stop_event,
        args.infer_max_batch,
        args.infer_batch_wait_ms,
        amp_enabled,
    )
    infer_server.start()

    while global_step < target_steps:
        if stop_path is not None:
            try:
                if stop_path.exists():
                    break
            except OSError as exc:
                _log_error(error_log, f"stop_file check failed: {exc}")

        try:
            result = result_queue.get(timeout=1.0)
        except Empty:
            continue

        if not isinstance(result, dict):
            continue
        if result.get("error"):
            _log_error(error_log, result["error"])
            continue

        examples_all = result.get("examples", [])
        ep_rewards = result.get("ep_rewards", [])
        ep_lengths = result.get("ep_lengths", [])
        ep_max_lens = result.get("ep_max_lens", [])
        death_types = result.get("death_types", [])
        steps_taken = int(result.get("steps", 0))
        if steps_taken <= 0:
            continue

        replay.add_many(examples_all)
        global_step += steps_taken
        death_steps_window += steps_taken

        for ep_reward, ep_len, ep_max_len, death_type in zip(ep_rewards, ep_lengths, ep_max_lens, death_types):
            if ep_len <= 0:
                continue
            stats.ep_rewards.append(float(ep_reward))
            stats.ep_lengths.append(int(ep_len))
            stats.ep_max_lens.append(int(ep_max_len))
            stats.best_train_max_len = max(stats.best_train_max_len, int(ep_max_len))
            if death_type == "wall":
                death_wall_window += 1
            elif death_type == "self":
                death_self_window += 1

        last_ep_len = int(result.get("last_ep_len", 0))
        last_ep_reward = float(result.get("last_ep_reward", 0.0))
        last_ep_max_len = int(result.get("last_ep_max_len", 0))

        training_started = len(replay) >= args.replay_warmup
        now = time.time()
        if now - last_status_time >= 10.0:
            with status_lock:
                status_state["steps"] = global_step
                status_state["replay"] = len(replay)
                status_state["training_started"] = training_started
                status_state["last_ep_len"] = last_ep_len
                status_state["last_ep_reward"] = last_ep_reward
                status_state["last_ep_max_len"] = last_ep_max_len
            last_status_time = now

        loss_val = 0.0
        if training_started:
            with model_lock:
                model.train()
                for _ in range(args.train_steps):
                    if len(replay) < args.batch_size:
                        break
                    b_obs, b_policy, b_value = replay.sample(args.batch_size, np_rng)
                    obs_t = torch.from_numpy(b_obs).to(device)
                    policy_t = torch.from_numpy(b_policy).to(device)
                    value_t = torch.from_numpy(b_value).to(device)

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        logits, value_pred = model(obs_t)
                        log_probs = F.log_softmax(logits, dim=1)
                        policy_loss = -(policy_t * log_probs).sum(dim=1).mean()
                        value_loss = F.mse_loss(value_pred.squeeze(1), value_t)
                        loss = policy_loss + args.value_loss_weight * value_loss

                    if amp_enabled:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    loss_val = float(loss.item())
                model.eval()

        if log_ready and (global_step - last_eval >= args.eval_interval):
            try:
                eval_sims = args.eval_sims if args.eval_sims > 0 else args.mcts_sims
                eval_max_len = 0
                total_eval_reward = 0.0
                with model_lock:
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

        if log_ready and (global_step - last_log >= args.log_interval):
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

    stop_event.set()
    for proc in workers:
        try:
            proc.join(timeout=1.0)
        except Exception:
            pass

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

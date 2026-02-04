import argparse
import csv
import math
import random
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from snek_env import SnekConfig

AMP_ENABLED = False


@dataclass
class TrainStats:
    ep_rewards: List[float]
    ep_lengths: List[int]
    ep_max_lens: List[int]
    best_train_max_len: int = 0
    best_eval_max_len: int = 0


@dataclass
class MCTSWorkspace:
    max_batch: int
    max_nodes_per_root: int
    max_depth: int
    grid_w: int
    grid_h: int
    device: torch.device
    child_index: torch.Tensor
    child_prior: torch.Tensor
    node_visit: torch.Tensor
    node_value_sum: torch.Tensor
    node_expanded: torch.Tensor
    node_terminal: torch.Tensor
    body_age: torch.Tensor
    length: torch.Tensor
    head_x: torch.Tensor
    head_y: torch.Tensor
    direction: torch.Tensor
    food_x: torch.Tensor
    food_y: torch.Tensor
    steps_since_food: torch.Tensor
    rng_state: torch.Tensor
    root_offsets: torch.Tensor
    wall_mask: torch.Tensor
    dx: torch.Tensor
    dy: torch.Tensor
    opposite: torch.Tensor
    legal_mask_table: torch.Tensor
    path_nodes: torch.Tensor
    cur_nodes: torch.Tensor
    values: torch.Tensor
    done_mask: torch.Tensor


def make_mcts_workspace(
    max_batch: int,
    max_nodes_per_root: int,
    max_depth: int,
    config: SnekConfig,
    device: torch.device,
) -> MCTSWorkspace:
    grid_w = int(config.grid_w)
    grid_h = int(config.grid_h)
    total_nodes = max_batch * max_nodes_per_root

    child_index = torch.empty((total_nodes, 4), dtype=torch.int64, device=device)
    child_prior = torch.empty((total_nodes, 4), dtype=torch.float32, device=device)
    node_visit = torch.empty((total_nodes,), dtype=torch.float32, device=device)
    node_value_sum = torch.empty((total_nodes,), dtype=torch.float32, device=device)
    node_expanded = torch.empty((total_nodes,), dtype=torch.bool, device=device)
    node_terminal = torch.empty((total_nodes,), dtype=torch.bool, device=device)

    body_age = torch.empty((total_nodes, grid_h, grid_w), dtype=torch.int16, device=device)
    length = torch.empty((total_nodes,), dtype=torch.int16, device=device)
    head_x = torch.empty((total_nodes,), dtype=torch.int16, device=device)
    head_y = torch.empty((total_nodes,), dtype=torch.int16, device=device)
    direction = torch.empty((total_nodes,), dtype=torch.int16, device=device)
    food_x = torch.empty((total_nodes,), dtype=torch.int16, device=device)
    food_y = torch.empty((total_nodes,), dtype=torch.int16, device=device)
    steps_since_food = torch.empty((total_nodes,), dtype=torch.int32, device=device)
    rng_state = torch.empty((total_nodes,), dtype=torch.int64, device=device)

    root_offsets = torch.arange(max_batch, device=device, dtype=torch.int64) * max_nodes_per_root

    wall_mask = torch.zeros((grid_h, grid_w), dtype=torch.float32, device=device)
    wall_mask[0, :] = 1.0
    wall_mask[grid_h - 1, :] = 1.0
    wall_mask[:, 0] = 1.0
    wall_mask[:, grid_w - 1] = 1.0

    dx = torch.tensor([0, 0, -1, 1], dtype=torch.int16, device=device)
    dy = torch.tensor([-1, 1, 0, 0], dtype=torch.int16, device=device)
    opposite = torch.tensor([1, 0, 3, 2], dtype=torch.int16, device=device)

    legal_mask_table = torch.ones((4, 4), dtype=torch.bool, device=device)
    reverse = torch.tensor([1, 0, 3, 2], dtype=torch.int64, device=device)
    legal_mask_table[torch.arange(4, device=device), reverse] = False

    path_nodes = torch.empty((max_batch, max_depth), dtype=torch.int64, device=device)
    cur_nodes = torch.empty((max_batch,), dtype=torch.int64, device=device)
    values = torch.empty((max_batch,), dtype=torch.float32, device=device)
    done_mask = torch.empty((max_batch,), dtype=torch.bool, device=device)

    return MCTSWorkspace(
        max_batch=max_batch,
        max_nodes_per_root=max_nodes_per_root,
        max_depth=max_depth,
        grid_w=grid_w,
        grid_h=grid_h,
        device=device,
        child_index=child_index,
        child_prior=child_prior,
        node_visit=node_visit,
        node_value_sum=node_value_sum,
        node_expanded=node_expanded,
        node_terminal=node_terminal,
        body_age=body_age,
        length=length,
        head_x=head_x,
        head_y=head_y,
        direction=direction,
        food_x=food_x,
        food_y=food_y,
        steps_since_food=steps_since_food,
        rng_state=rng_state,
        root_offsets=root_offsets,
        wall_mask=wall_mask,
        dx=dx,
        dy=dy,
        opposite=opposite,
        legal_mask_table=legal_mask_table,
        path_nodes=path_nodes,
        cur_nodes=cur_nodes,
        values=values,
        done_mask=done_mask,
    )

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

    def add_many(self, items):
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

def outcome_value_tensor(length: torch.Tensor, grid_cells: int) -> torch.Tensor:
    denom = max(1, grid_cells - 3)
    progress = (length.to(torch.float32) - 3.0) / float(denom)
    progress = progress.clamp(0.0, 1.0)
    return progress * 2.0 - 1.0


def build_obs(body_age: torch.Tensor, food_x: torch.Tensor, food_y: torch.Tensor, wall_mask: torch.Tensor):
    body_mask = body_age >= 0
    head_mask = body_age == 0
    body_only = body_mask & ~head_mask

    b, h, w = body_age.shape
    food_mask = torch.zeros((b, h, w), dtype=torch.bool, device=body_age.device)
    idx = torch.arange(b, device=body_age.device)
    fx = food_x.to(torch.int64).clamp(0, w - 1)
    fy = food_y.to(torch.int64).clamp(0, h - 1)
    food_mask[idx, fy, fx] = True

    obs = torch.stack(
        [
            body_only.to(torch.float32),
            head_mask.to(torch.float32),
            food_mask.to(torch.float32),
            wall_mask.expand(b, -1, -1),
        ],
        dim=1,
    )
    return obs


def _lcg_next(state: torch.Tensor) -> torch.Tensor:
    a = torch.tensor(6364136223846793005, dtype=torch.int64, device=state.device)
    c = torch.tensor(1442695040888963407, dtype=torch.int64, device=state.device)
    mask = torch.tensor((1 << 63) - 1, dtype=torch.int64, device=state.device)
    return torch.bitwise_and(state * a + c, mask)


def resample_food(
    body_age: torch.Tensor,
    food_x: torch.Tensor,
    food_y: torch.Tensor,
    rng_state: torch.Tensor,
    mask: torch.Tensor,
    grid_w: int,
    grid_h: int,
    attempts: int = 8,
):
    device = body_age.device
    mask = mask.to(device=device, dtype=torch.bool)
    if mask.numel() == 0:
        return food_x, food_y, rng_state

    occupancy = (body_age >= 0).view(body_age.shape[0], -1)
    free_cells = (~occupancy).any(dim=1)
    mask = mask & free_cells
    if not mask.any():
        return food_x, food_y, rng_state

    rng = rng_state
    new_x = food_x
    new_y = food_y
    need = mask
    grid_cells = grid_w * grid_h
    grid_cells_t = torch.tensor(grid_cells, dtype=torch.int64, device=device)

    for _ in range(attempts):
        rng = _lcg_next(rng)
        idx = (rng >> 33) % grid_cells_t
        free = ~occupancy.gather(1, idx.unsqueeze(1)).squeeze(1)
        accept = need & free
        new_x = torch.where(accept, (idx % grid_w).to(new_x.dtype), new_x)
        new_y = torch.where(accept, (idx // grid_w).to(new_y.dtype), new_y)
        need = need & ~accept

    arange = torch.arange(grid_cells, device=device, dtype=torch.int64).view(1, -1)
    scores = torch.where(occupancy, torch.full_like(arange, grid_cells + 1), arange)
    first_idx = scores.argmin(dim=1)
    new_x = torch.where(need, (first_idx % grid_w).to(new_x.dtype), new_x)
    new_y = torch.where(need, (first_idx // grid_w).to(new_y.dtype), new_y)

    rng_state = torch.where(mask, rng, rng_state)
    food_x = torch.where(mask, new_x, food_x)
    food_y = torch.where(mask, new_y, food_y)
    return food_x, food_y, rng_state


def init_state(batch: int, config: SnekConfig, device: torch.device, seed: int):
    grid_w = int(config.grid_w)
    grid_h = int(config.grid_h)
    body_age = torch.full((batch, grid_h, grid_w), -1, dtype=torch.int16, device=device)
    length = torch.full((batch,), 3, dtype=torch.int16, device=device)
    head_x = torch.zeros((batch,), dtype=torch.int16, device=device)
    head_y = torch.zeros((batch,), dtype=torch.int16, device=device)
    direction = torch.full((batch,), 3, dtype=torch.int16, device=device)
    food_x = torch.zeros((batch,), dtype=torch.int16, device=device)
    food_y = torch.zeros((batch,), dtype=torch.int16, device=device)
    steps_since_food = torch.zeros((batch,), dtype=torch.int32, device=device)

    cx = grid_w // 2
    cy = grid_h // 2
    body_age[:, cy, cx] = 0
    body_age[:, cy, cx - 1] = 1
    body_age[:, cy, cx - 2] = 2
    head_x[:] = cx
    head_y[:] = cy

    base_seed = seed if seed > 0 else 12345
    rng_state = (
        torch.arange(batch, dtype=torch.int64, device=device) * torch.tensor(9973, dtype=torch.int64, device=device)
        + torch.tensor(base_seed, dtype=torch.int64, device=device)
    )

    food_x, food_y, rng_state = resample_food(
        body_age, food_x, food_y, rng_state, torch.ones(batch, device=device), grid_w, grid_h
    )
    return {
        "body_age": body_age,
        "length": length,
        "head_x": head_x,
        "head_y": head_y,
        "direction": direction,
        "food_x": food_x,
        "food_y": food_y,
        "steps_since_food": steps_since_food,
        "rng_state": rng_state,
    }

def step_state(state: dict, actions: torch.Tensor, config: SnekConfig, dx: torch.Tensor, dy: torch.Tensor, opposite: torch.Tensor):
    device = actions.device
    grid_w = int(config.grid_w)
    grid_h = int(config.grid_h)
    grid_cells = grid_w * grid_h

    actions = actions.to(device=device, dtype=torch.int16)
    direction = state["direction"]
    opp = opposite[direction.to(torch.int64)]
    is_reverse = actions == opp
    new_dir = torch.where(is_reverse, direction, actions)

    head_x = state["head_x"]
    head_y = state["head_y"]
    dx_t = dx[new_dir.to(torch.int64)]
    dy_t = dy[new_dir.to(torch.int64)]
    new_head_x = head_x + dx_t
    new_head_y = head_y + dy_t

    hit_wall = (
        (new_head_x < 0)
        | (new_head_x >= grid_w)
        | (new_head_y < 0)
        | (new_head_y >= grid_h)
    )

    valid = ~hit_wall
    clamped_x = new_head_x.clamp(0, grid_w - 1)
    clamped_y = new_head_y.clamp(0, grid_h - 1)
    linear = clamped_y.to(torch.int64) * grid_w + clamped_x.to(torch.int64)

    body_age = state["body_age"]
    flat_body = (body_age >= 0).view(body_age.shape[0], -1)
    hit_self = flat_body.gather(1, linear.view(-1, 1)).squeeze(1) & valid

    terminated = hit_wall | hit_self
    rewards = torch.full((body_age.shape[0],), -1.0 / float(grid_cells), dtype=torch.float32, device=device)
    rewards = torch.where(terminated, torch.tensor(float(config.death_penalty), device=device), rewards)
    if config.zero_out_on_death:
        score = (state["length"] - 3).to(torch.float32)
        rewards = torch.where(
            terminated & (score > 0),
            rewards - score * float(config.food_reward),
            rewards,
        )

    alive = ~terminated
    ate = alive & (new_head_x == state["food_x"]) & (new_head_y == state["food_y"])
    length_new = state["length"] + ate.to(state["length"].dtype)
    win = alive & (length_new >= grid_cells)

    rewards = torch.where(win, float(config.food_reward + config.win_reward), rewards)
    ate_no_win = ate & ~win
    rewards = torch.where(ate_no_win, float(config.food_reward), rewards)

    if config.max_no_food_steps is not None and config.max_no_food_steps > 0:
        steps_since_food = torch.where(alive, state["steps_since_food"] + 1, state["steps_since_food"])
        steps_since_food = torch.where(ate, torch.zeros_like(steps_since_food), steps_since_food)
        truncated = alive & (steps_since_food >= int(config.max_no_food_steps))
    else:
        steps_since_food = state["steps_since_food"]
        truncated = torch.zeros_like(alive)

    done = terminated | win | truncated

    body = body_age
    body = torch.where((body >= 0) & alive[:, None, None], body + 1, body)

    head_mask = torch.zeros((body.shape[0], grid_cells), dtype=torch.bool, device=device)
    head_mask.scatter_(1, linear.view(-1, 1), alive.view(-1, 1))
    head_mask = head_mask.view(body.shape[0], grid_h, grid_w)
    body = torch.where(head_mask, torch.zeros_like(body), body)

    length_new = torch.where(alive, length_new, state["length"])
    body = torch.where(body >= length_new.view(-1, 1, 1), -1, body)

    food_x, food_y, rng_state = resample_food(
        body,
        state["food_x"],
        state["food_y"],
        state["rng_state"],
        ate_no_win,
        grid_w,
        grid_h,
    )

    next_state = {
        "body_age": body,
        "length": length_new,
        "head_x": torch.where(alive, new_head_x, head_x),
        "head_y": torch.where(alive, new_head_y, head_y),
        "direction": torch.where(alive, new_dir, direction),
        "food_x": food_x,
        "food_y": food_y,
        "steps_since_food": steps_since_food,
        "rng_state": rng_state,
    }

    info = {
        "length": length_new,
        "death_wall": hit_wall,
        "death_self": hit_self,
    }

    return next_state, rewards, done, info


def masked_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits.masked_fill(~legal_mask, -1e9), dim=-1)


def legal_mask_from_dir(direction: torch.Tensor, table: Optional[torch.Tensor] = None) -> torch.Tensor:
    if table is not None:
        return table[direction.to(torch.int64)]
    b = direction.shape[0]
    mask = torch.ones((b, 4), dtype=torch.bool, device=direction.device)
    reverse = torch.tensor([1, 0, 3, 2], dtype=torch.int16, device=direction.device)
    rev_action = reverse[direction.to(torch.int64)].to(torch.int64)
    mask[torch.arange(b, device=direction.device), rev_action] = False
    return mask


@torch.inference_mode()
def policy_value_batch(model: nn.Module, obs_batch: torch.Tensor, legal_mask: torch.Tensor):
    with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
        logits, values = model(obs_batch)
    probs = masked_softmax(logits, legal_mask)
    return probs, values.squeeze(1)


def select_puct(
    node_idx: torch.Tensor,
    child_index: torch.Tensor,
    child_prior: torch.Tensor,
    node_visit: torch.Tensor,
    node_value_sum: torch.Tensor,
    direction: torch.Tensor,
    c_puct: float,
    legal_mask_table: Optional[torch.Tensor] = None,
):
    idx = node_idx.to(torch.int64)
    child_idx = child_index[idx]
    priors = child_prior[idx]
    legal_mask = legal_mask_from_dir(direction[idx], legal_mask_table)

    max_idx = node_visit.shape[0] - 1
    child_idx_clamped = torch.clamp(child_idx, min=0, max=max_idx)
    child_visits = node_visit[child_idx_clamped]
    child_values = torch.zeros_like(child_visits, dtype=torch.float32)
    nonzero = child_visits > 0
    child_values = torch.where(nonzero, node_value_sum[child_idx_clamped] / child_visits.clamp(min=1), child_values)

    parent_visits = node_visit[idx].to(torch.float32)
    sqrt_visits = torch.sqrt(torch.clamp(parent_visits, min=1.0))

    u = c_puct * priors * sqrt_visits.unsqueeze(1) / (1.0 + child_visits.to(torch.float32))
    score = child_values + u
    score = torch.where(legal_mask, score, torch.tensor(-1e9, device=score.device))
    actions = score.argmax(dim=1)
    return actions

def mcts_search_batch(
    model: nn.Module,
    root_state: dict,
    config: SnekConfig,
    sims: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    device: torch.device,
    max_nodes_per_root: int,
    max_depth: int,
    workspace: Optional[MCTSWorkspace] = None,
    roots: Optional[torch.Tensor] = None,
    next_free: Optional[torch.Tensor] = None,
    reuse_tree: bool = False,
    active_mask: Optional[torch.Tensor] = None,
):
    batch = root_state["body_age"].shape[0]
    if workspace is None or batch > workspace.max_batch or max_nodes_per_root > workspace.max_nodes_per_root or max_depth > workspace.max_depth:
        workspace = make_mcts_workspace(max(batch, 1), max_nodes_per_root, max_depth, config, device)

    grid_w = workspace.grid_w
    grid_h = workspace.grid_h
    grid_cells = grid_w * grid_h

    total_nodes = batch * max_nodes_per_root
    child_index = workspace.child_index[:total_nodes]
    child_prior = workspace.child_prior[:total_nodes]
    node_visit = workspace.node_visit[:total_nodes]
    node_value_sum = workspace.node_value_sum[:total_nodes]
    node_expanded = workspace.node_expanded[:total_nodes]
    node_terminal = workspace.node_terminal[:total_nodes]

    body_age = workspace.body_age[:total_nodes]
    length = workspace.length[:total_nodes]
    head_x = workspace.head_x[:total_nodes]
    head_y = workspace.head_y[:total_nodes]
    direction = workspace.direction[:total_nodes]
    food_x = workspace.food_x[:total_nodes]
    food_y = workspace.food_y[:total_nodes]
    steps_since_food = workspace.steps_since_food[:total_nodes]
    rng_state = workspace.rng_state[:total_nodes]

    if not reuse_tree:
        child_index.fill_(-1)
        child_prior.zero_()
        node_visit.zero_()
        node_value_sum.zero_()
        node_expanded.zero_()
        node_terminal.zero_()

        body_age.fill_(-1)
        length.zero_()
        head_x.zero_()
        head_y.zero_()
        direction.zero_()
        food_x.zero_()
        food_y.zero_()
        steps_since_food.zero_()
        rng_state.zero_()

    if roots is None:
        if workspace.max_nodes_per_root == max_nodes_per_root:
            roots = workspace.root_offsets[:batch]
        else:
            roots = torch.arange(batch, device=device, dtype=torch.int64) * max_nodes_per_root

    body_age[roots] = root_state["body_age"]
    length[roots] = root_state["length"]
    head_x[roots] = root_state["head_x"]
    head_y[roots] = root_state["head_y"]
    direction[roots] = root_state["direction"]
    food_x[roots] = root_state["food_x"]
    food_y[roots] = root_state["food_y"]
    steps_since_food[roots] = root_state["steps_since_food"]
    rng_state[roots] = root_state["rng_state"]

    wall_mask = workspace.wall_mask

    if active_mask is not None:
        active_roots = roots[active_mask]
    else:
        active_roots = roots

    if active_roots.numel() > 0:
        obs_root = build_obs(body_age[active_roots], food_x[active_roots], food_y[active_roots], wall_mask)
        legal_mask = legal_mask_from_dir(direction[active_roots], workspace.legal_mask_table)
        needs_expand = ~node_expanded[active_roots]
        if bool(needs_expand.any().item()):
            root_priors, _root_values = policy_value_batch(
                model, obs_root[needs_expand], legal_mask[needs_expand]
            )
            child_prior[active_roots[needs_expand]] = root_priors
            node_expanded[active_roots[needs_expand]] = True
        if dirichlet_eps > 0 and dirichlet_alpha > 0:
            noise = torch.distributions.Dirichlet(
                torch.full((4,), dirichlet_alpha, device=device)
            ).sample((active_roots.shape[0],))
            child_prior[active_roots] = child_prior[active_roots] * (1.0 - dirichlet_eps) + noise * dirichlet_eps

    dx = torch.tensor([0, 0, -1, 1], dtype=torch.int16, device=device)
    dy = torch.tensor([-1, 1, 0, 0], dtype=torch.int16, device=device)
    opposite = torch.tensor([1, 0, 3, 2], dtype=torch.int16, device=device)

    if next_free is None:
        next_free = roots + 1

    for _ in range(sims):
        cur_nodes = workspace.cur_nodes[:batch]
        cur_nodes.copy_(roots)
        values = workspace.values[:batch]
        values.zero_()
        done_mask = workspace.done_mask[:batch]
        if active_mask is None:
            done_mask.zero_()
        else:
            done_mask.copy_(~active_mask)
        path_nodes = workspace.path_nodes[:batch, :max_depth]
        path_nodes.fill_(-1)

        for depth in range(max_depth):
            active = ~done_mask
            if not bool(active.any()):
                break

            path_nodes[active, depth] = cur_nodes[active]

            term = node_terminal[cur_nodes] & active
            if bool(term.any()):
                term_nodes = cur_nodes[term]
                values[term] = outcome_value_tensor(length[term_nodes], grid_cells)
                done_mask[term] = True

            eval_mask = (~node_expanded[cur_nodes]) & active
            if bool(eval_mask.any()):
                eval_nodes = cur_nodes[eval_mask]
                obs = build_obs(body_age[eval_nodes], food_x[eval_nodes], food_y[eval_nodes], wall_mask)
                legal = legal_mask_from_dir(direction[eval_nodes], workspace.legal_mask_table)
                priors, vals = policy_value_batch(model, obs, legal)
                child_prior[eval_nodes] = priors
                node_expanded[eval_nodes] = True
                values[eval_mask] = vals
                done_mask[eval_mask] = True

            active = ~done_mask
            if not bool(active.any()):
                break

            active_idx = active.nonzero(as_tuple=False).squeeze(1)
            active_nodes = cur_nodes[active_idx]
            actions = select_puct(
                active_nodes,
                child_index,
                child_prior,
                node_visit,
                node_value_sum,
                direction,
                c_puct,
                workspace.legal_mask_table,
            )
            child = child_index[active_nodes, actions]

            need_new = child < 0
            if bool(need_new.any()):
                new_idx = active_idx[need_new]
                parent_nodes = active_nodes[need_new]
                parent_actions = actions[need_new]
                root_ids = (parent_nodes // max_nodes_per_root).to(torch.int64)
                alloc = next_free[root_ids]
                limit = roots[root_ids] + max_nodes_per_root
                valid = (alloc < limit) & (alloc < total_nodes)
                new_nodes = torch.where(valid, alloc, torch.full_like(alloc, -1))
                next_free[root_ids] = torch.where(valid, alloc + 1, alloc)

                if bool(valid.any()):
                    valid_idx = new_idx[valid]
                    valid_nodes = new_nodes[valid]
                    valid_parent_nodes = parent_nodes[valid]
                    valid_parent_actions = parent_actions[valid]

                    parent_state = {
                        "body_age": body_age[valid_parent_nodes],
                        "length": length[valid_parent_nodes],
                        "head_x": head_x[valid_parent_nodes],
                        "head_y": head_y[valid_parent_nodes],
                        "direction": direction[valid_parent_nodes],
                        "food_x": food_x[valid_parent_nodes],
                        "food_y": food_y[valid_parent_nodes],
                        "steps_since_food": steps_since_food[valid_parent_nodes],
                        "rng_state": rng_state[valid_parent_nodes],
                    }

                    next_state, _rewards, done, _info = step_state(
                        parent_state, valid_parent_actions, config, dx, dy, opposite
                    )
                    body_age[valid_nodes] = next_state["body_age"]
                    length[valid_nodes] = next_state["length"]
                    head_x[valid_nodes] = next_state["head_x"]
                    head_y[valid_nodes] = next_state["head_y"]
                    direction[valid_nodes] = next_state["direction"]
                    food_x[valid_nodes] = next_state["food_x"]
                    food_y[valid_nodes] = next_state["food_y"]
                    steps_since_food[valid_nodes] = next_state["steps_since_food"]
                    rng_state[valid_nodes] = next_state["rng_state"]

                    node_terminal[valid_nodes] = done
                    child_index[valid_parent_nodes, valid_parent_actions] = valid_nodes

                    obs = build_obs(body_age[valid_nodes], food_x[valid_nodes], food_y[valid_nodes], wall_mask)
                    legal = legal_mask_from_dir(direction[valid_nodes], workspace.legal_mask_table)
                    priors, vals = policy_value_batch(model, obs, legal)
                    child_prior[valid_nodes] = priors
                    node_expanded[valid_nodes] = True

                    values[valid_idx] = torch.where(done, outcome_value_tensor(length[valid_nodes], grid_cells), vals)
                    done_mask[valid_idx] = True

                invalid = ~valid
                if bool(invalid.any()):
                    invalid_idx = new_idx[invalid]
                    values[invalid_idx] = outcome_value_tensor(length[cur_nodes[invalid_idx]], grid_cells)
                    done_mask[invalid_idx] = True

            follow = ~need_new
            if bool(follow.any()):
                follow_idx = active_idx[follow]
                follow_child = child[follow]
                cur_nodes[follow_idx] = follow_child

        leftover = ~done_mask
        if bool(leftover.any()):
            eval_nodes = cur_nodes[leftover]
            obs = build_obs(body_age[eval_nodes], food_x[eval_nodes], food_y[eval_nodes], wall_mask)
            legal = legal_mask_from_dir(direction[eval_nodes], workspace.legal_mask_table)
            _priors, vals = policy_value_batch(model, obs, legal)
            values[leftover] = vals

        values_rep = values.view(batch, 1).expand(batch, max_depth)
        mask = path_nodes >= 0
        flat_nodes = path_nodes[mask].view(-1)
        flat_values = values_rep[mask].view(-1)
        if flat_nodes.numel() > 0:
            node_visit.index_add_(0, flat_nodes, torch.ones_like(flat_values))
            node_value_sum.index_add_(0, flat_nodes, flat_values)

    counts = torch.zeros((batch, 4), dtype=torch.float32, device=device)
    root_children = child_index[roots]
    child_visits = node_visit[torch.clamp(root_children, min=0, max=total_nodes - 1)]
    counts = torch.where(root_children >= 0, child_visits, counts)
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


def self_play_batch_gpu(
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
    max_nodes_per_root: int,
    max_depth: int,
    status_hook=None,
    status_interval: int = 10,
    progress_hook=None,
):
    state = init_state(batch_size, config, device, seed)
    ep_rewards = torch.zeros((batch_size,), dtype=torch.float32, device=device)
    ep_lengths = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    ep_max_lens = state["length"].to(torch.int32)
    death_types = torch.full((batch_size,), -1, dtype=torch.int8, device=device)

    active = torch.ones((batch_size,), dtype=torch.bool, device=device)
    total_steps = 0
    steps_since_status = 0

    workspace = make_mcts_workspace(batch_size, max_nodes_per_root, max_depth, config, device)
    wall_mask = workspace.wall_mask
    dx = workspace.dx
    dy = workspace.dy
    opposite = workspace.opposite

    roots = workspace.root_offsets[:batch_size].clone()
    next_free = roots + 1
    reuse_tree = False

    obs_buf = torch.empty(
        (batch_size, max_steps, 4, config.grid_h, config.grid_w), dtype=torch.float32, device=device
    )
    policy_buf = torch.empty((batch_size, max_steps, 4), dtype=torch.float32, device=device)

    while bool(active.any()):
        active_idx = active.nonzero(as_tuple=False).squeeze(1)
        if active_idx.numel() == 0:
            break

        root_state = {
            "body_age": state["body_age"][active_idx],
            "length": state["length"][active_idx],
            "head_x": state["head_x"][active_idx],
            "head_y": state["head_y"][active_idx],
            "direction": state["direction"][active_idx],
            "food_x": state["food_x"][active_idx],
            "food_y": state["food_y"][active_idx],
            "steps_since_food": state["steps_since_food"][active_idx],
            "rng_state": state["rng_state"][active_idx],
        }

        # Build a compact root_state for active envs but run MCTS over full batch with mask.
        full_root_state = {
            "body_age": state["body_age"],
            "length": state["length"],
            "head_x": state["head_x"],
            "head_y": state["head_y"],
            "direction": state["direction"],
            "food_x": state["food_x"],
            "food_y": state["food_y"],
            "steps_since_food": state["steps_since_food"],
            "rng_state": state["rng_state"],
        }

        counts = mcts_search_batch(
            model,
            full_root_state,
            config,
            sims,
            c_puct,
            dirichlet_alpha,
            dirichlet_eps,
            device,
            max_nodes_per_root,
            max_depth=max_depth,
            workspace=workspace,
            roots=roots,
            next_free=next_free,
            reuse_tree=reuse_tree,
            active_mask=active,
        )
        reuse_tree = True

        obs = build_obs(state["body_age"][active_idx], state["food_x"][active_idx], state["food_y"][active_idx], wall_mask)

        counts_active = counts[active_idx]
        totals = counts_active.sum(dim=1, keepdim=True)
        policy = torch.where(totals > 0, counts_active / totals, torch.full_like(counts_active, 0.25))
        temps = torch.where(ep_lengths[active_idx] < temp_threshold, torch.tensor(temperature, device=device), torch.tensor(0.0, device=device))
        temp_mask = temps > 0
        actions = policy.argmax(dim=1)
        if bool(temp_mask.any().item()):
            scaled = policy.clone()
            scaled[temp_mask] = scaled[temp_mask] ** (1.0 / temps[temp_mask].unsqueeze(1))
            scaled_sum = scaled[temp_mask].sum(dim=1, keepdim=True).clamp(min=1e-6)
            scaled[temp_mask] = scaled[temp_mask] / scaled_sum
            actions[temp_mask] = torch.multinomial(scaled[temp_mask], 1).squeeze(1)

        step_idx = ep_lengths[active_idx].to(torch.int64)
        obs_buf[active_idx, step_idx] = obs
        policy_buf[active_idx, step_idx] = policy

        actions_t = actions.to(torch.int16)
        next_state, rewards, done, info = step_state(root_state, actions_t, config, dx, dy, opposite)

        ep_rewards[active_idx] += rewards
        ep_lengths[active_idx] += 1
        ep_max_lens[active_idx] = torch.maximum(ep_max_lens[active_idx], info["length"].to(torch.int32))
        total_steps += int(active_idx.numel())
        steps_since_status += int(active_idx.numel())

        done_now = done | (ep_lengths[active_idx] >= max_steps)
        if bool(done_now.any().item()):
            done_idx = active_idx[done_now]
            wall = info["death_wall"][done_now]
            self_hit = info["death_self"][done_now]
            death_types[done_idx] = torch.where(wall, torch.tensor(0, dtype=torch.int8, device=device), death_types[done_idx])
            death_types[done_idx] = torch.where(self_hit, torch.tensor(1, dtype=torch.int8, device=device), death_types[done_idx])
            active[done_idx] = False

        for k in state:
            state[k][active_idx] = next_state[k]

        # Tree reuse: move roots to chosen child, creating it if needed.
        if active_idx.numel() > 0:
            root_nodes = roots[active_idx]
            child_nodes = workspace.child_index[root_nodes, actions.to(torch.int64)]
            need_child = child_nodes < 0
            if bool(need_child.any().item()):
                need_idx = active_idx[need_child]
                root_nodes_need = root_nodes[need_child]
                actions_need = actions[need_child].to(torch.int64)

                alloc = next_free[need_idx]
                limit = workspace.root_offsets[need_idx] + max_nodes_per_root
                valid = alloc < limit

                if bool(valid.any().item()):
                    valid_idx = need_idx[valid]
                    valid_nodes = alloc[valid]
                    workspace.child_index[root_nodes_need[valid], actions_need[valid]] = valid_nodes

                    # Fill new node state from next_state (already computed).
                    workspace.body_age[valid_nodes] = next_state["body_age"][need_child][valid]
                    workspace.length[valid_nodes] = next_state["length"][need_child][valid]
                    workspace.head_x[valid_nodes] = next_state["head_x"][need_child][valid]
                    workspace.head_y[valid_nodes] = next_state["head_y"][need_child][valid]
                    workspace.direction[valid_nodes] = next_state["direction"][need_child][valid]
                    workspace.food_x[valid_nodes] = next_state["food_x"][need_child][valid]
                    workspace.food_y[valid_nodes] = next_state["food_y"][need_child][valid]
                    workspace.steps_since_food[valid_nodes] = next_state["steps_since_food"][need_child][valid]
                    workspace.rng_state[valid_nodes] = next_state["rng_state"][need_child][valid]

                    workspace.node_expanded[valid_nodes] = False
                    workspace.node_terminal[valid_nodes] = done[need_child][valid]
                    workspace.node_visit[valid_nodes] = 0.0
                    workspace.node_value_sum[valid_nodes] = 0.0
                    workspace.child_index[valid_nodes] = -1
                    workspace.child_prior[valid_nodes] = 0.0

                    next_free[valid_idx] = alloc[valid] + 1

                # If out of nodes, reset tree for those envs.
                invalid = need_idx[~valid]
                if invalid.numel() > 0:
                    start = workspace.root_offsets[invalid]
                    for s in start.tolist():
                        e = s + max_nodes_per_root
                        workspace.child_index[s:e] = -1
                        workspace.child_prior[s:e] = 0.0
                        workspace.node_visit[s:e] = 0.0
                        workspace.node_value_sum[s:e] = 0.0
                        workspace.node_expanded[s:e] = False
                        workspace.node_terminal[s:e] = False
                        workspace.body_age[s:e] = -1
                        workspace.length[s:e] = 0
                        workspace.head_x[s:e] = 0
                        workspace.head_y[s:e] = 0
                        workspace.direction[s:e] = 0
                        workspace.food_x[s:e] = 0
                        workspace.food_y[s:e] = 0
                        workspace.steps_since_food[s:e] = 0
                        workspace.rng_state[s:e] = 0
                    roots[invalid] = start
                    next_free[invalid] = start + 1

                child_nodes = workspace.child_index[root_nodes, actions.to(torch.int64)]

            roots[active_idx] = child_nodes

        if status_hook is not None and steps_since_status >= status_interval:
            if active_idx.numel() > 0:
                last_reward = float(ep_rewards[active_idx[0]].item())
                last_max_len = int(ep_max_lens.max().item())
            else:
                last_reward = 0.0
                last_max_len = 0
            status_hook(total_steps, last_reward, last_max_len)
            if progress_hook is not None:
                progress_hook(total_steps, ep_rewards, ep_lengths, ep_max_lens, death_types)
            steps_since_status = 0

    examples_all = []
    for i in range(batch_size):
        ep_len = int(ep_lengths[i].item())
        if ep_len <= 0:
            continue
        outcome = outcome_value_tensor(state["length"][i].unsqueeze(0), config.grid_w * config.grid_h).item()
        obs_cpu = obs_buf[i, :ep_len].detach().cpu().numpy()
        policy_cpu = policy_buf[i, :ep_len].detach().cpu().numpy()
        for j in range(ep_len):
            examples_all.append((obs_cpu[j], policy_cpu[j], float(outcome)))

    return (
        examples_all,
        ep_rewards.detach().cpu().tolist(),
        ep_lengths.detach().cpu().tolist(),
        ep_max_lens.detach().cpu().tolist(),
        death_types.detach().cpu().tolist(),
        total_steps,
    )

def eval_episode_gpu(
    model: nn.Module,
    device: torch.device,
    config: SnekConfig,
    sims: int,
    c_puct: float,
    rng: np.random.Generator,
    seed: int,
    max_steps: int,
    max_nodes_per_root: int,
    max_depth: int,
):
    state = init_state(1, config, device, seed)
    done = False
    steps = 0
    ep_reward = 0.0
    ep_max_len = int(state["length"][0].item())

    workspace = make_mcts_workspace(1, max_nodes_per_root, max_depth, config, device)
    dx = workspace.dx
    dy = workspace.dy
    opposite = workspace.opposite

    while not done and steps < max_steps:
        counts = mcts_search_batch(
            model,
            state,
            config,
            sims,
            c_puct,
            dirichlet_alpha=0.0,
            dirichlet_eps=0.0,
            device=device,
            max_nodes_per_root=max_nodes_per_root,
            max_depth=max_depth,
            workspace=workspace,
        )
        counts_cpu = counts.detach().cpu().numpy()[0]
        action = int(counts_cpu.argmax()) if counts_cpu.sum() > 0 else 0

        next_state, rewards, done_t, info = step_state(
            state, torch.tensor([action], device=device), config, dx, dy, opposite
        )
        state = next_state
        ep_reward += float(rewards[0].item())
        steps += 1
        ep_max_len = max(ep_max_len, int(info["length"][0].item()))
        done = bool(done_t[0].item())

    return ep_max_len, ep_reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=12)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--mcts-sims", type=int, default=64)
    parser.add_argument("--mcts-max-nodes", type=int, default=512)
    parser.add_argument("--mcts-max-depth", type=int, default=32)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-eps", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temp-threshold", type=int, default=30)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--selfplay-batch", type=int, default=8)
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
    parser.add_argument("--clean-logs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--out-dir", type=str, default="rl_out_az_gpu")
    parser.add_argument("--live-plot", action="store_true")
    parser.add_argument("--plot-refresh-ms", type=int, default=1000)
    parser.add_argument("--plot-no-stream", action="store_true")
    parser.add_argument("--stop-file", type=str, default="rl_out_az_gpu/stop.txt")
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

    global AMP_ENABLED
    AMP_ENABLED = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)
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
        ckpt_path = out_dir / f"checkpoint_az_gpu{ckpt_suffix}.pt"
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
        if torch.is_tensor(vals):
            if vals.numel() == 0:
                return 0.0
            return float(vals.float().mean().item())
        return float(np.mean(vals)) if vals else 0.0

    def _max_vals(vals):
        if torch.is_tensor(vals):
            if vals.numel() == 0:
                return 0.0
            return float(vals.max().item())
        return float(max(vals)) if vals else 0.0

    def log_progress(step_progress, ep_rewards, ep_lengths, ep_max_lens, death_types):
        nonlocal last_log, last_log_time, last_log_steps, best_train_rate, last_best_len, last_best_time, best_eval_rate
        progress_steps = global_step + int(step_progress)
        if progress_steps - last_log < args.log_interval:
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
    while global_step < target_steps:
        model.eval()
        seed = py_rng.randrange(1 << 30)

        (
            examples_all,
            ep_rewards,
            ep_lengths,
            ep_max_lens,
            death_types,
            steps_taken,
        ) = self_play_batch_gpu(
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
            args.selfplay_batch,
            args.mcts_max_nodes,
            args.mcts_max_depth,
            status_hook=status_update,
            status_interval=10,
            progress_hook=log_progress,
        )

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
            if death_type == 0:
                death_wall_window += 1
            elif death_type == 1:
                death_self_window += 1

        last_ep_len = int(ep_lengths[-1]) if ep_lengths else 0
        last_ep_reward = float(ep_rewards[-1]) if ep_rewards else 0.0
        last_ep_max_len = int(ep_max_lens[-1]) if ep_max_lens else 0

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
            model.train()
            for _ in range(args.train_steps):
                if len(replay) < args.batch_size:
                    break
                b_obs, b_policy, b_value = replay.sample(args.batch_size, np_rng)
                obs_t = torch.from_numpy(b_obs).to(device)
                policy_t = torch.from_numpy(b_policy).to(device)
                value_t = torch.from_numpy(b_value).to(device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                    logits, value_pred = model(obs_t)
                    log_probs = F.log_softmax(logits, dim=1)
                    policy_loss = -(policy_t * log_probs).sum(dim=1).mean()
                    value_loss = F.mse_loss(value_pred.squeeze(1), value_t)
                    loss = policy_loss + args.value_loss_weight * value_loss

                if AMP_ENABLED:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                loss_val = float(loss.item())
        else:
            loss_val = 0.0

        if log_ready and (global_step - last_eval >= args.eval_interval):
            try:
                eval_sims = args.eval_sims if args.eval_sims > 0 else args.mcts_sims
                eval_max_len = 0
                total_eval_reward = 0.0
                model.eval()
                for _ in range(args.eval_episodes):
                    seed = py_rng.randrange(1 << 30)
                    ep_max_len, ep_reward = eval_episode_gpu(
                        model,
                        device,
                        config,
                        eval_sims,
                        args.c_puct,
                        np_rng,
                        seed,
                        args.eval_max_steps,
                        args.mcts_max_nodes,
                        args.mcts_max_depth,
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
                _save_checkpoint_async(_to_cpu(ckpt), out_dir / f"checkpoint_az_gpu{ckpt_suffix}.pt", save_state)
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
        _save_checkpoint_async(_to_cpu(ckpt), out_dir / f"checkpoint_az_gpu{ckpt_suffix}.pt", save_state)
    except Exception as exc:
        _log_error(error_log, f"final checkpoint failed: {exc}")


if __name__ == "__main__":
    main()

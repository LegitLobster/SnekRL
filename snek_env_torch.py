import random
from typing import List, Optional, Tuple

import torch

from snek_env import SnekConfig


def _sample_board_size(current_size: int, max_size: int, decay: float) -> int:
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


def _neighbors(pos: Tuple[int, int], w: int, h: int, blocked: set) -> List[Tuple[int, int]]:
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


def _random_snake(config: SnekConfig, min_len: int, max_len: int, max_tries: int):
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
            snake = list(reversed(path))  # head -> tail
            if len(snake) > 1:
                hx, hy = snake[0]
                nx, ny = snake[1]
                direction = (hx - nx, hy - ny)
            else:
                direction = (1, 0)
            return snake, direction
    return None, None


def _is_reachable(config: SnekConfig, snake: List[Tuple[int, int]], target: Tuple[int, int]) -> bool:
    w, h = config.grid_w, config.grid_h
    start = snake[0]
    if start == target:
        return True
    blocked = set(snake)
    queue = [start]
    visited = {start}
    while queue:
        x, y = queue.pop(0)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            pos = (nx, ny)
            if pos in visited or pos in blocked:
                continue
            if pos == target:
                return True
            visited.add(pos)
            queue.append(pos)
    return False


def _sample_food(config: SnekConfig, snake: List[Tuple[int, int]], ensure_reachable: bool, attempt_limit: int) -> Tuple[int, int]:
    grid_cells = config.grid_w * config.grid_h
    if len(snake) >= grid_cells:
        return snake[0]
    attempts = 0
    while True:
        pos = (random.randrange(config.grid_w), random.randrange(config.grid_h))
        if pos not in snake:
            if not ensure_reachable:
                return pos
            if _is_reachable(config, snake, pos):
                return pos
            attempts += 1
            if attempts >= attempt_limit:
                return pos


class TorchSnekBatch:
    def __init__(self, n_envs: int, config: SnekConfig, max_grid: int, device: torch.device, random_start: dict):
        self.n_envs = n_envs
        self.device = device
        self.max_grid = int(max_grid)
        self.config = config
        self.random_start = dict(random_start)
        self.base_grid = config.grid_w

        self.age = torch.zeros((n_envs, self.max_grid, self.max_grid), device=device, dtype=torch.int64)
        self.head_x = torch.zeros((n_envs,), device=device, dtype=torch.int64)
        self.head_y = torch.zeros((n_envs,), device=device, dtype=torch.int64)
        self.dir = torch.zeros((n_envs,), device=device, dtype=torch.int64)
        self.length = torch.full((n_envs,), 3, device=device, dtype=torch.int64)
        self.step_idx = torch.zeros((n_envs,), device=device, dtype=torch.int64)
        self.food_x = torch.zeros((n_envs,), device=device, dtype=torch.int64)
        self.food_y = torch.zeros((n_envs,), device=device, dtype=torch.int64)
        self.score = torch.zeros((n_envs,), device=device, dtype=torch.int64)
        self.size = torch.full((n_envs,), self.base_grid, device=device, dtype=torch.int64)
        self.steps_since_food = torch.zeros((n_envs,), device=device, dtype=torch.int64)
        self.long_start_flags = [False for _ in range(n_envs)]
        self.enable_long_start = False

    def update_config(self, config: SnekConfig, random_start: Optional[dict] = None):
        self.config = config
        self.base_grid = config.grid_w
        if random_start is not None:
            self.random_start = dict(random_start)

    def _dir_to_idx(self, direction: Tuple[int, int]) -> int:
        dx, dy = direction
        if dx == 0 and dy == -1:
            return 0
        if dx == 0 and dy == 1:
            return 1
        if dx == -1 and dy == 0:
            return 2
        return 3

    def _reset_single(self, i: int):
        prob = float(self.random_start.get("prob", 0.0))
        min_len_frac = float(self.random_start.get("min_len", 0.3))
        max_len_frac = float(self.random_start.get("max_len", 0.7))
        max_tries = int(self.random_start.get("max_tries", 200))
        max_grid = int(self.random_start.get("max_grid", self.base_grid))
        size_decay = float(self.random_start.get("size_decay", 0.5))

        long_start = False
        size = self.base_grid
        snake = None
        direction = (1, 0)

        if self.enable_long_start and prob > 0.0 and random.random() < prob:
            size = _sample_board_size(self.base_grid, max_grid, size_decay)
            cfg = SnekConfig(grid_w=size, grid_h=size)
            grid_cells = size * size
            min_len = max(3, int(grid_cells * min_len_frac))
            max_len = max(3, int(grid_cells * max_len_frac))
            snake, direction = _random_snake(cfg, min_len, max_len, max_tries)
            if snake is not None:
                long_start = True

        if snake is None:
            size = self.base_grid
            cx, cy = size // 2, size // 2
            snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
            direction = (1, 0)
            long_start = False

        length = len(snake)
        self.size[i] = size
        self.length[i] = length
        self.step_idx[i] = length
        self.score[i] = max(0, length - 3)
        self.steps_since_food[i] = 0

        self.head_x[i] = snake[0][0]
        self.head_y[i] = snake[0][1]
        self.dir[i] = self._dir_to_idx(direction)

        self.age[i].zero_()
        for j, (x, y) in enumerate(snake):
            self.age[i, y, x] = length - j

        cfg = SnekConfig(grid_w=size, grid_h=size)
        food = _sample_food(cfg, snake, ensure_reachable=long_start, attempt_limit=200)
        self.food_x[i] = food[0]
        self.food_y[i] = food[1]
        self.long_start_flags[i] = long_start

    def _reset_all(self):
        for i in range(self.n_envs):
            self._reset_single(i)

    def reset_all(self):
        self._reset_all()
        return self.get_obs()

    def reset_mask(self, done_mask: torch.Tensor):
        idxs = done_mask.nonzero(as_tuple=False).flatten().tolist()
        for i in idxs:
            self._reset_single(i)

    def get_obs(self):
        t = self.step_idx.view(self.n_envs, 1, 1)
        length = self.length.view(self.n_envs, 1, 1)
        occupied = (self.age > 0) & ((t - self.age) < length)

        head = torch.zeros_like(self.age, dtype=torch.bool)
        idx = torch.arange(self.n_envs, device=self.device)
        head[idx, self.head_y, self.head_x] = True
        body = occupied & ~head

        food = torch.zeros_like(self.age, dtype=torch.bool)
        food[idx, self.food_y, self.food_x] = True

        obs = torch.zeros((self.n_envs, 3, self.max_grid, self.max_grid), device=self.device, dtype=torch.float32)
        obs[:, 0] = body.float()
        obs[:, 1] = head.float()
        obs[:, 2] = food.float()
        return obs

    def step(self, actions: torch.Tensor):
        actions = actions.to(self.device, dtype=torch.int64)
        reverse = torch.tensor([1, 0, 3, 2], device=self.device, dtype=torch.int64)
        cur_dir = self.dir
        new_dir = torch.where(actions == reverse[cur_dir], cur_dir, actions)
        self.dir = new_dir

        dx = torch.tensor([0, 0, -1, 1], device=self.device, dtype=torch.int64)
        dy = torch.tensor([-1, 1, 0, 0], device=self.device, dtype=torch.int64)
        new_x = self.head_x + dx[new_dir]
        new_y = self.head_y + dy[new_dir]

        size = self.size
        out = (new_x < 0) | (new_x >= size) | (new_y < 0) | (new_y >= size)

        nx = new_x.clamp(0, self.max_grid - 1)
        ny = new_y.clamp(0, self.max_grid - 1)
        idx = torch.arange(self.n_envs, device=self.device)
        age_nh = self.age[idx, ny, nx]

        t_next = self.step_idx + 1
        will_grow = (~out) & (new_x == self.food_x) & (new_y == self.food_y)
        length_prime = self.length + will_grow.to(torch.int64)
        body_collision = (age_nh > 0) & ((t_next - age_nh) < length_prime) & (~out)
        terminated = out | body_collision

        grid_cells = size * size
        step_penalty = -1.0 / grid_cells.to(torch.float32)
        reward = step_penalty
        reward = torch.where(terminated, torch.tensor(self.config.death_penalty, device=self.device), reward)
        if self.config.zero_out_on_death:
            reward = reward - (self.score.to(torch.float32) * float(self.config.food_reward)) * terminated.to(torch.float32)

        eat_reward = torch.tensor(self.config.food_reward, device=self.device)
        reward = torch.where(will_grow & ~terminated, eat_reward, reward)

        win = (~terminated) & will_grow & (length_prime >= grid_cells)
        reward = torch.where(win, reward + float(self.config.win_reward), reward)
        terminated = terminated | win

        self.step_idx = torch.where(terminated, self.step_idx, t_next)
        self.length = torch.where(terminated, self.length, length_prime)
        self.score = torch.where(terminated, self.score, self.score + will_grow.to(torch.int64))
        self.steps_since_food = torch.where(
            terminated,
            self.steps_since_food,
            torch.where(will_grow, torch.zeros_like(self.steps_since_food), self.steps_since_food + 1),
        )

        valid = ~terminated
        if valid.any():
            v_idx = idx[valid]
            self.head_x[valid] = new_x[valid]
            self.head_y[valid] = new_y[valid]
            self.age[v_idx, new_y[valid], new_x[valid]] = self.step_idx[valid]

        grow_mask = (will_grow & ~terminated).detach().cpu().numpy()
        if grow_mask.any():
            for i in range(self.n_envs):
                if not grow_mask[i]:
                    continue
                size_i = int(self.size[i].item())
                cfg = SnekConfig(grid_w=size_i, grid_h=size_i)
                t_i = int(self.step_idx[i].item())
                length_i = int(self.length[i].item())
                age_i = self.age[i].detach().cpu()
                occupied = (age_i > 0) & ((t_i - age_i) < length_i)
                snake = []
                ys, xs = torch.where(occupied)
                for x, y in zip(xs.tolist(), ys.tolist()):
                    snake.append((x, y))
                food = _sample_food(cfg, snake, ensure_reachable=False, attempt_limit=200)
                self.food_x[i] = food[0]
                self.food_y[i] = food[1]

        obs = self.get_obs()
        return obs, reward, terminated, self.length

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np


@dataclass
class SnekConfig:
    grid_w: int = 12
    grid_h: int = 12
    max_no_food_steps: Optional[int] = None
    step_penalty: float = -0.1
    food_reward: float = 10.0
    death_penalty: float = -10.0
    distance_reward_scale: float = 0.0
    win_reward: float = 10.0
    zero_out_on_death: bool = False


class SnekEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 12}

    def __init__(self, config: Optional[SnekConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.config = config or SnekConfig()
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(4)
        # Channels-first: 3 x H x W
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, self.config.grid_h, self.config.grid_w),
            dtype=np.float32,
        )

        self._rng = random.Random()
        self._snake = []
        self._direction = (1, 0)
        self._food = (0, 0)
        self._steps_since_food = 0
        self._score = 0
        self._ensure_reachable_food = False
        self._attempt_limit = 50

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng.seed(seed)
        return [seed]

    def _random_food(self):
        if len(self._snake) >= self.config.grid_w * self.config.grid_h:
            return self._food
        attempts = 0
        while True:
            pos = (self._rng.randrange(self.config.grid_w), self._rng.randrange(self.config.grid_h))
            if pos not in self._snake:
                if not self._ensure_reachable_food:
                    return pos
                if self._is_reachable(pos):
                    return pos
                attempts += 1
                if attempts >= self._attempt_limit:
                    return pos

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)

        start = (self.config.grid_w // 2, self.config.grid_h // 2)
        self._snake = [start, (start[0] - 1, start[1]), (start[0] - 2, start[1])]
        self._direction = (1, 0)
        self._ensure_reachable_food = bool(options.get("ensure_reachable_food", False)) if options else False
        self._attempt_limit = int(options.get("food_attempt_limit", 50)) if options else 50

        if options and "snake" in options:
            snake = list(options["snake"])
            if snake:
                self._snake = snake
            if "direction" in options:
                self._direction = tuple(options["direction"])

        if options and "food" in options and options["food"] is not None:
            self._food = tuple(options["food"])
        else:
            self._food = self._random_food()
        self._steps_since_food = 0
        self._score = max(0, len(self._snake) - 3)

        return self._get_obs(), {}

    def step(self, action: int):
        # 0=up,1=down,2=left,3=right
        if action == 0:
            new_dir = (0, -1)
        elif action == 1:
            new_dir = (0, 1)
        elif action == 2:
            new_dir = (-1, 0)
        else:
            new_dir = (1, 0)

        # Prevent reverse
        if (new_dir[0] != -self._direction[0]) or (new_dir[1] != -self._direction[1]):
            self._direction = new_dir

        head_x, head_y = self._snake[0]
        old_dist = abs(self._food[0] - head_x) + abs(self._food[1] - head_y)
        new_head = (head_x + self._direction[0], head_y + self._direction[1])

        terminated = False
        grid_cells = self.config.grid_w * self.config.grid_h
        reward = -1.0 / max(1, grid_cells)

        if (
            new_head[0] < 0
            or new_head[0] >= self.config.grid_w
            or new_head[1] < 0
            or new_head[1] >= self.config.grid_h
            or new_head in self._snake
        ):
            terminated = True
            reward = self.config.death_penalty
            if self.config.zero_out_on_death and self._score > 0:
                reward -= self._score * self.config.food_reward
        else:
            self._snake.insert(0, new_head)
            if new_head == self._food:
                reward = self.config.food_reward
                self._score += 1
                if len(self._snake) >= (self.config.grid_w * self.config.grid_h):
                    terminated = True
                    reward += self.config.win_reward
                else:
                    self._food = self._random_food()
                    self._steps_since_food = 0
            else:
                self._snake.pop()
                self._steps_since_food += 1

        if not terminated and len(self._snake) >= (self.config.grid_w * self.config.grid_h):
            terminated = True
            reward += self.config.win_reward

        if self.config.max_no_food_steps is None or self.config.max_no_food_steps <= 0:
            truncated = False
        else:
            truncated = self._steps_since_food >= self.config.max_no_food_steps
        obs = self._get_obs()

        info = {"length": len(self._snake)}
        return obs, reward, terminated, truncated, info

    def _is_reachable(self, target):
        w, h = self.config.grid_w, self.config.grid_h
        start = self._snake[0]
        if start == target:
            return True
        blocked = set(self._snake)
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

    def _get_obs(self):
        h, w = self.config.grid_h, self.config.grid_w
        obs = np.zeros((3, h, w), dtype=np.float32)

        for (x, y) in self._snake[1:]:
            if 0 <= x < w and 0 <= y < h:
                obs[0, y, x] = 1.0

        head_x, head_y = self._snake[0]
        if 0 <= head_x < w and 0 <= head_y < h:
            obs[1, head_y, head_x] = 1.0

        food_x, food_y = self._food
        if 0 <= food_x < w and 0 <= food_y < h:
            obs[2, food_y, food_x] = 1.0

        return obs

    def render(self):
        if self.render_mode != "rgb_array":
            return None

        cell = 16
        h, w = self.config.grid_h, self.config.grid_w
        img = np.zeros((h * cell, w * cell, 3), dtype=np.uint8)

        # Background
        img[:] = (14, 20, 24)

        # Draw food
        fx, fy = self._food
        img[fy * cell : (fy + 1) * cell, fx * cell : (fx + 1) * cell] = (235, 88, 88)

        # Draw snake body
        for (x, y) in self._snake[1:]:
            img[y * cell : (y + 1) * cell, x * cell : (x + 1) * cell] = (78, 211, 134)

        # Draw head
        hx, hy = self._snake[0]
        img[hy * cell : (hy + 1) * cell, hx * cell : (hx + 1) * cell] = (52, 180, 110)

        return img
        self._ensure_reachable_food = bool(options.get("ensure_reachable_food", False)) if options else False
        self._attempt_limit = int(options.get("food_attempt_limit", 50)) if options else 50

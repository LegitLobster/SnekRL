import random
from typing import List, Optional, Tuple

import numpy as np

from snek_env import SnekConfig


ACTION_DIRS = {
    0: (0, -1),
    1: (0, 1),
    2: (-1, 0),
    3: (1, 0),
}
DIR_TO_ACTION = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
REVERSE_ACTION = {0: 1, 1: 0, 2: 3, 3: 2}


class SnekState:
    def __init__(self, config: SnekConfig, rng_state: Optional[tuple] = None):
        self.config = config
        self._rng = random.Random()
        if rng_state is not None:
            self._rng.setstate(rng_state)
        self._snake: List[Tuple[int, int]] = []
        self._direction: Tuple[int, int] = (1, 0)
        self._food: Tuple[int, int] = (0, 0)
        self._steps_since_food = 0
        self._score = 0
        self.last_wall = False
        self.last_self = False

    @classmethod
    def new(cls, config: SnekConfig, seed: Optional[int] = None) -> "SnekState":
        state = cls(config)
        if seed is not None:
            state._rng.seed(seed)
        state.reset()
        return state

    def clone(self) -> "SnekState":
        clone = SnekState(self.config, self._rng.getstate())
        clone._snake = list(self._snake)
        clone._direction = self._direction
        clone._food = self._food
        clone._steps_since_food = self._steps_since_food
        clone._score = self._score
        clone.last_wall = self.last_wall
        clone.last_self = self.last_self
        return clone

    def reset(self) -> None:
        w, h = self.config.grid_w, self.config.grid_h
        start = (w // 2, h // 2)
        self._snake = [start, (start[0] - 1, start[1]), (start[0] - 2, start[1])]
        self._direction = (1, 0)
        self._food = self._random_food()
        self._steps_since_food = 0
        self._score = max(0, len(self._snake) - 3)
        self.last_wall = False
        self.last_self = False

    @property
    def length(self) -> int:
        return len(self._snake)

    def legal_actions(self) -> List[int]:
        current = DIR_TO_ACTION.get(self._direction, 3)
        reverse_action = REVERSE_ACTION[current]
        return [a for a in range(4) if a != reverse_action]

    def step(self, action: int):
        if action not in ACTION_DIRS:
            raise ValueError(f"invalid action: {action}")

        new_dir = ACTION_DIRS[action]
        # Prevent reverse.
        if new_dir[0] != -self._direction[0] or new_dir[1] != -self._direction[1]:
            self._direction = new_dir

        head_x, head_y = self._snake[0]
        new_head = (head_x + self._direction[0], head_y + self._direction[1])

        terminated = False
        grid_cells = self.config.grid_w * self.config.grid_h
        reward = -1.0 / max(1, grid_cells)

        hit_wall = (
            new_head[0] < 0
            or new_head[0] >= self.config.grid_w
            or new_head[1] < 0
            or new_head[1] >= self.config.grid_h
        )
        hit_self = new_head in self._snake
        self.last_wall = bool(hit_wall)
        self.last_self = bool(hit_self)

        if hit_wall or hit_self:
            terminated = True
            reward = self.config.death_penalty
            if self.config.zero_out_on_death and self._score > 0:
                reward -= self._score * self.config.food_reward
        else:
            self._snake.insert(0, new_head)
            if new_head == self._food:
                reward = self.config.food_reward
                self._score += 1
                if len(self._snake) >= grid_cells:
                    terminated = True
                    reward += self.config.win_reward
                else:
                    self._food = self._random_food()
                    self._steps_since_food = 0
            else:
                self._snake.pop()
                self._steps_since_food += 1

        if not terminated and len(self._snake) >= grid_cells:
            terminated = True
            reward += self.config.win_reward

        if self.config.max_no_food_steps is None or self.config.max_no_food_steps <= 0:
            truncated = False
        else:
            truncated = self._steps_since_food >= self.config.max_no_food_steps
        if truncated:
            terminated = True

        info = {"length": len(self._snake)}
        if terminated:
            if hit_wall:
                info["death"] = "wall"
            elif hit_self:
                info["death"] = "self"
        return reward, terminated, info

    def _random_food(self) -> Tuple[int, int]:
        if len(self._snake) >= self.config.grid_w * self.config.grid_h:
            return self._food
        while True:
            pos = (self._rng.randrange(self.config.grid_w), self._rng.randrange(self.config.grid_h))
            if pos not in self._snake:
                return pos

    def to_obs(self) -> np.ndarray:
        h, w = self.config.grid_h, self.config.grid_w
        obs = np.zeros((4, h, w), dtype=np.float32)

        for (x, y) in self._snake[1:]:
            if 0 <= x < w and 0 <= y < h:
                obs[0, y, x] = 1.0

        head_x, head_y = self._snake[0]
        if 0 <= head_x < w and 0 <= head_y < h:
            obs[1, head_y, head_x] = 1.0

        food_x, food_y = self._food
        if 0 <= food_x < w and 0 <= food_y < h:
            obs[2, food_y, food_x] = 1.0

        obs[3, 0, :] = 1.0
        obs[3, h - 1, :] = 1.0
        obs[3, :, 0] = 1.0
        obs[3, :, w - 1] = 1.0

        return obs


def outcome_value(state: SnekState) -> float:
    grid_cells = state.config.grid_w * state.config.grid_h
    denom = max(1, grid_cells - 3)
    progress = max(0.0, min(1.0, (state.length - 3) / float(denom)))
    return progress * 2.0 - 1.0

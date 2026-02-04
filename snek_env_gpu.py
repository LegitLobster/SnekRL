import torch

from snek_env import SnekConfig


class SnekEnvGPU:
    def __init__(self, n_envs: int, config: SnekConfig, device: torch.device):
        self.n_envs = int(n_envs)
        self.config = config
        self.device = device
        self.grid_w = int(config.grid_w)
        self.grid_h = int(config.grid_h)
        self.grid_cells = self.grid_w * self.grid_h
        self._base_reward = torch.tensor(-1.0 / float(self.grid_cells), dtype=torch.float32, device=self.device)
        self._death_penalty_t = torch.tensor(float(config.death_penalty), dtype=torch.float32, device=self.device)
        self._food_reward_t = torch.tensor(float(config.food_reward), dtype=torch.float32, device=self.device)
        self._win_reward_t = torch.tensor(float(config.win_reward), dtype=torch.float32, device=self.device)
        self._reset_len = torch.tensor(3, dtype=torch.int16, device=self.device)

        self.body_age = torch.full(
            (self.n_envs, self.grid_h, self.grid_w),
            -1,
            dtype=torch.int16,
            device=self.device,
        )
        self.length = torch.full((self.n_envs,), 3, dtype=torch.int16, device=self.device)
        self.head_x = torch.zeros((self.n_envs,), dtype=torch.int16, device=self.device)
        self.head_y = torch.zeros((self.n_envs,), dtype=torch.int16, device=self.device)
        self.direction = torch.zeros((self.n_envs,), dtype=torch.int16, device=self.device)
        self.food_x = torch.zeros((self.n_envs,), dtype=torch.int16, device=self.device)
        self.food_y = torch.zeros((self.n_envs,), dtype=torch.int16, device=self.device)
        self.steps_since_food = torch.zeros((self.n_envs,), dtype=torch.int32, device=self.device)

        wall = torch.zeros((self.grid_h, self.grid_w), dtype=torch.float32, device=self.device)
        wall[0, :] = 1.0
        wall[self.grid_h - 1, :] = 1.0
        wall[:, 0] = 1.0
        wall[:, self.grid_w - 1] = 1.0
        self.wall_mask = wall

        self._dx = torch.tensor([0, 0, -1, 1], dtype=torch.int16, device=self.device)
        self._dy = torch.tensor([-1, 1, 0, 0], dtype=torch.int16, device=self.device)
        self._opposite = torch.tensor([1, 0, 3, 2], dtype=torch.int16, device=self.device)

        cx = self.grid_w // 2
        cy = self.grid_h // 2
        reset_body = torch.full(
            (self.grid_h, self.grid_w),
            -1,
            dtype=self.body_age.dtype,
            device=self.device,
        )
        reset_body[cy, cx] = 0
        reset_body[cy, cx - 1] = 1
        reset_body[cy, cx - 2] = 2
        self._reset_body = reset_body
        self._reset_head_x = torch.tensor(cx, dtype=torch.int16, device=self.device)
        self._reset_head_y = torch.tensor(cy, dtype=torch.int16, device=self.device)
        self._reset_dir = torch.tensor(3, dtype=torch.int16, device=self.device)

        self.reset()

    def _resample_food(self, mask: torch.Tensor):
        mask = mask.to(device=self.device, dtype=torch.bool)
        occupancy = (self.body_age >= 0).view(self.n_envs, -1)
        scores = torch.rand((self.n_envs, self.grid_cells), device=self.device)
        scores = scores.masked_fill(occupancy, -1.0)
        new_idx = scores.argmax(dim=1)
        new_x = (new_idx % self.grid_w).to(self.food_x.dtype)
        new_y = (new_idx // self.grid_w).to(self.food_y.dtype)
        self.food_x = torch.where(mask, new_x, self.food_x)
        self.food_y = torch.where(mask, new_y, self.food_y)

    def _reset_mask(self, mask: torch.Tensor):
        mask = mask.to(device=self.device, dtype=torch.bool)
        self.body_age = torch.where(mask[:, None, None], self._reset_body, self.body_age)
        self.length = torch.where(mask, self._reset_len, self.length)
        self.head_x = torch.where(mask, self._reset_head_x, self.head_x)
        self.head_y = torch.where(mask, self._reset_head_y, self.head_y)
        self.direction = torch.where(mask, self._reset_dir, self.direction)
        self.steps_since_food = torch.where(mask, torch.zeros_like(self.steps_since_food), self.steps_since_food)
        self._resample_food(mask)

    def reset(self):
        mask = torch.ones((self.n_envs,), dtype=torch.bool, device=self.device)
        self._reset_mask(mask)
        return self._obs()

    def step(self, actions: torch.Tensor):
        actions = actions.to(self.device).to(torch.int16)
        opp = self._opposite[self.direction.to(torch.int64)]
        is_reverse = actions == opp
        new_dir = torch.where(is_reverse, self.direction, actions)
        self.direction = new_dir

        dx = self._dx[new_dir.to(torch.int64)]
        dy = self._dy[new_dir.to(torch.int64)]
        new_head_x = self.head_x + dx
        new_head_y = self.head_y + dy

        hit_wall = (
            (new_head_x < 0)
            | (new_head_x >= self.grid_w)
            | (new_head_y < 0)
            | (new_head_y >= self.grid_h)
        )

        valid = ~hit_wall
        clamped_x = new_head_x.clamp(0, self.grid_w - 1)
        clamped_y = new_head_y.clamp(0, self.grid_h - 1)
        linear = clamped_y.to(torch.int64) * self.grid_w + clamped_x.to(torch.int64)
        flat_body = (self.body_age >= 0).view(self.n_envs, -1)
        hit_self = flat_body.gather(1, linear.view(-1, 1)).squeeze(1) & valid

        terminated = hit_wall | hit_self
        rewards = self._base_reward.expand(self.n_envs)
        rewards = torch.where(terminated, self._death_penalty_t, rewards)
        if self.config.zero_out_on_death:
            score = (self.length - 3).to(torch.float32)
            rewards = torch.where(
                terminated & (score > 0),
                rewards - score * self._food_reward_t,
                rewards,
            )

        alive = ~terminated
        ate = alive & (new_head_x == self.food_x) & (new_head_y == self.food_y)
        length_new = self.length + ate.to(self.length.dtype)
        win = alive & (length_new >= self.grid_cells)

        rewards = torch.where(win, self._food_reward_t + self._win_reward_t, rewards)
        ate_no_win = ate & ~win
        rewards = torch.where(ate_no_win, self._food_reward_t, rewards)

        if self.config.max_no_food_steps is not None and self.config.max_no_food_steps > 0:
            self.steps_since_food = torch.where(alive, self.steps_since_food + 1, self.steps_since_food)
            self.steps_since_food = torch.where(ate, torch.zeros_like(self.steps_since_food), self.steps_since_food)
            truncated = alive & (self.steps_since_food >= int(self.config.max_no_food_steps))
        else:
            truncated = torch.zeros_like(alive)

        done = terminated | win | truncated

        body = self.body_age
        body = torch.where((body >= 0) & alive[:, None, None], body + 1, body)

        head_mask = torch.zeros((self.n_envs, self.grid_cells), dtype=torch.bool, device=self.device)
        head_mask.scatter_(1, linear.view(-1, 1), alive.view(-1, 1))
        head_mask = head_mask.view(self.n_envs, self.grid_h, self.grid_w)
        body = torch.where(head_mask, torch.zeros_like(body), body)

        length_new = torch.where(alive, length_new, self.length)
        body = torch.where(body >= length_new.view(-1, 1, 1), -1, body)
        self.body_age = body
        self.length = length_new
        self.head_x = torch.where(alive, new_head_x, self.head_x)
        self.head_y = torch.where(alive, new_head_y, self.head_y)

        self._resample_food(ate_no_win)
        self._reset_mask(done)

        obs = self._obs()
        info = {
            "length": length_new,
            "death_wall": hit_wall,
            "death_self": hit_self,
        }
        return obs, rewards, done, info

    def _obs(self):
        body_mask = self.body_age >= 0
        head_mask = self.body_age == 0
        body_only = body_mask & ~head_mask

        food_mask = torch.zeros(
            (self.n_envs, self.grid_h, self.grid_w),
            dtype=torch.bool,
            device=self.device,
        )
        idx = torch.arange(self.n_envs, device=self.device)
        food_mask[idx, self.food_y.to(torch.int64), self.food_x.to(torch.int64)] = True

        obs = torch.stack(
            [
                body_only.to(torch.float32),
                head_mask.to(torch.float32),
                food_mask.to(torch.float32),
                self.wall_mask.expand(self.n_envs, -1, -1),
            ],
            dim=1,
        )
        return obs

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn as nn

from snek_env import SnekConfig, SnekEnv


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


def pad_obs(obs, target_h, target_w):
    if obs.shape[1] == target_h and obs.shape[2] == target_w:
        return obs
    padded = np.zeros((obs.shape[0], target_h, target_w), dtype=obs.dtype)
    h = min(target_h, obs.shape[1])
    w = min(target_w, obs.shape[2])
    padded[:, :h, :w] = obs[:, :h, :w]
    return padded


def record_gif(qnet, out_path, config, device, target_h, target_w):
    env = SnekEnv(config, render_mode="rgb_array")
    obs, _ = env.reset()
    frames = []
    done = False

    while not done:
        obs_t = torch.as_tensor(pad_obs(obs, target_h, target_w), device=device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = torch.argmax(qnet(obs_t), dim=1).item()
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        done = terminated or truncated

    if frames:
        imageio.mimsave(out_path, frames, fps=12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--max-grid", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = SnekConfig()
    config.grid_w = args.grid
    config.grid_h = args.grid

    qnet = QNet(n_actions=4).to(device)
    state = torch.load(args.model, map_location=device)
    if "advantage.weight" in state or "value.weight" in state:
        qnet.load_state_dict(state)
    else:
        qnet.load_state_dict(state, strict=False)
        if "fc.3.weight" in state:
            adv_w = state["fc.3.weight"]
            adv_b = state.get("fc.3.bias")
            if adv_w.shape == qnet.advantage.weight.data.shape:
                qnet.advantage.weight.data.copy_(adv_w)
                if adv_b is not None:
                    qnet.advantage.bias.data.copy_(adv_b)
                value_w = adv_w.mean(dim=0, keepdim=True)
                qnet.value.weight.data.copy_(value_w)
                if adv_b is not None:
                    qnet.value.bias.data.copy_(adv_b.mean().unsqueeze(0))
    qnet.eval()

    record_gif(qnet, Path(args.out), config, device, args.max_grid, args.max_grid)


if __name__ == "__main__":
    main()

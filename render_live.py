import argparse
import time
from pathlib import Path

import numpy as np
import pygame
import torch

from snek_env import SnekConfig, SnekEnv


class QNet(torch.nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 8 * 8, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


def load_model(model, path, device):
    if path.exists():
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rl_out/latest.pt")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--reload-sec", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = SnekConfig()
    env = SnekEnv(config, render_mode="rgb_array")
    obs, _ = env.reset()

    model = QNet(n_actions=4).to(device)
    model.eval()

    model_path = Path(args.model)
    last_reload = 0.0
    load_model(model, model_path, device)

    pygame.init()
    cell = 16
    screen = pygame.display.set_mode((config.grid_w * cell, config.grid_h * cell))
    pygame.display.set_caption("Snek RL Live")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        now = time.time()
        if now - last_reload >= args.reload_sec:
            load_model(model, model_path, device)
            last_reload = now

        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = torch.argmax(model(obs_t), dim=1).item()

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

        frame = env.render()
        if frame is not None:
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from rf_spectrum_rl import RFSpectrumEnv


def run(episodes: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    env = RFSpectrumEnv(seed=seed)

    for episode in range(episodes):
        _, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        total_reward = 0.0
        detections = 0
        steps = 0
        done = False

        while not done:
            action = env.action_space.sample(rng)
            _, reward, terminated, truncated, step_info = env.step(action)
            total_reward += reward
            detections += int(step_info["selected_occupied"])
            steps += 1
            done = terminated or truncated

        print(
            f"episode={episode + 1} reward={total_reward:.1f} "
            f"detections={detections}/{steps} emitters={len(info['events'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run(episodes=args.episodes, seed=args.seed)


if __name__ == "__main__":
    main()

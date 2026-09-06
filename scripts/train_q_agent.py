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
from rf_spectrum_rl.agents import EnergyScanAgent, QLearningConfig, TabularQAgent, evaluate_agent, train_q_agent
from rf_spectrum_rl.metrics import ScanMetricTracker


def random_baseline(episodes: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    env = RFSpectrumEnv(seed=seed)
    rewards: list[float] = []
    tracker = ScanMetricTracker()

    for episode in range(episodes):
        env.reset(seed=seed + episode)
        done = False
        reward_sum = 0.0
        while not done:
            action = env.action_space.sample(rng)
            _, reward, terminated, truncated, info = env.step(action)
            tracker.update(
                reward,
                selected_occupied=bool(info["selected_occupied"]),
                occupied_channels=info["occupied_channels"],
            )
            reward_sum += reward
            done = terminated or truncated
        rewards.append(reward_sum)

    metrics = tracker.result().as_dict()
    metrics["mean_reward"] = float(np.mean(rewards))
    metrics["episodes"] = float(episodes)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="artifacts/q_agent.npz")
    args = parser.parse_args()

    env = RFSpectrumEnv(seed=args.seed)
    config = QLearningConfig(episodes=args.episodes, seed=args.seed)
    agent, rewards = train_q_agent(env, config)
    agent.save(ROOT / args.output)

    trained_metrics = evaluate_agent(RFSpectrumEnv(seed=args.seed + 1), agent, args.eval_episodes, seed=args.seed + 20_000)
    smart_scan_metrics = evaluate_agent(
        RFSpectrumEnv(seed=args.seed + 2),
        EnergyScanAgent(n_actions=env.config.scene.n_channels),
        args.eval_episodes,
        seed=args.seed + 25_000,
    )
    random_metrics = random_baseline(args.eval_episodes, seed=args.seed + 30_000)

    print(f"trained_episodes={args.episodes}")
    print(f"train_reward_first_25={np.mean(rewards[:25]):.2f}")
    print(f"train_reward_last_25={np.mean(rewards[-25:]):.2f}")
    print(
        "trained_eval "
        f"mean_reward={trained_metrics['mean_reward']:.2f} "
        f"detection_rate={trained_metrics['detection_rate']:.3f} "
        f"false_alarm_rate={trained_metrics['false_alarm_rate']:.3f} "
        f"scan_efficiency={trained_metrics['scan_efficiency']:.3f}"
    )
    print(
        "random_eval "
        f"mean_reward={random_metrics['mean_reward']:.2f} "
        f"detection_rate={random_metrics['detection_rate']:.3f} "
        f"false_alarm_rate={random_metrics['false_alarm_rate']:.3f} "
        f"scan_efficiency={random_metrics['scan_efficiency']:.3f}"
    )
    print(
        "smart_scan_eval "
        f"mean_reward={smart_scan_metrics['mean_reward']:.2f} "
        f"detection_rate={smart_scan_metrics['detection_rate']:.3f} "
        f"false_alarm_rate={smart_scan_metrics['false_alarm_rate']:.3f} "
        f"scan_efficiency={smart_scan_metrics['scan_efficiency']:.3f}"
    )
    print(f"saved={ROOT / args.output}")

    loaded = TabularQAgent.load(ROOT / args.output, seed=args.seed)
    assert loaded.n_actions == agent.n_actions


if __name__ == "__main__":
    main()

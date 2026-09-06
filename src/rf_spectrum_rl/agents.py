from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .env import RFSpectrumEnv
from .metrics import ScanMetricTracker


State = tuple[int, int, int]


def encode_observation(observation: np.ndarray, energy_bins: int = 8) -> State:
    latest = observation[-1]
    top_channel = int(np.argmax(latest))
    top_energy = float(latest[top_channel])
    energy_bin = min(energy_bins - 1, max(0, int(top_energy * energy_bins)))
    active_count = int(np.count_nonzero(latest > 0.18))
    active_bin = min(energy_bins - 1, active_count)
    return top_channel, energy_bin, active_bin


@dataclass
class QLearningConfig:
    episodes: int = 500
    learning_rate: float = 0.15
    discount: float = 0.92
    epsilon_start: float = 0.45
    epsilon_end: float = 0.05
    energy_bins: int = 8
    seed: int = 7


class TabularQAgent:
    def __init__(self, n_actions: int, energy_bins: int = 8, seed: int = 7):
        self.n_actions = n_actions
        self.energy_bins = energy_bins
        self.rng = np.random.default_rng(seed)
        self.q: dict[State, np.ndarray] = {}

    def act(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_actions))
        state = encode_observation(observation, self.energy_bins)
        return int(np.argmax(self._values(state)))

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        learning_rate: float,
        discount: float,
    ) -> None:
        state = encode_observation(observation, self.energy_bins)
        next_state = encode_observation(next_observation, self.energy_bins)
        values = self._values(state)
        next_best = 0.0 if done else float(np.max(self._values(next_state)))
        target = reward + discount * next_best
        values[action] += learning_rate * (target - values[action])

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self.q.keys()), dtype=np.int32)
        values = np.array(list(self.q.values()), dtype=np.float32)
        np.savez(output, keys=keys, values=values, n_actions=self.n_actions, energy_bins=self.energy_bins)

    @classmethod
    def load(cls, path: str | Path, seed: int = 7) -> "TabularQAgent":
        data = np.load(Path(path), allow_pickle=False)
        agent = cls(n_actions=int(data["n_actions"]), energy_bins=int(data["energy_bins"]), seed=seed)
        for key, value in zip(data["keys"], data["values"]):
            agent.q[tuple(int(part) for part in key)] = value.astype(np.float32)
        return agent

    def _values(self, state: State) -> np.ndarray:
        if state not in self.q:
            self.q[state] = np.zeros(self.n_actions, dtype=np.float32)
        return self.q[state]


class EnergyScanAgent:
    """Deterministic Smart Scan baseline that follows the strongest channel."""

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def act(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        del epsilon
        latest = observation[-1]
        return int(np.argmax(latest) % self.n_actions)


def train_q_agent(env: RFSpectrumEnv, config: QLearningConfig) -> tuple[TabularQAgent, list[float]]:
    agent = TabularQAgent(
        n_actions=env.config.scene.n_channels,
        energy_bins=config.energy_bins,
        seed=config.seed,
    )
    rewards: list[float] = []

    for episode in range(config.episodes):
        epsilon = _linear_decay(config.epsilon_start, config.epsilon_end, episode, config.episodes)
        observation, _ = env.reset(seed=config.seed + episode)
        total_reward = 0.0
        done = False

        while not done:
            action = agent.act(observation, epsilon=epsilon)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(
                observation,
                action,
                reward,
                next_observation,
                done,
                learning_rate=config.learning_rate,
                discount=config.discount,
            )
            observation = next_observation
            total_reward += reward

        rewards.append(total_reward)

    return agent, rewards


def evaluate_agent(env: RFSpectrumEnv, agent: object, episodes: int, seed: int = 10_000) -> dict[str, float]:
    episode_rewards: list[float] = []
    tracker = ScanMetricTracker()

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        done = False
        reward_sum = 0.0
        while not done:
            action = agent.act(observation, epsilon=0.0)  # type: ignore[attr-defined]
            observation, reward, terminated, truncated, info = env.step(action)
            tracker.update(
                reward,
                selected_occupied=bool(info["selected_occupied"]),
                occupied_channels=info["occupied_channels"],
            )
            reward_sum += reward
            done = terminated or truncated
        episode_rewards.append(reward_sum)

    metrics = tracker.result().as_dict()
    metrics["mean_reward"] = float(np.mean(episode_rewards))
    metrics["episodes"] = float(episodes)
    return metrics


def _linear_decay(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return end
    ratio = min(1.0, max(0.0, step / (total_steps - 1)))
    return start + ratio * (end - start)

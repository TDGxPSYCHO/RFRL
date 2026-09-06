from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    from . import spaces  # type: ignore[no-redef]

from .scene import RFScene, SceneConfig


@dataclass(frozen=True)
class SpectrumEnvConfig:
    scene: SceneConfig = field(default_factory=SceneConfig)
    window_steps: int = 8
    correct_reward: float = 1.0
    missed_signal_penalty: float = -2.0
    false_alarm_penalty: float = -0.5


class _BaseEnv(gym.Env if gym is not None else object):  # type: ignore[misc]
    pass


class RFSpectrumEnv(_BaseEnv):
    """Gymnasium-style RF sensing environment.

    Action: choose one frequency channel to sense.
    Observation: rolling spectrogram window shaped (window_steps, n_channels).
    Reward: positive for choosing an occupied channel, smaller penalty for false alarms,
    larger penalty for missing all currently occupied channels.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: SpectrumEnvConfig | None = None, seed: int | None = None):
        self.config = config or SpectrumEnvConfig()
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(self.config.scene.n_channels)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.5,
            shape=(self.config.window_steps, self.config.scene.n_channels),
            dtype=np.float32,
        )
        self._spectrogram: np.ndarray | None = None
        self._occupancy: np.ndarray | None = None
        self._scene: RFScene | None = None
        self._step_index = 0
        self._episode_seed = seed

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._episode_seed = seed

        scene_seed = int(self.rng.integers(0, np.iinfo(np.int32).max))
        scene = RFScene.random(self.config.scene, seed=scene_seed)
        self._scene = scene
        self._spectrogram, self._occupancy = scene.render(seed=scene_seed + 1)
        self._step_index = 0
        return self._observation(), {"events": scene.events}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._spectrogram is None or self._occupancy is None:
            raise RuntimeError("Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be a channel in [0, {self.config.scene.n_channels - 1}].")

        occupied_now = self._occupancy[self._step_index]
        selected_occupied = bool(occupied_now[int(action)])
        any_occupied = bool(occupied_now.any())
        selected_channel = int(action)
        active_events = []
        if self._scene is not None:
            for event in self._scene.events:
                if selected_channel in event.channels_at(self._step_index, self.config.scene.n_channels):
                    active_events.append(
                        {
                            "label": event.label,
                            "kind": event.kind.value,
                            "encrypted": event.encrypted,
                        }
                    )

        if selected_occupied:
            reward = self.config.correct_reward
        elif any_occupied:
            reward = self.config.missed_signal_penalty
        else:
            reward = self.config.false_alarm_penalty

        info = {
            "selected_channel": selected_channel,
            "selected_occupied": selected_occupied,
            "occupied_channels": np.flatnonzero(occupied_now).astype(int).tolist(),
            "selected_power": float(self._spectrogram[self._step_index, selected_channel]),
            "observed_chunk": self._observed_chunk(selected_channel),
            "active_events": active_events,
        }

        self._step_index += 1
        terminated = self._step_index >= self.config.scene.n_steps
        return self._observation(), float(reward), terminated, False, info

    def render(self) -> str:
        if self._occupancy is None:
            return "<not reset>"
        idx = min(self._step_index, self.config.scene.n_steps - 1)
        occupied = np.flatnonzero(self._occupancy[idx]).astype(int).tolist()
        return f"step={idx} occupied_channels={occupied}"

    def _observed_chunk(self, channel: int, radius: int = 1) -> np.ndarray:
        if self._spectrogram is None:
            raise RuntimeError("Call reset() before requesting an observed chunk.")
        start = max(0, channel - radius)
        stop = min(self.config.scene.n_channels, channel + radius + 1)
        return self._spectrogram[self._step_index, start:stop].astype(np.float32)

    def _observation(self) -> np.ndarray:
        if self._spectrogram is None:
            raise RuntimeError("Call reset() before requesting observations.")
        end = self._step_index + 1
        start = max(0, end - self.config.window_steps)
        window = self._spectrogram[start:end]
        if len(window) < self.config.window_steps:
            pad = np.zeros((self.config.window_steps - len(window), self.config.scene.n_channels), dtype=np.float32)
            window = np.vstack([pad, window])
        return window.astype(np.float32)

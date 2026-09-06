from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Discrete:
    """Small fallback for gymnasium.spaces.Discrete."""

    n: int

    def sample(self, rng: np.random.Generator | None = None) -> int:
        generator = rng if rng is not None else np.random.default_rng()
        return int(generator.integers(0, self.n))

    def contains(self, value: object) -> bool:
        if not isinstance(value, (int, np.integer)):
            return False
        return 0 <= int(value) < self.n


@dataclass(frozen=True)
class Box:
    """Small fallback for gymnasium.spaces.Box."""

    low: float
    high: float
    shape: tuple[int, ...]
    dtype: type

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        generator = rng if rng is not None else np.random.default_rng()
        values = generator.uniform(self.low, self.high, size=self.shape)
        return values.astype(self.dtype)

    def contains(self, value: object) -> bool:
        array = np.asarray(value)
        return array.shape == self.shape and array.min() >= self.low and array.max() <= self.high

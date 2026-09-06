from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class SignalKind(str, Enum):
    NARROWBAND = "narrowband"
    FREQUENCY_HOPPING = "frequency_hopping"
    CHIRP_LPI = "chirp_lpi"
    ENCRYPTED = "encrypted"


@dataclass(frozen=True)
class SignalEvent:
    kind: SignalKind
    start_step: int
    duration_steps: int
    center_channel: int
    bandwidth_channels: int
    power_db: float
    encrypted: bool = False
    hop_interval: int = 1
    hop_span: int = 0
    chirp_rate: float = 0.0
    label: str = ""

    def channels_at(self, step: int, n_channels: int) -> range:
        if step < self.start_step or step >= self.start_step + self.duration_steps:
            return range(0)

        center = self.center_channel
        elapsed = step - self.start_step
        if self.kind == SignalKind.FREQUENCY_HOPPING and self.hop_span > 0:
            hop_index = elapsed // max(1, self.hop_interval)
            offset = ((hop_index * 3) % (2 * self.hop_span + 1)) - self.hop_span
            center = self.center_channel + offset
        elif self.kind == SignalKind.CHIRP_LPI:
            center = self.center_channel + int(round(self.chirp_rate * elapsed))

        half = max(0, self.bandwidth_channels // 2)
        start = max(0, center - half)
        stop = min(n_channels, center + half + 1)
        return range(start, stop)


def apply_signal(
    spectrogram: np.ndarray,
    event: SignalEvent,
    rng: np.random.Generator,
) -> None:
    n_steps, n_channels = spectrogram.shape
    amplitude = 10 ** (event.power_db / 20.0)

    for step in range(n_steps):
        active_channels = list(event.channels_at(step, n_channels))
        if not active_channels:
            continue

        rolloff = rng.uniform(0.75, 1.0, size=len(active_channels))
        if event.kind == SignalKind.CHIRP_LPI:
            rolloff *= np.linspace(0.65, 1.0, len(active_channels))
        spectrogram[step, active_channels] += amplitude * rolloff

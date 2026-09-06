from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .signals import SignalEvent, SignalKind, apply_signal


@dataclass(frozen=True)
class SceneConfig:
    n_steps: int = 64
    n_channels: int = 32
    noise_floor_db: float = -35.0
    oscillator_drift_std: float = 0.015
    max_signals: int = 5


@dataclass
class RFScene:
    config: SceneConfig = field(default_factory=SceneConfig)
    events: list[SignalEvent] = field(default_factory=list)

    @classmethod
    def random(cls, config: SceneConfig, seed: int | None = None) -> "RFScene":
        rng = np.random.default_rng(seed)
        events: list[SignalEvent] = []
        n_events = int(rng.integers(2, config.max_signals + 1))

        kinds = [
            SignalKind.NARROWBAND,
            SignalKind.FREQUENCY_HOPPING,
            SignalKind.CHIRP_LPI,
            SignalKind.ENCRYPTED,
        ]
        for idx in range(n_events):
            kind = kinds[int(rng.integers(0, len(kinds)))]
            duration = int(rng.integers(config.n_steps // 4, config.n_steps))
            start = int(rng.integers(0, max(1, config.n_steps - duration + 1)))
            center = int(rng.integers(2, max(3, config.n_channels - 2)))
            bandwidth = int(rng.integers(1, 4))
            power = float(rng.uniform(-4.0, 10.0))

            events.append(
                SignalEvent(
                    kind=kind,
                    start_step=start,
                    duration_steps=duration,
                    center_channel=center,
                    bandwidth_channels=bandwidth,
                    power_db=power,
                    encrypted=kind == SignalKind.ENCRYPTED,
                    hop_interval=int(rng.integers(1, 5)),
                    hop_span=int(rng.integers(2, 7)) if kind == SignalKind.FREQUENCY_HOPPING else 0,
                    chirp_rate=float(rng.uniform(-0.35, 0.35)) if kind == SignalKind.CHIRP_LPI else 0.0,
                    label=f"emitter_{idx}",
                )
            )

        return cls(config=config, events=events)

    @classmethod
    def smart_scan_demo(cls, config: SceneConfig | None = None) -> "RFScene":
        cfg = config or SceneConfig(n_steps=72, n_channels=40, max_signals=4)
        return cls(
            config=cfg,
            events=[
                SignalEvent(
                    kind=SignalKind.FREQUENCY_HOPPING,
                    start_step=4,
                    duration_steps=58,
                    center_channel=14,
                    bandwidth_channels=2,
                    power_db=5.5,
                    hop_interval=3,
                    hop_span=6,
                    label="frequency_hopping_link",
                ),
                SignalEvent(
                    kind=SignalKind.CHIRP_LPI,
                    start_step=12,
                    duration_steps=46,
                    center_channel=28,
                    bandwidth_channels=3,
                    power_db=-1.5,
                    chirp_rate=-0.18,
                    label="chirp_lpi_emitter",
                ),
                SignalEvent(
                    kind=SignalKind.NARROWBAND,
                    start_step=20,
                    duration_steps=34,
                    center_channel=7,
                    bandwidth_channels=1,
                    power_db=8.0,
                    label="narrowband_control_signal",
                ),
                SignalEvent(
                    kind=SignalKind.ENCRYPTED,
                    start_step=36,
                    duration_steps=26,
                    center_channel=34,
                    bandwidth_channels=2,
                    power_db=4.0,
                    encrypted=True,
                    label="encrypted_signal_detect_only",
                ),
            ],
        )

    def render(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        cfg = self.config
        noise_floor = 10 ** (cfg.noise_floor_db / 20.0)
        spectrogram = rng.normal(noise_floor, noise_floor * 0.35, size=(cfg.n_steps, cfg.n_channels))

        drift = rng.normal(0.0, cfg.oscillator_drift_std, size=cfg.n_steps).cumsum()
        spectrogram *= 1.0 + drift[:, None]

        occupancy = np.zeros((cfg.n_steps, cfg.n_channels), dtype=np.int8)
        for event in self.events:
            apply_signal(spectrogram, event, rng)
            for step in range(cfg.n_steps):
                channels = list(event.channels_at(step, cfg.n_channels))
                if channels:
                    occupancy[step, channels] = 1

        spectrogram = np.maximum(spectrogram, 0.0)
        scale = float(np.percentile(spectrogram, 99))
        if scale > 0:
            spectrogram = spectrogram / scale
        return spectrogram.astype(np.float32), occupancy

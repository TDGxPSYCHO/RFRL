from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from rf_spectrum_rl import RFScene, RFSpectrumEnv, SpectrumEnvConfig
from rf_spectrum_rl.agents import EnergyScanAgent, QLearningConfig, train_q_agent
from rf_spectrum_rl.classifier import SpectrumChunkClassifier
from rf_spectrum_rl.pipeline import ClosedLoopSmartScanAgent, RoundRobinAgent, run_closed_loop
from rf_spectrum_rl.scene import SceneConfig


class FixedSceneEnv(RFSpectrumEnv):
    def __init__(self, scene: RFScene):
        super().__init__(
            config=SpectrumEnvConfig(scene=scene.config, window_steps=8),
            seed=99,
        )
        self.scene = scene

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        render_seed = seed if seed is not None else 99
        self._scene = self.scene
        self._spectrogram, self._occupancy = self.scene.render(seed=render_seed)
        self._step_index = 0
        return self._observation(), {"events": self.scene.events}


def main() -> None:
    scene = RFScene.smart_scan_demo(SceneConfig(n_steps=72, n_channels=40, max_signals=4))
    train_env = RFSpectrumEnv(config=SpectrumEnvConfig(scene=scene.config, window_steps=8), seed=21)
    _, rewards = train_q_agent(train_env, QLearningConfig(episodes=250, seed=21))
    classifier = SpectrumChunkClassifier()
    _, info = FixedSceneEnv(scene).reset(seed=1234)
    smart_metrics, smart_trace = run_closed_loop(
        FixedSceneEnv(scene),
        ClosedLoopSmartScanAgent(n_actions=scene.config.n_channels),
        classifier,
        seed=1234,
    )
    energy_metrics, _ = run_closed_loop(
        FixedSceneEnv(scene),
        EnergyScanAgent(n_actions=scene.config.n_channels),
        classifier,
        seed=1234,
    )
    round_robin_metrics, round_robin_trace = run_closed_loop(
        FixedSceneEnv(scene),
        RoundRobinAgent(n_actions=scene.config.n_channels),
        classifier,
        seed=1234,
    )
    print("Smart Scan Technology - RF Sensing Simulation Layer")
    print("Mission: detect and characterize RF activity; encrypted content is not decoded.")
    print("")
    print("Scenario emitters:")
    for event in info["events"]:
        encrypted_note = " detect-only" if event.encrypted else ""
        print(
            f"- {event.label}: kind={event.kind.value} "
            f"channel={event.center_channel} duration={event.duration_steps}{encrypted_note}"
        )
    print("")
    print(f"training_reward_first_25={np.mean(rewards[:25]):.2f}")
    print(f"training_reward_last_25={np.mean(rewards[-25:]):.2f}")
    print("")
    print("Layer comparison against simulator ground truth:")
    print(_format_metrics("closed_loop_smart_scan", smart_metrics))
    print(_format_metrics("energy_scan", energy_metrics))
    print(_format_metrics("round_robin", round_robin_metrics))
    print("")
    print("Sample closed-loop trace:")
    for item in smart_trace[:12]:
        truth = ",".join(event["kind"] for event in item.ground_truth) or "none"
        encrypted = " detect-only" if item.classification.encrypted_detect_only else ""
        print(
            f"step={item.step:02d} channel={item.selected_channel:02d} "
            f"classifier={item.classification.signal_type} "
            f"confidence={item.classification.confidence:.2f}{encrypted} truth={truth}"
        )
    print("")
    print(f"round_robin_sample_channels={[item.selected_channel for item in round_robin_trace[:20]]}")


def _format_metrics(name: str, metrics: dict[str, float]) -> str:
    return (
        f"{name} "
        f"detection_rate={metrics['detection_rate']:.3f} "
        f"occupied_recall={metrics['occupied_recall']:.3f} "
        f"false_alarm_rate={metrics['false_alarm_rate']:.3f} "
        f"scan_efficiency={metrics['scan_efficiency']:.3f} "
        f"reward={metrics['mean_reward']:.2f}"
    )


if __name__ == "__main__":
    main()

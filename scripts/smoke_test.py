from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rf_spectrum_rl import RFScene, RFSpectrumEnv, SpectrumEnvConfig
from rf_spectrum_rl.agents import EnergyScanAgent, QLearningConfig, evaluate_agent, train_q_agent
from rf_spectrum_rl.classifier import SpectrumChunkClassifier
from rf_spectrum_rl.pipeline import ClosedLoopSmartScanAgent, RoundRobinAgent, run_closed_loop
from rf_spectrum_rl.scene import SceneConfig


def test_scene() -> None:
    config = SceneConfig(n_steps=16, n_channels=12, max_signals=3)
    scene = RFScene.random(config, seed=123)
    spectrogram, occupancy = scene.render(seed=456)
    assert spectrogram.shape == (16, 12)
    assert occupancy.shape == (16, 12)
    assert len(scene.events) >= 2
    assert occupancy.max() == 1


def test_env() -> None:
    config = SpectrumEnvConfig(scene=SceneConfig(n_steps=5, n_channels=8), window_steps=3)
    env = RFSpectrumEnv(config=config, seed=1)
    observation, info = env.reset(seed=2)
    assert observation.shape == (3, 8)
    assert info["events"]

    done = False
    steps = 0
    while not done:
        observation, reward, terminated, truncated, step_info = env.step(0)
        assert observation.shape == (3, 8)
        assert isinstance(reward, float)
        assert step_info["selected_channel"] == 0
        done = terminated or truncated
        steps += 1

    assert steps == 5


def test_agent() -> None:
    config = SpectrumEnvConfig(scene=SceneConfig(n_steps=8, n_channels=6), window_steps=3)
    env = RFSpectrumEnv(config=config, seed=11)
    agent, rewards = train_q_agent(env, QLearningConfig(episodes=3, seed=12))
    metrics = evaluate_agent(env, agent, episodes=2, seed=13)
    assert len(rewards) == 3
    assert 0.0 <= metrics["detection_rate"] <= 1.0

    smart_metrics = evaluate_agent(env, EnergyScanAgent(n_actions=6), episodes=2, seed=14)
    assert smart_metrics["scan_efficiency"] >= 0.0


def test_smart_scan_preset() -> None:
    scene = RFScene.smart_scan_demo()
    spectrogram, occupancy = scene.render(seed=99)
    assert spectrogram.shape == (scene.config.n_steps, scene.config.n_channels)
    assert occupancy.sum() > 0
    assert any(event.encrypted for event in scene.events)


def test_pipeline() -> None:
    scene = RFScene.smart_scan_demo(SceneConfig(n_steps=10, n_channels=12, max_signals=4))
    env = RFSpectrumEnv(config=SpectrumEnvConfig(scene=scene.config, window_steps=3), seed=20)
    metrics, trace = run_closed_loop(
        env,
        ClosedLoopSmartScanAgent(n_actions=12),
        SpectrumChunkClassifier(),
        seed=21,
    )
    assert len(trace) == 10
    assert metrics["steps"] == 10.0

    baseline_metrics, _ = run_closed_loop(
        RFSpectrumEnv(config=SpectrumEnvConfig(scene=scene.config, window_steps=3), seed=22),
        RoundRobinAgent(n_actions=12),
        SpectrumChunkClassifier(),
        seed=23,
    )
    assert baseline_metrics["steps"] == 10.0


def main() -> None:
    test_scene()
    test_env()
    test_agent()
    test_smart_scan_preset()
    test_pipeline()
    print("smoke tests passed")


if __name__ == "__main__":
    main()

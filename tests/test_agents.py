from rf_spectrum_rl import RFSpectrumEnv, SpectrumEnvConfig
from rf_spectrum_rl.agents import EnergyScanAgent, QLearningConfig, evaluate_agent, train_q_agent
from rf_spectrum_rl.scene import SceneConfig


def test_q_agent_trains_and_evaluates():
    env = RFSpectrumEnv(
        config=SpectrumEnvConfig(scene=SceneConfig(n_steps=8, n_channels=6), window_steps=3),
        seed=1,
    )
    agent, rewards = train_q_agent(env, QLearningConfig(episodes=3, seed=2))
    metrics = evaluate_agent(env, agent, episodes=2, seed=100)

    assert len(rewards) == 3
    assert 0.0 <= metrics["detection_rate"] <= 1.0
    assert 0.0 <= metrics["false_alarm_rate"] <= 1.0
    assert 0.0 <= metrics["scan_efficiency"] <= 1.0
    assert metrics["episodes"] == 2.0


def test_energy_scan_agent_evaluates():
    env = RFSpectrumEnv(
        config=SpectrumEnvConfig(scene=SceneConfig(n_steps=8, n_channels=6), window_steps=3),
        seed=3,
    )
    metrics = evaluate_agent(env, EnergyScanAgent(n_actions=6), episodes=2, seed=4)

    assert 0.0 <= metrics["detection_rate"] <= 1.0
    assert metrics["episodes"] == 2.0

from rf_spectrum_rl import RFSpectrumEnv, SpectrumEnvConfig
from rf_spectrum_rl.scene import SceneConfig


def test_env_reset_and_step_contract():
    config = SpectrumEnvConfig(scene=SceneConfig(n_steps=10, n_channels=8), window_steps=4)
    env = RFSpectrumEnv(config=config, seed=1)

    observation, info = env.reset(seed=2)
    assert observation.shape == (4, 8)
    assert "events" in info

    next_observation, reward, terminated, truncated, step_info = env.step(0)
    assert next_observation.shape == (4, 8)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert step_info["selected_channel"] == 0


def test_env_finishes_episode():
    config = SpectrumEnvConfig(scene=SceneConfig(n_steps=3, n_channels=6), window_steps=2)
    env = RFSpectrumEnv(config=config, seed=3)
    env.reset(seed=4)

    done = False
    steps = 0
    while not done:
        _, _, terminated, truncated, _ = env.step(0)
        done = terminated or truncated
        steps += 1

    assert steps == 3

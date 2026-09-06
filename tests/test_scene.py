import numpy as np

from rf_spectrum_rl.scene import RFScene, SceneConfig


def test_random_scene_renders_expected_shapes():
    config = SceneConfig(n_steps=16, n_channels=12, max_signals=3)
    scene = RFScene.random(config, seed=123)

    spectrogram, occupancy = scene.render(seed=456)

    assert spectrogram.shape == (16, 12)
    assert occupancy.shape == (16, 12)
    assert spectrogram.dtype == np.float32
    assert occupancy.max() == 1
    assert len(scene.events) >= 2


def test_smart_scan_demo_includes_detect_only_encrypted_signal():
    scene = RFScene.smart_scan_demo()

    assert any(event.encrypted for event in scene.events)
    assert any(event.label == "frequency_hopping_link" for event in scene.events)

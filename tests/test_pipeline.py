from rf_spectrum_rl import RFScene, RFSpectrumEnv, SpectrumEnvConfig
from rf_spectrum_rl.classifier import SpectrumChunkClassifier
from rf_spectrum_rl.pipeline import ClosedLoopSmartScanAgent, RoundRobinAgent, run_closed_loop
from rf_spectrum_rl.scene import SceneConfig


class FixedSceneEnv(RFSpectrumEnv):
    def __init__(self, scene: RFScene):
        super().__init__(SpectrumEnvConfig(scene=scene.config, window_steps=4), seed=1)
        self.scene = scene

    def reset(self, *, seed=None, options=None):
        del options
        self._scene = self.scene
        self._spectrogram, self._occupancy = self.scene.render(seed=seed)
        self._step_index = 0
        return self._observation(), {"events": self.scene.events}


def test_environment_step_exposes_layer_three_inputs():
    scene = RFScene.smart_scan_demo(SceneConfig(n_steps=12, n_channels=16, max_signals=4))
    env = FixedSceneEnv(scene)
    env.reset(seed=5)

    _, _, _, _, info = env.step(8)

    assert "observed_chunk" in info
    assert "active_events" in info
    assert "selected_power" in info


def test_closed_loop_pipeline_runs_against_round_robin_baseline():
    scene = RFScene.smart_scan_demo(SceneConfig(n_steps=24, n_channels=20, max_signals=4))
    classifier = SpectrumChunkClassifier()

    smart_metrics, smart_trace = run_closed_loop(
        FixedSceneEnv(scene),
        ClosedLoopSmartScanAgent(n_actions=20),
        classifier,
        seed=8,
    )
    baseline_metrics, baseline_trace = run_closed_loop(
        FixedSceneEnv(scene),
        RoundRobinAgent(n_actions=20),
        classifier,
        seed=8,
    )

    assert len(smart_trace) == 24
    assert len(baseline_trace) == 24
    assert 0.0 <= smart_metrics["scan_efficiency"] <= 1.0
    assert 0.0 <= baseline_metrics["scan_efficiency"] <= 1.0

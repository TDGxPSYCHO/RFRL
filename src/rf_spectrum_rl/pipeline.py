from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .classifier import ClassificationResult, SpectrumChunkClassifier
from .env import RFSpectrumEnv
from .metrics import ScanMetricTracker


@dataclass(frozen=True)
class PipelineStep:
    step: int
    selected_channel: int
    reward: float
    classification: ClassificationResult
    ground_truth: list[dict]


class RoundRobinAgent:
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.next_channel = 0

    def act(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        del observation, epsilon
        action = self.next_channel
        self.next_channel = (self.next_channel + 1) % self.n_actions
        return action

    def observe_feedback(self, selected_channel: int, classification: ClassificationResult, reward: float) -> None:
        del selected_channel, classification, reward


class ClosedLoopSmartScanAgent:
    """Layer 1 scan policy with classifier feedback.

    It starts from energy scanning, then revisits channels where Layer 3
    reported useful detections.
    """

    def __init__(self, n_actions: int, revisit_bonus: float = 0.35):
        self.n_actions = n_actions
        self.revisit_bonus = revisit_bonus
        self.channel_scores = np.zeros(n_actions, dtype=np.float32)

    def act(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        del epsilon
        latest = observation[-1]
        score = latest + self.channel_scores
        return int(np.argmax(score))

    def observe_feedback(self, selected_channel: int, classification: ClassificationResult, reward: float) -> None:
        self.channel_scores *= 0.96
        if classification.detected and reward > 0:
            self.channel_scores[selected_channel] += self.revisit_bonus * classification.confidence
        elif reward < 0:
            self.channel_scores[selected_channel] *= 0.5


def run_closed_loop(
    env: RFSpectrumEnv,
    agent: object,
    classifier: SpectrumChunkClassifier,
    seed: int,
) -> tuple[dict[str, float], list[PipelineStep]]:
    observation, _ = env.reset(seed=seed)
    tracker = ScanMetricTracker()
    trace: list[PipelineStep] = []
    done = False
    step = 0

    while not done:
        action = agent.act(observation, epsilon=0.0)  # type: ignore[attr-defined]
        next_observation, reward, terminated, truncated, info = env.step(action)
        classification = classifier.classify(
            info["observed_chunk"],
            selected_power=float(info["selected_power"]),
            active_events=info["active_events"],
        )
        if hasattr(agent, "observe_feedback"):
            agent.observe_feedback(action, classification, reward)  # type: ignore[attr-defined]

        tracker.update(
            reward,
            selected_occupied=bool(info["selected_occupied"]),
            occupied_channels=info["occupied_channels"],
        )
        trace.append(
            PipelineStep(
                step=step,
                selected_channel=int(action),
                reward=float(reward),
                classification=classification,
                ground_truth=info["active_events"],
            )
        )
        observation = next_observation
        done = terminated or truncated
        step += 1

    return tracker.result().as_dict(), trace

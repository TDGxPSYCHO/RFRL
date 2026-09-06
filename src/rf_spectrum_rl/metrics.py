from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScanMetrics:
    mean_reward: float
    detection_rate: float
    occupied_recall: float
    false_alarm_rate: float
    scan_efficiency: float
    steps: int

    def as_dict(self) -> dict[str, float]:
        return {
            "mean_reward": self.mean_reward,
            "detection_rate": self.detection_rate,
            "occupied_recall": self.occupied_recall,
            "false_alarm_rate": self.false_alarm_rate,
            "scan_efficiency": self.scan_efficiency,
            "steps": float(self.steps),
        }


class ScanMetricTracker:
    def __init__(self) -> None:
        self.reward = 0.0
        self.steps = 0
        self.selected_hits = 0
        self.selected_false = 0
        self.occupied_steps = 0
        self.missed_occupied_steps = 0

    def update(self, reward: float, selected_occupied: bool, occupied_channels: list[int]) -> None:
        self.reward += reward
        self.steps += 1
        occupied_now = bool(occupied_channels)
        self.occupied_steps += int(occupied_now)
        self.selected_hits += int(selected_occupied)
        self.selected_false += int((not selected_occupied) and occupied_now)
        self.missed_occupied_steps += int((not selected_occupied) and occupied_now)

    def result(self) -> ScanMetrics:
        return ScanMetrics(
            mean_reward=self.reward,
            detection_rate=self.selected_hits / max(1, self.steps),
            occupied_recall=(self.occupied_steps - self.missed_occupied_steps) / max(1, self.occupied_steps),
            false_alarm_rate=self.selected_false / max(1, self.steps),
            scan_efficiency=self.selected_hits / max(1, self.selected_hits + self.selected_false),
            steps=self.steps,
        )

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationResult:
    detected: bool
    signal_type: str
    confidence: float
    encrypted_detect_only: bool = False


class SpectrumChunkClassifier:
    """Layer 3 classifier over the chunk selected by Layer 1.

    This is an explainable baseline. A TorchSig-trained neural classifier can
    replace this class as long as it returns ClassificationResult.
    """

    def __init__(self, detection_threshold: float = 0.18):
        self.detection_threshold = detection_threshold

    def classify(self, observed_chunk: np.ndarray, selected_power: float, active_events: list[dict]) -> ClassificationResult:
        peak = float(np.max(observed_chunk)) if observed_chunk.size else selected_power
        confidence = min(1.0, max(0.0, peak))
        detected = peak >= self.detection_threshold

        if not detected:
            return ClassificationResult(False, "noise", confidence)

        if active_events:
            event = active_events[0]
            signal_type = str(event["kind"])
            encrypted = bool(event["encrypted"])
            return ClassificationResult(True, signal_type, confidence, encrypted_detect_only=encrypted)

        return ClassificationResult(True, "anomaly", confidence)

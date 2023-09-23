from dataclasses import dataclass


@dataclass
class DetectionResult():
    result: str
    confidence: float

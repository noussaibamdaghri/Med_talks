from typing import Literal


ConfidenceLevel = Literal["low", "medium", "high"]


def compute_confidence(score: float) -> ConfidenceLevel:
    """
    Converts a classifier score into a confidence level.
    """
    if score >= 0.75:
        return "high"
    elif score >= 0.4:
        return "medium"
    else:
        return "low"

from pydantic import BaseModel
from typing import Literal


class ActionPlan(BaseModel):
    """
    A structured decision created by Person B.
    Consumed by Person D to decide what to do next.
    """

    # What domain are we in
    domain: Literal["medical", "non-medical", "invalid", "unknown"]

    # Explicit intent for downstream logic (VERY IMPORTANT)
    intent: str
    # Examples:
    # - "medical_definition"
    # - "medical_reasoning"
    # - "uncertain_medical_question"
    # - "general_question"
    # - "dangerous_or_sensitive"

    # Confidence score coming from the classifier
    confidence: float

    # Risk assessment
    risk_level: Literal["low", "medium", "high"]

    # Whether retrieval (Person A / C) is required
    needs_external_data: bool

    # How the LLM is allowed to behave
    llm_mode: Literal["direct", "refusal"]

from pydantic import BaseModel
from typing import Literal

class ActionPlan(BaseModel):
    """
    A structured plan created by Person B that defines the next steps.
    """
    domain: Literal["medical", "non-medical"]
    risk_level: Literal["low", "medium", "high"]
    needs_external_data: bool
    llm_mode: Literal["normal", "cautious", "refusal"]

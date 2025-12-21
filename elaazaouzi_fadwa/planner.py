try:
    # First try relative import (when run as package)
    from .plans import ActionPlan
    from .confidence import compute_confidence
except ImportError:
    # Fall back to absolute import (when run directly)
    from plans import ActionPlan
    from confidence import compute_confidence


class PersonBPlanner:
    """Creates ActionPlans based on classifier results"""
    
    def create_plan(self, classifier_result: dict, confidence_level: str) -> ActionPlan:
        """
        Create an ActionPlan based on classification and confidence.
        
        Args:
            classifier_result: Dictionary with 'label' and 'score'
            confidence_level: 'high', 'medium', or 'low'
            
        Returns:
            ActionPlan object
        """
        label = classifier_result["label"].lower()
        
        # Determine domain
        if "non-medical" in label:
            domain = "non-medical"
        else:
            domain = "medical"
        
        # Determine risk level
        if "emergency" in label:
            risk_level = "high"
        elif "general medical" in label:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Determine if external data is needed
        # Medical questions need data, non-medical don't
        needs_external_data = domain == "medical"
        
        # Determine LLM mode
        if confidence_level == "low":
            llm_mode = "refusal"
        elif risk_level == "high":
            llm_mode = "cautious"
        else:
            llm_mode = "normal"
        
        return ActionPlan(
            domain=domain,
            risk_level=risk_level,
            needs_external_data=needs_external_data,
            llm_mode=llm_mode
        )
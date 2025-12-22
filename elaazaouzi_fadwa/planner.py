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
        """
        label = classifier_result["label"].lower()
        
        # Determine domain
        if "non-medical" in label:
            domain = "non-medical"
            risk_level = "low"
            needs_external_data = False
        else:
            domain = "medical"
            
            # Determine risk level based on medical category
            if "reasoning" in label or "multiple choice" in label:
                # These could involve diagnosis - higher risk
                risk_level = "medium"
            elif "stepwise" in label:
                # Procedures might have risks
                risk_level = "medium"
            else:  # definition questions
                risk_level = "low"
                
            needs_external_data = True
        
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
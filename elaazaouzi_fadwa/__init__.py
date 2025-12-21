"""
Person B - Triage Nurse Module
Medical query classification and action planning.
"""

# Import and expose the main components
try:
    # Try relative imports first (when run as package)
    from .classifier import ZeroShotClassifier
    from .confidence import compute_confidence
    from .plans import ActionPlan
    from .planner import PersonBPlanner
    from .validators import validate_input
except ImportError:
    # Fall back to absolute imports (when run directly from folder)
    from classifier import ZeroShotClassifier
    from confidence import compute_confidence
    from plans import ActionPlan
    from planner import PersonBPlanner
    from validators import validate_input
# Define what gets imported with "from elaazaouzi_fadwa import *"
__all__ = [
    "ZeroShotClassifier",
    "compute_confidence",
    "ActionPlan", 
    "PersonBPlanner",
    "validate_input"
]

# Package metadata
__version__ = "1.0.0"
__author__ = "Fadwa El Aazaouzi"
__description__ = "Medical triage nurse: classifies queries and creates action plans"

# Optional: Add a main function for direct execution
def triage_pipeline(query: str):
    """
    Complete Person B triage pipeline for a single query.
    
    Args:
        query: User's medical question
        
    Returns:
        tuple: (classification, confidence, action_plan)
    """
    # Validate input
    validation = validate_input(query)
    if not validation["valid"]:
        raise ValueError(f"Invalid input: {validation['reason']}")
    
    # Classify
    classifier = ZeroShotClassifier()
    classification = classifier.classify(query)
    
    # Compute confidence
    confidence = compute_confidence(classification["score"])
    
    # Create action plan
    planner = PersonBPlanner()
    action_plan = planner.create_plan(classification, confidence)
    
    return classification, confidence, action_plan


def main():
    """Command-line interface for Person B module."""
    import sys
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        try:
            classification, confidence, action_plan = triage_pipeline(query)
            
            print("=" * 60)
            print("PERSON B - TRIAGE NURSE REPORT")
            print("=" * 60)
            print(f"Query: {query}")
            print(f"\nClassification: {classification['label']} (score: {classification['score']:.3f})")
            print(f"Confidence: {confidence}")
            print(f"\nAction Plan:")
            print(f"  • Domain: {action_plan.domain}")
            print(f"  • Risk Level: {action_plan.risk_level}")
            print(f"  • Needs External Data: {action_plan.needs_external_data}")
            print(f"  • LLM Mode: {action_plan.llm_mode}")
            print("=" * 60)
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python -m elaazaouzi_fadwa 'your medical query here'")
        print("Example: python -m elaazaouzi_fadwa 'I have chest pain'")


if __name__ == "__main__":
    main()
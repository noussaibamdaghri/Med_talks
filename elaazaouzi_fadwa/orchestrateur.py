from classifier import ZeroShotClassifier
from confidence import compute_confidence
from planner import PersonBPlanner
from validators import validate_input
from plans import ActionPlan


class PersonBOrchestrator:
    """
    Infirmière de triage – décide ce que le système a le droit de faire
    """

    def __init__(self):
        self.classifier = ZeroShotClassifier()
        self.planner = PersonBPlanner()

    def classify_question(self, question: str) -> ActionPlan:
        # 1. Validation
        validation = validate_input(question)

        if not validation["valid"]:
            return ActionPlan(
                domain="invalid",
                intent="invalid_input",
                confidence=0.0,
                risk_level="low",
                needs_external_data=False,
                llm_mode="refusal"
            )

        if validation.get("risk_level") == "high":
            return ActionPlan(
                domain="medical",
                intent="dangerous_or_sensitive",
                confidence=1.0,
                risk_level="high",
                needs_external_data=False,
                llm_mode="refusal"
            )

        # 2. Classification (raw text is OK)
        classification = self.classifier.classify(question)

        # 3. Confidence
        confidence = compute_confidence(classification["score"])

        # 4. Decision
        return self.planner.create_plan(classification, confidence)

"""
Orchestrateur Person B
Point d'entrée unique pour Person D
"""

from .preprocessor import TextPreprocessor
from .classifier import ZeroShotClassifier
from .confidence import compute_confidence
from .planner import PersonBPlanner
from .validators import validate_input
from .plans import ActionPlan


class PersonBOrchestrator:
    """
    Infirmière de triage – décide ce que le système a le droit de faire
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.classifier = ZeroShotClassifier()
        self.planner = PersonBPlanner()

    def classify_question(self, question: str) -> ActionPlan:
        """
        Point d'entrée principal de Person B (appelé par Person D)
        """

        # 1. Validation (sécurité AVANT TOUT)
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

        # 2. Prétraitement (forme demandée par Person D)
        processed = self.preprocessor.process(question)

        # 3. Classification
        classification = self.classifier.classify(processed)

        # 4. Confiance
        confidence = compute_confidence(classification["score"])

        # 5. Planification (décision finale)
        action_plan = self.planner.create_plan(classification, confidence)

        return action_plan

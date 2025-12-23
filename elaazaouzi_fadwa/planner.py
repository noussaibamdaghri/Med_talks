from .plans import ActionPlan


class PersonBPlanner:
    """
    Decision logic for Person B.
    Turns classification + confidence into an ActionPlan.
    """

    def create_plan(self, classification: dict, confidence: str) -> ActionPlan:
        """
        Args:
            classification: output of ZeroShotClassifier
            confidence: "high" | "medium" | "low"

        Returns:
            ActionPlan
        """

        label = classification["label"]
        score = classification["score"]

        # 1. Non-medical questions → allow normal LLM response
        if label == "non-medical question":
            return ActionPlan(
                domain="non-medical",
                intent="general_question",
                confidence=score,
                risk_level="low",
                needs_external_data=False,
                llm_mode="direct"
            )

        # 2. Low confidence medical → be conservative
        if confidence == "low":
            return ActionPlan(
                domain="medical",
                intent="uncertain_medical_question",
                confidence=score,
                risk_level="medium",
                needs_external_data=True,
                llm_mode="refusal"
            )

        # 3. Definition questions with good confidence
        if label == "medical definition question" and confidence in ["medium", "high"]:
            return ActionPlan(
                domain="medical",
                intent="medical_definition",
                confidence=score,
                risk_level="low",
                needs_external_data=True,
                llm_mode="direct"
            )

        # 4. Reasoning / stepwise / MCQ → always cautious
        if label in [
            "medical reasoning question",
            "medical stepwise procedure question",
            "medical multiple choice question"
        ]:
            return ActionPlan(
                domain="medical",
                intent="medical_reasoning",
                confidence=score,
                risk_level="medium",
                needs_external_data=True,
                llm_mode="refusal"
            )

        # 5. Fallback (should not happen, but safe)
        return ActionPlan(
            domain="unknown",
            intent="unclassified",
            confidence=score,
            risk_level="medium",
            needs_external_data=True,
            llm_mode="refusal"
        )

from transformers import pipeline

class ZeroShotClassifier:
    def __init__(self):
        # Initialize the model once to be efficient
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

    def classify(self, text: str) -> dict:
        """
        Classifies medical queries into appropriate reasoning types.
        
        Categories:
        1. Definition - Asking for medical term definitions
        2. Reasoning - Requiring medical reasoning/diagnosis
        3. Stepwise - Step-by-step procedure or treatment questions
        4. Multiple Choice - Differential diagnosis or option-based questions
        5. Non-medical - General questions not related to medicine
        """
        labels = [
            "medical definition question",
            "medical reasoning question", 
            "medical stepwise procedure question",
            "medical multiple choice question",
            "non-medical question"
        ]
        
        result = self.classifier(text, labels, multi_label=False)
        
        # Return a clean dictionary
        return {
            "label": result["labels"][0],
            "score": result["scores"][0],
            "all_scores": dict(zip(result["labels"], result["scores"]))
        }

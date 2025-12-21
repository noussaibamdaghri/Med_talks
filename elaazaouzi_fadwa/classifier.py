from transformers import pipeline

class ZeroShotClassifier:
    def __init__(self):
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

    def classify(self, text: str) -> dict:
        labels = [
            "medical emergency",
            "general medical question",
            "non-medical"
        ]
        result = self.classifier(text, labels, multi_label=False)

        return {
            "label": result["labels"][0],
            "score": result["scores"][0],
        }

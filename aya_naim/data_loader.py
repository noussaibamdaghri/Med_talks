import pandas as pd
from datasets import load_dataset

class MedicalDataLoader:
    """AYA NAIM: Load medical flashcards"""
    
    def __init__(self):
        self.df = None
        print("AYA: MedicalDataLoader ready")
    
    def load(self):
        """Load the dataset"""
        try:
            dataset = load_dataset(
                "medalpaca/medical_meadow_medical_flashcards",
                "medical_meadow_wikidoc_medical_flashcards"
            )
            self.df = dataset['train'].to_pandas()
        except:
            # Fallback method
            self.df = pd.read_json(
                "hf://datasets/medalpaca/medical_meadow_medical_flashcards/medical_meadow_wikidoc_medical_flashcards.json"
            )
        
        print(f"AYA: Loaded {len(self.df)} medical flashcards")
        return self.df
    
    def info(self):
        """Get dataset info"""
        if self.df is None:
            return {"error": "No data loaded"}
        return {
            "records": len(self.df),
            "columns": list(self.df.columns),
            "sample": self.df.iloc[0].to_dict()
        }
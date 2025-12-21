import pandas as pd
import sys

print("=== TESTING AYA'S DATASET LOADING ===")

try:
    # Try loading with datasets library first
    from datasets import load_dataset
    dataset = load_dataset(
        "medalpaca/medical_meadow_medical_flashcards", 
        "medical_meadow_wikidoc_medical_flashcards"
    )
    df = dataset['train'].to_pandas()
    
    print(f"1. Dataset loaded successfully via datasets library!")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    print(f"\n2. First 3 rows:")
    print(df.head(3))
    
    print(f"\n3. Basic info:")
    print(f"   Total records: {len(df)}")
    
    # Check common column names
    for col in ['question', 'answer', 'input', 'output', 'text']:
        if col in df.columns:
            sample = str(df[col].iloc[0])[:80]
            print(f"   Sample {col}: {sample}...")
            break
    
except Exception as e:
    print(f"ERROR: {e}")
    print("\nTrying alternative method...")
    
    try:
        # Alternative: direct pandas read
        df = pd.read_json(
            "hf://datasets/medalpaca/medical_meadow_medical_flashcards/medical_meadow_wikidoc_medical_flashcards.json"
        )
        print(f"\n✅ Loaded via pandas!")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
    except Exception as e2:
        print(f"Also failed: {e2}")
        print("\nInstall datasets library: pip install datasets")
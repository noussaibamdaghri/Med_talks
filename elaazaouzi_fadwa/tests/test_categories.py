import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from elaazaouzi_fadwa.classifier import ZeroShotClassifier

def test_medical_categories():
    """Test the new medical category classification"""
    
    classifier = ZeroShotClassifier()
    
    test_cases = [
        ("What is hypertension?", "definition"),
        ("Why does diabetes cause frequent urination?", "reasoning"),
        ("How to administer insulin injection step by step?", "stepwise"),
        ("Is this migraine, tension headache, or cluster headache?", "multiple choice"),
        ("What time does the pharmacy open?", "non-medical")
    ]
    
    print("=" * 60)
    print("TESTING MEDICAL CATEGORY CLASSIFICATION")
    print("=" * 60)
    
    for query, expected in test_cases:
        result = classifier.classify(query)
        print(f"\nQuery: '{query}'")
        print(f"Top label: {result['label']}")
        print(f"Score: {result['score']:.3f}")
        
        # Show all scores
        print("All categories:")
        for label, score in result['all_scores'].items():
            print(f"  - {label}: {score:.3f}")
        
        # Check if correct category is detected
        if expected == "non-medical":
            expected_type = "non-medical"
        else:
            expected_type = "medical"
    
        if expected_type in result["label"].lower():
            print(f"  ✓ Correctly identified as {expected_type}")
        else:
            print(f"  ⚠️  Identified as {result['label']}, expected {expected_type}")
    
    print("\n" + "=" * 60)
    print("CATEGORY TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_medical_categories()
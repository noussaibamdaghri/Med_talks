import sys
import os

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath('.'))

from elaazaouzi_fadwa.classifier import ZeroShotClassifier
from elaazaouzi_fadwa.confidence import compute_confidence
from elaazaouzi_fadwa.plans import ActionPlan
from elaazaouzi_fadwa.planner import PersonBPlanner

def test_classifier():
    """Test the zero-shot classifier"""
    print("🧪 Testing Classifier...")
    
    classifier = ZeroShotClassifier()
    
    # Test cases with expected medical/non-medical
    test_queries = [
        ("What is myocardial infarction?", "medical"),
        ("Why does chest pain radiate to the left arm?", "medical"),
        ("How do I perform CPR step by step?", "medical"),
        ("Is it bronchitis, pneumonia, or COVID-19?", "medical"),
        ("What's the weather today?", "non-medical")
    ]
    
    for query, expected_type in test_queries:
        result = classifier.classify(query)
        print(f"  Query: '{query[:30]}...'")
        print(f"  Result: {result['label']} (score: {result['score']:.3f})")
        
        # Check if medical/non-medical is correctly identified
        if expected_type == "medical":
            assert "medical" in result["label"].lower()
            print(f"  ✓ Correctly identified as medical")
        else:
            assert "non-medical" in result["label"].lower()
            print(f"  ✓ Correctly identified as non-medical")
        print()

def test_confidence():
    """Test confidence calculation"""
    print("📊 Testing Confidence Calculator...")
    
    test_cases = [
        (0.85, "high"),
        (0.75, "high"),
        (0.65, "medium"),
        (0.45, "medium"),
        (0.35, "low"),
        (0.10, "low")
    ]
    
    for score, expected in test_cases:
        result = compute_confidence(score)
        print(f"  Score: {score:.2f} → Confidence: {result}")
        assert result == expected
        print(f"  ✓ Correct")
    print()

def test_planner():
    """Test that ActionPlans are created correctly"""
    print("📋 Testing Planner...")
    
    planner = PersonBPlanner()
    
    # Test medical reasoning case (medium risk)
    reasoning_result = {"label": "medical reasoning question", "score": 0.8}
    reasoning_plan = planner.create_plan(reasoning_result, "high")
    
    assert isinstance(reasoning_plan, ActionPlan)
    assert reasoning_plan.domain == "medical"
    assert reasoning_plan.risk_level == "medium"  # reasoning = medium risk
    assert reasoning_plan.needs_external_data == True
    assert reasoning_plan.llm_mode == "normal"
    print("  ✓ Medical reasoning plan created correctly")
    
    # Test non-medical case
    nonmedical_result = {"label": "non-medical question", "score": 0.85}
    nonmedical_plan = planner.create_plan(nonmedical_result, "high")
    
    assert nonmedical_plan.domain == "non-medical"
    assert nonmedical_plan.risk_level == "low"
    assert nonmedical_plan.needs_external_data == False
    assert nonmedical_plan.llm_mode == "normal"
    print("  ✓ Non-medical plan created correctly")
    
    # Test low confidence case
    low_confidence_result = {"label": "medical reasoning question", "score": 0.3}
    low_confidence_plan = planner.create_plan(low_confidence_result, "low")
    
    assert low_confidence_plan.llm_mode == "refusal"
    print("  ✓ Low confidence triggers refusal mode")
    
    print()

def run_all_tests():
    """Run all test functions"""
    print("=" * 60)
    print("PERSON B - SIMPLE TEST")
    print("=" * 60)
    
    try:
        test_classifier()
        test_confidence()
        test_planner()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return True
    except AssertionError as e:
        print("=" * 60)
        print(f"❌ ASSERTION ERROR: {e}")
        print("=" * 60)
        return False
    except Exception as e:
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
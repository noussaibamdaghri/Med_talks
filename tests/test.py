import sys
import os

# FIXED: Use '..' not '...'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from elaazaouzi_fadwa.classifier import ZeroShotClassifier
from elaazaouzi_fadwa.confidence import compute_confidence
from elaazaouzi_fadwa.plans import ActionPlan
from elaazaouzi_fadwa.planner import PersonBPlanner
from elaazaouzi_fadwa.validators import validate_input

def test_classifier():
    """Test the zero-shot classifier"""
    print("🧪 Testing Classifier...")
    
    classifier = ZeroShotClassifier()
    
    # Test cases
    test_queries = [
        ("I have chest pain and difficulty breathing", "medical emergency"),
        ("What are symptoms of flu?", "general medical question"),
        ("What's the weather today?", "non-medical")
    ]
    
    for query, expected_label in test_queries:
        result = classifier.classify(query)
        print(f"  Query: '{query[:30]}...'")
        print(f"  Result: {result['label']} (score: {result['score']:.3f})")
        
        # Check if medical/non-medical is correctly identified
        if "medical" in expected_label:
            assert "medical" in result["label"].lower()
            print("  ✓ Correctly identified as medical")
        else:
            assert "non-medical" in result["label"].lower()
            print("  ✓ Correctly identified as non-medical")
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
    """Test ActionPlan creation"""
    print("📋 Testing Planner...")
    
    planner = PersonBPlanner()
    
    # Test emergency case
    emergency_result = {"label": "medical emergency", "score": 0.9}
    plan = planner.create_plan(emergency_result, "high")
    
    assert plan.domain == "medical"
    assert plan.risk_level == "high"
    assert plan.needs_external_data == True
    assert plan.llm_mode == "cautious"
    print("  ✓ Emergency plan created correctly")
    
    print()

def run_simple_tests():
    """Run basic tests"""
    print("=" * 60)
    print("PERSON B - SIMPLE TEST")
    print("=" * 60)
    
    try:
        test_classifier()
        test_confidence()
        test_planner()
        
        print("=" * 60)
        print("✅ BASIC TESTS PASSED!")
        print("=" * 60)
        return True
    except Exception as e:
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = run_simple_tests()
    sys.exit(0 if success else 1)
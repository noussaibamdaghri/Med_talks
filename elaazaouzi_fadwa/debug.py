import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from elaazaouzi_fadwa.planner import PersonBPlanner
from elaazaouzi_fadwa.plans import ActionPlan

print("=" * 60)
print("DEBUGGING PLANNER FAILURE")
print("=" * 60)

# Create planner
planner = PersonBPlanner()

# Test the exact cases from your test
test_cases = [
    {"label": "medical reasoning question", "score": 0.8, "confidence": "high", "name": "Medical reasoning"},
    {"label": "non-medical question", "score": 0.85, "confidence": "high", "name": "Non-medical"},
    {"label": "medical reasoning question", "score": 0.3, "confidence": "low", "name": "Low confidence medical"}
]

for test in test_cases:
    print(f"\n📋 Testing: {test['name']}")
    print(f"  Input: label={test['label']}, confidence={test['confidence']}")
    
    try:
        plan = planner.create_plan(test, test['confidence'])
        print(f"  ✅ Plan created successfully")
        print(f"    Domain: {plan.domain}")
        print(f"    Risk Level: {plan.risk_level}")
        print(f"    Needs Data: {plan.needs_external_data}")
        print(f"    LLM Mode: {plan.llm_mode}")
        
        # Check it's an ActionPlan
        assert isinstance(plan, ActionPlan), f"Result is not an ActionPlan: {type(plan)}"
        
        # Check specific expectations
        if test['name'] == "Medical reasoning":
            assert plan.domain == "medical", f"Expected medical, got {plan.domain}"
            assert plan.risk_level == "medium", f"Expected medium risk, got {plan.risk_level}"
            assert plan.needs_external_data == True, "Medical should need external data"
            assert plan.llm_mode == "normal", f"Expected normal mode, got {plan.llm_mode}"
            print("  ✅ All assertions passed for medical reasoning")
            
        elif test['name'] == "Non-medical":
            assert plan.domain == "non-medical", f"Expected non-medical, got {plan.domain}"
            assert plan.risk_level == "low", f"Expected low risk, got {plan.risk_level}"
            assert plan.needs_external_data == False, "Non-medical should not need external data"
            assert plan.llm_mode == "normal", f"Expected normal mode, got {plan.llm_mode}"
            print("  ✅ All assertions passed for non-medical")
            
        elif test['name'] == "Low confidence medical":
            assert plan.llm_mode == "refusal", f"Low confidence should trigger refusal, got {plan.llm_mode}"
            print("  ✅ Low confidence triggers refusal")
            
    except AssertionError as e:
        print(f"  ❌ ASSERTION FAILED: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
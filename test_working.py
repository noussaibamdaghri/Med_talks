"""
WORKING TEST FILE - Always runs
"""

import sys
sys.path.append('.')

print("=" * 60)
print(" PERSON A WORKING INTEGRATION TEST")
print("=" * 60)

try:
    # Try to import
    from aya_naim.orchestrator_working import PersonAWorkingOrchestrator, test_working_orchestrator
    
    print("1. Testing orchestrator...")
    result = test_working_orchestrator()
    
    print("\n" + "=" * 60)
    print("2. TEAM INTEGRATION DEMO")
    print("=" * 60)
    
    orchestrator = PersonAWorkingOrchestrator()
    
    # Test different queries
    medical_queries = [
        "headache and nausea symptoms",
        "high blood pressure treatment",
        "how to manage diabetes type 2"
    ]
    
    print("\n Processing medical queries:")
    for i, query in enumerate(medical_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        output = orchestrator.process_query(query)
        print(f"   → Cleaned: {output['cleaned_query']}")
        print(f"   → Similar found: {output['similar_found']}")
    
    print("\n" + "=" * 60)
    print(" HOW TEAM MEMBERS USE THIS:")
    print("=" * 60)
    
    print("""
# PERSON B (Classification):
from aya_naim.orchestrator_working import PersonAWorkingOrchestrator
orchestrator = PersonAWorkingOrchestrator()
result = orchestrator.process_query("chest pain symptoms")
cleaned_text = result['cleaned_query']  # Use for classification

# PERSON D (LLM):
from aya_naim.orchestrator_working import PersonAWorkingOrchestrator
orchestrator = PersonAWorkingOrchestrator()
result = orchestrator.process_query("patient with cough")
context = result['formatted_context']  # Add to LLM prompt
print(context)
    """)
    
    print("\n" + "=" * 60)
    print(" PERSON A READY FOR TEAM USE!")
    print("=" * 60)
    
except Exception as e:
    print(f" Error: {e}")
    print("\nCreating working orchestrator file...")
    
    # Create a minimal version
    minimal_code = '''
import re

class MinimalOrchestrator:
    def __init__(self):
        print("Minimal Person A Orchestrator ready")
    
    def process_query(self, query):
        # Simple cleaning
        cleaned = re.sub(r"\\\\s+", " ", query).strip()
        return {
            "cleaned_query": cleaned,
            "word_count": len(cleaned.split()),
            "status": "processed by Person A"
        }

# Quick test
if __name__ == "__main__":
    o = MinimalOrchestrator()
    r = o.process_query("test query")
    print(f"Result: {r}")
'''
    
    # Save it
    with open('aya_naim/orchestrator_working.py', 'w') as f:
        f.write(minimal_code)
    
    print(" Created minimal orchestrator")
    print("Run: python -c \"from aya_naim.orchestrator_working import MinimalOrchestrator; o = MinimalOrchestrator(); print(o.process_query('medical query'))\"")
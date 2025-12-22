"""
TEST FILE FOR AYA NAIM ORCHESTRATOR
Run with: python test_orchestrator.py
"""

import sys
sys.path.append('.')

print("=" * 60)
print(" AYA NAIM ORCHESTRATOR INTEGRATION TEST")
print("=" * 60)

try:
    # Import the orchestrator
    from aya_naim.orchestrator import PersonAOrchestrator, test_person_a_orchestrator
    
    print("1. Testing basic orchestrator function...")
    print("-" * 40)
    basic_result = test_person_a_orchestrator()
    
    print("\n" + "=" * 60)
    print("2. TESTING TEAM INTEGRATION SCENARIOS")
    print("=" * 60)
    
    # Create orchestrator instance
    orchestrator = PersonAOrchestrator()
    
    # Test queries that other team members might use
    test_cases = [
        {
            "query": "headache and fever symptoms",
            "user": "Person B (Classification)",
            "use_case": "Classify symptom severity"
        },
        {
            "query": "treatment for high blood pressure",
            "user": "Person C (APIs)", 
            "use_case": "Find PubMed references"
        },
        {
            "query": "What is diabetes and how to manage it?",
            "user": "Person D (LLM)",
            "use_case": "Get context for LLM prompt"
        }
    ]
    
    print("\n MEDICAL QUERY PROCESSING DEMO:")
    print("-" * 40)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['user']}: '{test['query']}'")
        print(f"   Use case: {test['use_case']}")
        
        try:
            # Process the query
            result = orchestrator.process_query(test['query'])
            
            print(f"    Processed successfully")
            print(f"   • Cleaned query: {result['cleaned_query'][:50]}...")
            print(f"   • Similar questions found: {len(result['similar_results'])}")
            
            # Show first similar result
            if result['similar_results']:
                first_match = result['similar_results'][0]
                print(f"   • Top match: '{first_match[0][:60]}...'")
            
        except Exception as e:
            print(f"    Error: {e}")
    
    print("\n" + "=" * 60)
    print(" HOW OTHER TEAM MEMBERS USE PERSON A:")
    print("=" * 60)
    
    print("""
# FADWA (Classification) - Usage:
from aya_naim.orchestrator import PersonAOrchestrator
orchestrator = PersonAOrchestrator()
result = orchestrator.process_query("patient with chest pain")
cleaned_text = result['cleaned_query']  # Use this for classification

# SINDEL (LLM) - Usage:
from aya_naim.orchestrator import PersonAOrchestrator
orchestrator = PersonAOrchestrator()
result = orchestrator.process_query("diabetes management")
context = result['formatted_context']  # Add to LLM prompt

# STORING CONVERSATIONS:
record_id = orchestrator.store_conversation(
    user_query="What is hypertension?",
    llm_response="Hypertension is high blood pressure..."
)
    """)
    
    print("\n" + "=" * 60)
    print(" PERSON A READY FOR TEAM COLLABORATION!")
    print("=" * 60)
    print("\n Files in place:")
    print("• aya_naim/orchestrator.py - Main orchestrator")
    print("• aya_naim/data_loader.py - Data loading")
    print("• aya_naim/text_preprocessor.py - Text cleaning")
    print("• aya_naim/similarity_engine.py - Similarity search")
    print("• aya_naim/memory_store.py - Database storage")
    print("• aya_naim/metrics.py - Data analysis")
    
except ImportError as e:
    print(f" IMPORT ERROR: {e}")
    print("\nMake sure you have these files:")
    print("1. aya_naim/orchestrator.py")
    print("2. aya_naim/data_loader.py")
    print("3. aya_naim/text_preprocessor.py")
    print("\nRun: python -c \"import sys; sys.path.append('.'); from aya_naim.orchestrator import PersonAOrchestrator; print('Import successful!')\"")
except Exception as e:
    print(f" ERROR: {e}")
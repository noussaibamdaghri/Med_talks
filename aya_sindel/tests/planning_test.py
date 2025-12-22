
from planning import PromptManager, build_medical_prompt

def main():
    print(" Testing PromptManager...")
    
    # Create a test instance
    manager = PromptManager()
    
    # List available prompts
    print("\n Available prompt types:")
    for qtype, filename in manager.list_available_prompts().items():
        print(f"  • {qtype:20} → {filename}")
    
    # Test building a QCM prompt
    print("\n Testing QCM prompt:")
    qcm_prompt = manager.build_prompt(
        question="Which of the following is a symptom of diabetes?",
        question_type="qcm",
        dataset_info="Polyuria, polydipsia, weight loss",
        api_info="Increased thirst and frequent urination are common symptoms."
    )
    
    print(f"Prompt preview (first 300 chars):")
    print("-" * 50)
    print(qcm_prompt[:300] + "..." if len(qcm_prompt) > 300 else qcm_prompt)
    print("-" * 50)
    
    # Test building a definition prompt
    print("\n Testing Definition prompt:")
    def_prompt = manager.build_prompt(
        question="What is hypertension?",
        question_type="definition",
        dataset_info="Hypertension is high blood pressure, defined as >140/90 mmHg.",
        api_info="Primary hypertension has no identifiable cause."
    )
    
    print(f"Prompt preview (first 300 chars):")
    print("-" * 50)
    print(def_prompt[:300] + "..." if len(def_prompt) > 300 else def_prompt)
    print("-" * 50)
    
    # Test the quick function
    print("\n Testing quick function:")
    quick_prompt = build_medical_prompt(
        question="How to perform CPR?",
        question_type="stepwise",
        dataset_context="Step 1: Check responsiveness...",
        api_context="Recent guidelines emphasize chest compression depth."
    )
    
    print(f"Quick function prompt length: {len(quick_prompt)} chars")
    
    print("\n PromptManager tests completed!")

if __name__ == "__main__":
    main()

from handler import LLMHandler, get_llm_answer

def test_handler_basic():
    print("=" * 60)
    print("TESTING LLM HANDLER")
    print("=" * 60)
    
    # Initialize handler
    handler = LLMHandler()
    
    # Show stats
    stats = handler.get_stats()
    print(f"\nInitial Stats:")
    print(f"   Mode: {stats['llm']['mode'].upper()}")
    print(f"   Model: {stats['llm']['model']}")
    
    # Validate prompts
    handler.validate_prompts()
    
    print("\n" + "=" * 60)
    print(" TEST 1: Definition Question")
    print("=" * 60)
    
    # Test 1: Definition question
    result1 = handler.process_question(
        question="What is myocardial infarction?",
        question_type="definition",
        dataset_info="Myocardial infarction is heart attack caused by blocked coronary artery.",
        api_info="Symptoms include chest pain, shortness of breath, and nausea."
    )
    
    print(f" Status: {result1['status']}")
    print(f" Answer length: {len(result1['answer'])} chars")
    print(f"\n Answer preview:")
    print("-" * 40)
    print(result1['answer'][:200] + "..." if len(result1['answer']) > 200 else result1['answer'])
    print("-" * 40)
    
    print("\n" + "=" * 60)
    print("TEST 2: QCM Question")
    print("=" * 60)
    
    # Test 2: QCM question
    result2 = handler.process_question(
        question="Which is a symptom of diabetes?\nA) Polyuria\nB) Bradycardia\nC) Hypotension\nD) Constipation",
        question_type="qcm",
        dataset_info="Polyuria is excessive urination, common in diabetes."
    )
    
    print(f"Status: {result2['status']}")
    print(f"Answer length: {len(result2['answer'])} chars")
    print(f"\nAnswer preview:")
    print("-" * 40)
    print(result2['answer'][:200] + "..." if len(result2['answer']) > 200 else result2['answer'])
    print("-" * 40)
    
    print("\n" + "=" * 60)
    print("TEST 3: Stepwise Question")
    print("=" * 60)
    
    # Test 3: Stepwise question
    result3 = handler.process_question(
        question="How to perform CPR?",
        question_type="stepwise",
        dataset_info="1. Check responsiveness\n2. Call for help\n3. Begin chest compressions"
    )
    
    print(f"Status: {result3['status']}")
    print(f"Answer length: {len(result3['answer'])} chars")
    print(f"\nAnswer preview:")
    print("-" * 40)
    print(result3['answer'][:200] + "..." if len(result3['answer']) > 200 else result3['answer'])
    print("-" * 40)
    
    print("\n" + "=" * 60)
    print("TEST 4: Dataset Only Question")
    print("=" * 60)
    
    # Test 4: Dataset only question
    result4 = handler.process_question(
        question="What is hypertension?",
        question_type="dataset_only",
        dataset_info="Hypertension is defined as blood pressure >140/90 mmHg."
    )
    
    print(f"Status: {result4['status']}")
    print(f"Answer length: {len(result4['answer'])} chars")
    
    print("\n" + "=" * 60)
    print("TEST 5: Quick Function")
    print("=" * 60)
    
    # Test 5: Quick function
    quick_answer = get_llm_answer(
        question="What is asthma?",
        question_type="definition",
        dataset_info="Asthma is a chronic respiratory condition.",
        api_info="Triggered by allergens and exercise."
    )
    
    print(f"Quick answer length: {len(quick_answer)} chars")
    print(f"\n Quick answer preview:")
    print("-" * 40)
    print(quick_answer[:200] + "..." if len(quick_answer) > 200 else quick_answer)
    print("-" * 40)
    
    # Final stats
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    
    final_stats = handler.get_stats()
    print(f" Total questions processed: {final_stats['handler']['total_questions']}")
    print(f" Question types: {final_stats['handler']['question_types']}")
    print(f"Cache size: {final_stats['llm']['cache_size']} responses")
    
    # Summary
    print("\n" + "=" * 60)
    print("HANDLER TEST SUMMARY")
    print("=" * 60)
    
    all_tests = [result1, result2, result3, result4]
    successful = sum(1 for r in all_tests if r["status"] == "success")
    
    print(f" Tests run: {len(all_tests)}")
    print(f" Successful: {successful}")
    print(f" Failed: {len(all_tests) - successful}")
    
    if successful == len(all_tests):
        print("\nAll tests passed! Handler is working correctly.")
    else:
        print(f"\n  {len(all_tests) - successful} test(s) failed.")
    
    print("\nHandler test complete!")

def test_batch_processing():
    """Test batch processing functionality"""
    print("\n" + "=" * 60)
    print("🧪 TESTING BATCH PROCESSING")
    print("=" * 60)
    
    handler = LLMHandler()
    
    questions = [
        "What is pneumonia?",
        "How to treat a fracture?",
        "What are vital signs?"
    ]
    
    question_types = ["definition", "stepwise", "definition"]
    
    dataset_infos = [
        "Pneumonia is lung infection.",
        "Immobilize the fracture.",
        "Blood pressure, heart rate, respiration."
    ]
    
    print(f"Processing {len(questions)} questions in batch...")
    results = handler.batch_process(questions, question_types, dataset_infos)
    
    print(f"\nBatch results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. {questions[i][:30]}... → {result['status']} ({len(result['answer'])} chars)")

def test_error_handling():
    """Test error scenarios"""
    print("\n" + "=" * 60)
    print("TESTING ERROR HANDLING")
    print("=" * 60)
    
    handler = LLMHandler()
    
    # Test empty question
    print("\n Test 1: Empty question")
    result = handler.process_question(
        question="",
        question_type="definition",
        dataset_info="Some info"
    )
    print(f"   Status: {result['status']}")
    print(f"   Contains 'Error': {'Error' in result['answer']}")
    
    # Test invalid question type
    print("\nTest 2: Invalid question type")
    result = handler.process_question(
        question="Normal question",
        question_type="invalid_type",
        dataset_info="Some info"
    )
    print(f"   Status: {result['status']}")
    print(f"   Used fallback: {'dataset_only' in str(result['metadata'])}")

def run_all_tests():
    """Run all test suites"""
    print("STARTING COMPREHENSIVE HANDLER TESTS")
    print("=" * 60)
    
    # Run individual test suites
    test_handler_basic()
    test_batch_processing()
    test_error_handling()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    # Run specific test or all tests
    test_handler_basic()  # Just the basic tests
    # run_all_tests()     # Uncomment to run all tests

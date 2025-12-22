from llm import LLMClient, generate_response

def test_llm_basic():
    print(" Testing LLM Client...")
    
    # Initialize client
    client = LLMClient()
    
    # Show stats
    stats = client.get_stats()
    print(f"\n Client Status:")
    for key, value in stats.items():
        print(f"  {key:20}: {value}")
    
    # Test generation
    test_prompts = [
        ("What is diabetes?", "General medical"),
        ("Which is correct? A) X B) Y C) Z", "QCM"),
        ("How to take blood pressure?", "Stepwise"),
    ]
    
    for prompt, prompt_type in test_prompts:
        print(f"\n Testing {prompt_type} prompt:")
        print(f"   Prompt: {prompt[:50]}...")
        
        response = client.generate(prompt)
        
        print(f"   Response length: {len(response):,} chars")
        print(f"   Preview: {response[:100]}...")
    
    # Test cache
    print(f"\n Testing cache...")
    test_prompt = "Test cache prompt"
    response1 = client.generate(test_prompt)
    response2 = client.generate(test_prompt)  # Should use cache
    
    print(f"   First call: {len(response1):,} chars")
    print(f"   Second call: {len(response2):,} chars")
    
    # Test quick function
    print(f"\n Testing quick function...")
    quick_response = generate_response("Quick test prompt")
    print(f"   Quick response: {len(quick_response):,} chars")
    
    print("\n LLM tests completed!")

if __name__ == "__main__":
    test_llm_basic()

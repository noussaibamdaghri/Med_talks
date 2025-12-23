import os
import time
import json
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI package not installed. Use: pip install openai")


class LLMClient:
    """
    Simple LLM client for generating responses
    - Supports OpenAI GPT models
    - Has retry logic for reliability
    - Includes caching to reduce API calls
    - Works in development mode without API key
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 1000,
        cache_enabled: bool = True,
        cache_dir: str = ".llm_cache"
    ):
        """
        Initialize the LLM client
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use (gpt-3.5-turbo, gpt-4, etc.)
            temperature: Creativity level (0.0-1.0)
            max_tokens: Maximum response length
            cache_enabled: Cache responses to avoid duplicate calls
            cache_dir: Directory for cache files
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir
        
        # Get API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.has_api_key = bool(self.api_key)
        
        # Initialize client if available
        if OPENAI_AVAILABLE and self.has_api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.mode = "production"
            print(f" LLM Client initialized (Production Mode)")
            print(f"   Model: {model}, Cache: {'enabled' if cache_enabled else 'disabled'}")
        else:
            self.client = None
            self.mode = "development"
            print(f" LLM Client initialized (Development Mode)")
            print(f"   Mock responses will be generated")
            if not OPENAI_AVAILABLE:
                print("   Install: pip install openai")
            if not self.has_api_key:
                print("   Set OPENAI_API_KEY environment variable")
        
        # Setup cache directory
        if cache_enabled:
            os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate a unique cache key for the prompt"""
        # Include model and settings in the key
        key_content = f"{prompt}_{self.model}_{self.temperature}_{self.max_tokens}"
        return hashlib.md5(key_content.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get the cache file path for a key"""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Load response from cache if available and recent"""
        if not self.cache_enabled:
            return None
        
        cache_file = self._get_cache_path(cache_key)
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if cache is still fresh (24 hours)
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                current_time = datetime.now()
                hours_old = (current_time - cache_time).total_seconds() / 3600
                
                if hours_old < 24:  # Cache valid for 24 hours
                    print(f" Cache hit: {cache_key}")
                    return cache_data['response']
                else:
                    print(f" Cache expired: {cache_key} ({hours_old:.1f} hours old)")
            except Exception as e:
                print(f" Cache read error: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, response: str):
        """Save response to cache"""
        if not self.cache_enabled:
            return
        
        try:
            cache_data = {
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'model': self.model,
                'temperature': self.temperature,
                'prompt_length': len(response)
            }
            
            cache_file = self._get_cache_path(cache_key)
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            print(f" Cached response: {cache_key}")
        except Exception as e:
            print(f" Cache save error: {e}")
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate a response for the given prompt
        
        Args:
            prompt: The prompt to send to the LLM
            max_retries: Number of retry attempts on failure
            
        Returns:
            str: The generated response
        """
        if not prompt or not prompt.strip():
            return "Error: Empty prompt provided"
        
        # Check cache first
        cache_key = self._get_cache_key(prompt)
        cached_response = self._load_from_cache(cache_key)
        if cached_response:
            return cached_response
        
        # Development mode (no API key)
        if self.mode == "development":
            response = self._generate_mock_response(prompt)
            if self.cache_enabled:
                self._save_to_cache(cache_key, response)
            return response
        
        # Production mode (with API)
        for attempt in range(max_retries):
            try:
                print(f" API Call (Attempt {attempt + 1}/{max_retries})...")
                
           response = self.client.responses.create(
    model=self.model,
    input=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ],
    temperature=self.temperature,
    max_output_tokens=self.max_tokens
)

                # Extract response text
                if response.choices and response.choices[0].message:
                    result = response.output_text.strip()
                    
                    # Save to cache
                    if self.cache_enabled:
                        self._save_to_cache(cache_key, result)
                    
                    # Log token usage (optional)
                    if hasattr(response, 'usage'):
                        tokens = response.usage.total_tokens
                        print(f"   Tokens used: {tokens}")
                    
                    return result
                else:
                    raise ValueError("API returned empty response")
                    
            except Exception as e:
                error_msg = str(e)
                print(f" Attempt {attempt + 1} failed: {error_msg}")
                
                # Last attempt failed
                if attempt == max_retries - 1:
                    print(" All retries failed, returning error message")
                    return f"Error: Failed to generate response after {max_retries} attempts. {error_msg}"
                
                # Wait before retrying (exponential backoff)
                wait_time = 2 ** attempt
                print(f"   Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        return "Error: Failed to generate response"
    
    def _generate_mock_response(self, prompt: str) -> str:
        """
        Generate a mock response for development/testing
        """
        print("\n" + "═" * 60)
        print(" DEVELOPMENT MODE: Generating Mock Response")
        print(f" Prompt length: {len(prompt):,} characters")
        print("═" * 60)
        
        # Show prompt preview
        preview = prompt[:300].replace('\n', ' ')
        if len(prompt) > 300:
            preview += "..."
        print(f"\nPrompt preview: {preview}\n")
        print("═" * 60 + "\n")
        
        # Generate context-aware mock response
        prompt_lower = prompt.lower()
        
        # QCM/Exam questions
        if "medical multiple choice question" in prompt_lower or "multiple choice" in prompt_lower or "which of the following" in prompt_lower:
            return """**Mock QCM Response**

 Correct answer: C

 Justification: 
Option C is correct based on current medical guidelines and pathophysiology. It accurately reflects the most common presentation and mechanism.

 Analysis of other options:
• Option A: Incorrect - This describes a less common variant
• Option B: Incorrect - While sometimes associated, it's not the primary feature
• Option D: Incorrect - This is actually a contraindication

*Note: This is a mock response. Set OPENAI_API_KEY for real analysis.*"""
        
        # Definition questions
        elif "medical definition question" in prompt_lower or "what is" in prompt_lower or "define" in prompt_lower:
            return """**Mock Definition Response**

## Overview
[Medical term] is a [category] condition characterized by [key features].

## Key Characteristics
• Primary feature: [Description]
• Pathophysiology: [Mechanism]
• Clinical presentation: [Symptoms and signs]

## Diagnosis
• Diagnostic criteria: [How it's identified]
• Differential diagnosis: [What to rule out]

## Management
• First-line treatment: [Primary approach]
• Supportive care: [Additional measures]
• Monitoring: [Follow-up requirements]

*Note: This is a mock response. Set OPENAI_API_KEY for accurate medical definitions.*"""
        
        # Stepwise/Procedure questions
        elif "medical stepwise prodcedure question" in prompt_lower or "procedure" in prompt_lower or "how to" in prompt_lower:
            return """**Mock Stepwise Response**

**Procedure: [Procedure Name]**

**Pre-procedure:**
1. Verify patient identification and consent
2. Prepare necessary equipment and medications
3. Ensure proper positioning and sterile field

**During procedure:**
1. Perform initial assessment and baseline measurements
2. Execute primary steps with monitoring
3. Document findings and any deviations

**Post-procedure:**
1. Monitor patient for immediate complications
2. Provide appropriate aftercare instructions
3. Schedule follow-up as needed

**Key Points:**
• Maintain aseptic technique throughout
• Monitor vital signs continuously
• Have emergency equipment available

*Note: This is a mock response. Set OPENAI_API_KEY for detailed procedural guidance.*"""
        
        # Clinical reasoning
        elif "medical reasoning question" in prompt_lower or "differential" in prompt_lower or "clinical case" in prompt_lower:
            return """**Mock Clinical Reasoning Response**

**Case Analysis:**

**Presenting Features:**
• [Key symptom 1]
• [Key symptom 2]
• [Relevant history]

**Differential Diagnosis:**
1. **Most likely:** [Primary diagnosis] - Best fits presentation
2. **Consider:** [Secondary possibility] - Less likely but important
3. **Rule out:** [Critical exclusion] - Must be excluded

**Investigations:**
• Initial: [Basic tests]
• Confirmatory: [Definitive tests]
• Monitoring: [Follow-up tests]

**Management Plan:**
• Immediate: [Urgent actions]
• Short-term: [Treatment plan]
• Long-term: [Follow-up care]

*Note: This is a mock response. Set OPENAI_API_KEY for detailed clinical reasoning.*"""
        
        # General/fallback response
        else:
            return """**Mock Medical Response**

Based on the provided information, here is a comprehensive answer:

**Summary:**
[Brief overview of the topic]

**Key Points:**
• Point 1: [Important information]
• Point 2: [Additional details]
• Point 3: [Clinical relevance]

**Recommendations:**
1. [First recommendation]
2. [Second recommendation]
3. [Third recommendation]

**Important Considerations:**
- Always consult current guidelines
- Consider individual patient factors
- Monitor for potential complications

*Note: This response was generated in development mode. For accurate, up-to-date medical information, set the OPENAI_API_KEY environment variable and ensure you have an active internet connection.*"""
    
    def clear_cache(self):
        """Clear all cached responses"""
        if os.path.exists(self.cache_dir):
            import shutil
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            print(" Cache cleared")
        else:
            print("  Cache directory doesn't exist")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics and status"""
        cache_files = []
        if os.path.exists(self.cache_dir):
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
        
        return {
            "mode": self.mode,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "cache_enabled": self.cache_enabled,
            "cache_size": len(cache_files),
            "has_api_key": self.has_api_key,
            "openai_available": OPENAI_AVAILABLE
        }


def generate_response(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    **kwargs
) -> str:
    """
    Quick function to generate a response
    
    Example:
        response = generate_response("Explain hypertension", model="gpt-4")
    """
    client = LLMClient(api_key=api_key, model=model, **kwargs)
    return client.generate(prompt)

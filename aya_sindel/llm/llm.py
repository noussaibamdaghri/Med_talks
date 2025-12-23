import os
import time
import json
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers package not installed. Use: pip install torch transformers")


class LLMClient:
    """
    Simple LLM client for generating responses
    - Supports BLOOM models (open-source, free)
    - Has retry logic for reliability
    - Includes caching to reduce computation
    - Works in development mode without model
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,  # Kept for compatibility (ignored)
        model: str = "bloom-1.7b",  # Changed from "gpt-4o-mini"
        temperature: float = 0.3,
        max_tokens: int = 500,  # Reduced for BLOOM
        cache_enabled: bool = True,
        cache_dir: str = ".llm_cache"
    ):
        """
        Initialize the LLM client
        
        Args:
            api_key: Kept for compatibility (ignored for BLOOM)
            model: Model to use (bloom-560m, bloom-1.7b, bloom-3b)
            temperature: Creativity level (0.0-1.0)
            max_tokens: Maximum response length
            cache_enabled: Cache responses to avoid duplicate calls
            cache_dir: Directory for cache files
        """
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir
        
        # Extract BLOOM model size
        if model.startswith("bloom-"):
            self.bloom_size = model.replace("bloom-", "")
        else:
            self.bloom_size = "1.7b"  # Default
        
        # Map model sizes to HuggingFace IDs
        self.model_map = {
            "560m": "bigscience/bloom-560m",
            "1.7b": "bigscience/bloom-1b7",
            "3b": "bigscience/bloom-3b"
        }
        
        self.model_id = self.model_map.get(self.bloom_size, "bigscience/bloom-1b7")
        
        # Initialize model if available
        if TRANSFORMERS_AVAILABLE:
            try:
                self._initialize_model()
                self.mode = "production"
                print(f" LLM Client initialized (Production Mode)")
                print(f"   Model: BLOOM-{self.bloom_size}")
            except Exception as e:
                print(f" Error loading BLOOM model: {e}")
                print(" Falling back to development mode")
                self.mode = "development"
                self.model = None
                self.tokenizer = None
        else:
            self.mode = "development"
            self.model = None
            self.tokenizer = None
            print(f" LLM Client initialized (Development Mode)")
            print("   Mock responses will be generated")
            if not TRANSFORMERS_AVAILABLE:
                print("   Install: pip install torch transformers")
        
        # Setup cache directory
        if cache_enabled:
            os.makedirs(cache_dir, exist_ok=True)
    
    def _initialize_model(self):
        """Initialize the BLOOM model and tokenizer"""
        print(f"   Loading BLOOM-{self.bloom_size}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Move to device
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print(f"   Using GPU")
        else:
            print(f"   Using CPU")
        
        self.model.eval()  # Set to evaluation mode
        print(f"   Model loaded successfully!")
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate a unique cache key for the prompt"""
        key_content = f"{prompt}_{self.model_name}_{self.temperature}_{self.max_tokens}"
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
                'model': self.model_name,
                'temperature': self.temperature,
                'prompt_length': len(response)
            }
            
            cache_file = self._get_cache_path(cache_key)
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            print(f" Cached response: {cache_key}")
        except Exception as e:
            print(f" Cache save error: {e}")
    
    def generate(self, prompt: str, max_retries: int = 2) -> str:
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
        
        # Development mode (no model available)
        if self.mode == "development":
            response = self._generate_mock_response(prompt)
            if self.cache_enabled:
                self._save_to_cache(cache_key, response)
            return response
        
        # Production mode (with BLOOM)
        for attempt in range(max_retries):
            try:
                print(f" Generating with BLOOM (Attempt {attempt + 1}/{max_retries})...")
                start_time = time.time()
                
                # Format prompt for BLOOM
                formatted_prompt = f"Instruction: {prompt}\n\nResponse:"
                
                # Tokenize
                inputs = self.tokenizer(
                    formatted_prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=2048
                )
                
                # Move to device
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        top_p=0.95,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                # Decode only the new tokens
                prompt_length = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0][prompt_length:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                elapsed = time.time() - start_time
                print(f"   Generated in {elapsed:.2f}s")
                
                # Clean up response
                response = response.strip()
                
                # Save to cache
                if self.cache_enabled:
                    self._save_to_cache(cache_key, response)
                
                return response
                    
            except Exception as e:
                error_msg = str(e)
                print(f" Attempt {attempt + 1} failed: {error_msg}")
                
                # Last attempt failed
                if attempt == max_retries - 1:
                    print(" All retries failed, returning fallback")
                    return self._generate_mock_response(prompt)
                
                # Wait before retrying
                wait_time = 1
                print(f"   Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        return self._generate_mock_response(prompt)
    
    def _generate_mock_response(self, prompt: str) -> str:
        """
        Generate a mock response for development/testing
        """
        print("\n" + "═" * 60)
        print(f" DEVELOPMENT MODE: Generating Mock Response (BLOOM-{self.bloom_size})")
        print(f" Prompt length: {len(prompt):,} characters")
        print("═" * 60)
        
        # Show prompt preview
        preview = prompt[:300].replace('\n', ' ')
        if len(prompt) > 300:
            preview += "..."
        print(f"\nPrompt preview: {preview}\n")
        print("═" * 60 + "\n")
        
        # Same mock responses as before
        prompt_lower = prompt.lower()
        
        if "medical multiple choice question" in prompt_lower or "multiple choice" in prompt_lower or "which of the following" in prompt_lower:
            return """**Mock QCM Response**

 Correct answer: C

 Justification: 
Option C is correct based on current medical guidelines and pathophysiology.

*Note: This is a mock response. Install transformers for BLOOM model.*"""
        
        elif "medical definition question" in prompt_lower or "what is" in prompt_lower or "define" in prompt_lower:
            return """**Mock Definition Response**

## Overview
[Medical term] is a [category] condition characterized by [key features].

*Note: This is a mock response. Install transformers for BLOOM model.*"""
        
        elif "medical stepwise prodcedure question" in prompt_lower or "procedure" in prompt_lower or "how to" in prompt_lower:
            return """**Mock Stepwise Response**

**Procedure: [Procedure Name]**

**Pre-procedure:**
1. Verify patient identification and consent

*Note: This is a mock response. Install transformers for BLOOM model.*"""
        
        elif "medical reasoning question" in prompt_lower or "differential" in prompt_lower or "clinical case" in prompt_lower:
            return """**Mock Clinical Reasoning Response**

**Case Analysis:**

**Presenting Features:**
• [Key symptom 1]

*Note: This is a mock response. Install transformers for BLOOM model.*"""
        
        else:
            return f"""**Mock Medical Response**

Based on the provided information, here is a response:

*Note: This response was generated in development mode. For actual BLOOM model responses, install: pip install torch transformers*"""
    
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
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "cache_enabled": self.cache_enabled,
            "cache_size": len(cache_files),
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "bloom_size": self.bloom_size
        }


def generate_response(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "bloom-1.7b",  # Changed from "gpt-3.5-turbo"
    **kwargs
) -> str:
    """
    Quick function to generate a response
    
    Example:
        response = generate_response("Explain hypertension", model="bloom-3b")
    """
    client = LLMClient(api_key=api_key, model=model, **kwargs)
    return client.generate(prompt)

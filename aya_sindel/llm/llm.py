import os
import time
import json
import hashlib
from typing import Optional, Dict, Any, List
from datetime import datetime

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers package not installed. Use: pip install torch transformers accelerate bitsandbytes")


class BLOOMClient:
    
    def __init__(
        self,
        model_size: str = "1.7b",
        device: str = "auto",
        temperature: float = 0.7,
        max_tokens: int = 500,
        cache_enabled: bool = True,
        cache_dir: str = ".llm_cache",
        use_quantization: bool = True,
        download_model: bool = True
    ):
        self.model_size = model_size
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir
        self.use_quantization = use_quantization
        
        # Map model sizes to HuggingFace IDs
        self.model_map = {
            "560m": "bigscience/bloom-560m",
            "1.7b": "bigscience/bloom-1b7",
            "3b": "bigscience/bloom-3b",
            "7b": "bigscience/bloom-7b1"  # Requires 16GB+ RAM
        }
        
        if model_size not in self.model_map:
            print(f"Warning: Model size {model_size} not recognized. Using 1.7b")
            model_size = "1.7b"
        
        self.model_id = self.model_map[model_size]
        
        # Check for transformers availability
        if not TRANSFORMERS_AVAILABLE:
            self.mode = "mock"
            print(" BLOOM Client initialized (Mock Mode)")
            print("   Transformers not installed. Use: pip install torch transformers accelerate")
        else:
            # Initialize model
            try:
                self._initialize_model(download_model)
                self.mode = "production"
                print(f" BLOOM Client initialized (Production Mode)")
                print(f"   Model: BLOOM-{model_size}")
                print(f"   Device: {self.actual_device}")
                print(f"   Quantization: {'enabled' if use_quantization else 'disabled'}")
                print(f"   Cache: {'enabled' if cache_enabled else 'disabled'}")
            except Exception as e:
                print(f" Error loading BLOOM model: {e}")
                print(" Falling back to mock mode")
                self.mode = "mock"
        
        # Setup cache directory
        if cache_enabled:
            os.makedirs(cache_dir, exist_ok=True)
    
    def _initialize_model(self, download_model: bool = True):
        """Initialize the BLOOM model and tokenizer"""
        print(f"Loading BLOOM-{self.model_size}... This may take a few minutes.")
        
        # Determine device
        if self.device == "auto":
            if torch.cuda.is_available():
                self.actual_device = "cuda"
                print(f"   Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                self.actual_device = "cpu"
                print(f"   Using CPU (no GPU detected)")
        else:
            self.actual_device = self.device
        
        # Configure quantization if requested and on CUDA
        bnb_config = None
        if self.use_quantization and self.actual_device == "cuda":
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                print("   4-bit quantization enabled")
            except Exception as e:
                print(f"   Quantization setup failed: {e}")
                print("   Continuing without quantization")
        
        # Load tokenizer
        print(f"   Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir="./models" if download_model else None
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        print(f"   Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto" if self.actual_device == "cuda" else None,
            torch_dtype=torch.float16 if self.actual_device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            cache_dir="./models" if download_model else None,
            trust_remote_code=True
        )
        
        # Move to CPU if needed
        if self.actual_device == "cpu":
            self.model = self.model.to("cpu")
            self.model = self.model.float()  # Ensure float32 on CPU
        
        self.model.eval()  # Set to evaluation mode
        print(f"   BLOOM-{self.model_size} loaded successfully!")
    
    def _format_messages(self, prompt: str) -> str:
        """
        Format prompt for BLOOM (BLOOM doesn't have built-in chat template)
        """
        # Simple formatting - BLOOM works well with clear instruction prompts
        formatted_prompt = f"""Instruction: You are a helpful medical AI assistant. Provide accurate, clear, and concise responses.

Question: {prompt}

Response:"""
        
        return formatted_prompt
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate a unique cache key for the prompt"""
        key_content = f"{prompt}_{self.model_size}_{self.temperature}_{self.max_tokens}"
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
                'model': f"BLOOM-{self.model_size}",
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
            prompt: The prompt to send to BLOOM
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
        
        # Mock mode (transformers not installed)
        if self.mode == "mock":
            response = self._generate_mock_response(prompt)
            if self.cache_enabled:
                self._save_to_cache(cache_key, response)
            return response
        
        # Production mode with BLOOM
        formatted_prompt = self._format_messages(prompt)
        
        for attempt in range(max_retries):
            try:
                print(f" Generating with BLOOM-{self.model_size} (Attempt {attempt + 1}/{max_retries})...")
                start_time = time.time()
                
                # Tokenize
                inputs = self.tokenizer(
                    formatted_prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=2048 - self.max_tokens
                )
                
                # Move to device
                if self.actual_device == "cuda":
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
                    
            except RuntimeError as e:
                error_msg = str(e)
                print(f" Attempt {attempt + 1} failed: {error_msg}")
                
                # Check for out-of-memory errors
                if "out of memory" in error_msg.lower():
                    print("   Out of memory! Try smaller model or enable quantization")
                    return "Error: Out of memory. Try using BLOOM-560M or enable quantization."
                
                # Last attempt failed
                if attempt == max_retries - 1:
                    print(" All retries failed, returning fallback")
                    return self._generate_fallback_response(prompt)
                
                # Wait before retrying
                wait_time = 1 * (attempt + 1)
                print(f"   Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            
            except Exception as e:
                error_msg = str(e)
                print(f" Unexpected error: {error_msg}")
                return self._generate_fallback_response(prompt)
        
        return self._generate_fallback_response(prompt)
    
    def _generate_mock_response(self, prompt: str) -> str:
        """
        Generate a mock response when transformers not available
        """
        print("\n" + "═" * 60)
        print(" MOCK MODE: BLOOM not available")
        print(f" Prompt length: {len(prompt):,} characters")
        print("═" * 60)
        
        # Show prompt preview
        preview = prompt[:200].replace('\n', ' ')
        if len(prompt) > 200:
            preview += "..."
        print(f"\nPrompt preview: {preview}\n")
        print("═" * 60 + "\n")
        
        # Simple mock response
        return f"""**BLOOM Mock Response (BLOOM-{self.model_size})**

This is a mock response because the BLOOM model is not loaded.

To use the actual BLOOM model:
1. Install required packages: `pip install torch transformers accelerate bitsandbytes`
2. Restart the application

For now, here's what BLOOM would analyze about your query:

**Query Summary:** Your question appears to be about: {preview}

**Expected BLOOM Response:**
BLOOM-{self.model_size} would provide a detailed, multilingual response based on its training data.

**Note:** BLOOM is a 100% free, open-source model that runs locally on your machine."""
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """
        Generate a fallback response when BLOOM fails
        """
        return f"""**BLOOM-{self.model_size} Response**

I encountered an issue generating a complete response with BLOOM. Here's what I can provide:

**Analysis of your query:** Your question requires medical knowledge that BLOOM can provide with proper configuration.

**Troubleshooting steps:**
1. Ensure you have enough memory (BLOOM-{self.model_size} needs ~{self._get_memory_requirement()} RAM)
2. Try a smaller model: Use `model_size="560m"` for less memory usage
3. Enable quantization in constructor: `use_quantization=True`

**Quick answer based on available information:**
Please consult medical guidelines or a healthcare professional for accurate information.

*Model: BLOOM-{self.model_size} (Open-source, free)*"""
    
    def _get_memory_requirement(self) -> str:
        """Get approximate memory requirement for current model"""
        requirements = {
            "560m": "2-3 GB",
            "1.7b": "6-8 GB", 
            "3b": "10-12 GB",
            "7b": "16-20 GB"
        }
        return requirements.get(self.model_size, "8-10 GB")
    
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
        
        stats = {
            "mode": self.mode,
            "model": f"BLOOM-{self.model_size}",
            "model_id": self.model_id,
            "device": self.actual_device if hasattr(self, 'actual_device') else self.device,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "cache_enabled": self.cache_enabled,
            "cache_size": len(cache_files),
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "quantization_enabled": self.use_quantization,
            "memory_requirement": self._get_memory_requirement()
        }
        
        # Add GPU info if available
        if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
            stats["gpu_name"] = torch.cuda.get_device_name(0)
            stats["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return stats
    
    def batch_generate(self, prompts: List[str], batch_size: int = 2) -> List[str]:
        """
        Generate responses for multiple prompts (batched for efficiency)
        
        Args:
            prompts: List of prompts to process
            batch_size: Number of prompts to process at once
            
        Returns:
            List of responses
        """
        if self.mode == "mock":
            return [self._generate_mock_response(p) for p in prompts]
        
        responses = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
            
            for prompt in batch:
                response = self.generate(prompt)
                responses.append(response)
        
        return responses


# Compatibility wrapper for existing code
class LLMClient(BLOOMClient):
    """
    Backward compatibility - maintains the original LLMClient interface
    but uses BLOOM instead of OpenAI
    """
    def __init__(
        self,
        api_key: Optional[str] = None,  # Kept for compatibility (ignored)
        model: str = "bloom-1.7b",  # Now refers to BLOOM model size
        temperature: float = 0.7,
        max_tokens: int = 500,
        cache_enabled: bool = True,
        cache_dir: str = ".llm_cache",
        **kwargs
    ):
        # Extract BLOOM model size from model string
        model_size = "1.7b"  # Default
        if "bloom-" in model.lower():
            model_size = model.lower().replace("bloom-", "")
        
        # Initialize BLOOM client
        super().__init__(
            model_size=model_size,
            temperature=temperature,
            max_tokens=max_tokens,
            cache_enabled=cache_enabled,
            cache_dir=cache_dir,
            **kwargs
        )


def generate_response(
    prompt: str,
    model_size: str = "1.7b",
    **kwargs
) -> str:
    """
    Quick function to generate a response with BLOOM
    
    Example:
        response = generate_response("Explain hypertension", model_size="3b")
    """
    client = BLOOMClient(model_size=model_size, **kwargs)
    return client.generate(prompt)


# Example usage
if __name__ == "__main__":
    # Quick test
    print("Testing BLOOM Client...")
    
    # Test with smallest model
    client = BLOOMClient(model_size="560m", cache_enabled=True)
    
    # Get stats
    stats = client.get_stats()
    print("\nClient Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test generation
    test_prompt = "What is the treatment for hypertension?"
    print(f"\nGenerating response for: {test_prompt[:50]}...")
    
    response = client.generate(test_prompt)
    print(f"\nResponse:\n{response}")
    
    # Clear cache if needed
    # client.clear_cache()

import os
from typing import Optional, Dict

class PromptManager:
    def __init__(self, prompts_dir: str = "prompts"):
        """
        Initialize the prompt manager
        
        Args:
            prompts_dir: Directory containing prompt text files
        """
        self.prompts_dir = prompts_dir
        self.prompts_cache = {}
        
        # Map question types to prompt files (based on your prompts)
        self.prompt_mapping = {
            "medical multiple choice question": "aya_sindel/prompts/qcm.txt",
            "medical definition question": "aya_sindel/prompts/api_prompt.txt", 
            "medical reasoning question": "aya_sindel/prompts/api_prompt.txt",
            "medical stepwise prodcedure question": "aya_sindel/prompts/stepwise.txt"
        }
        
        self._verify_prompt_files()
    
    def _verify_prompt_files(self) -> None:
        """Check that all expected prompt files exist"""
        required_files = set(self.prompt_mapping.values())
        
        for filename in required_files:
            filepath = os.path.join(self.prompts_dir, filename)
            if not os.path.exists(filepath):
                print(f" Prompt file not found: {filename}")
                print(f"   Expected at: {filepath}")
                print(f"   Using fallback for {filename}")
    
    def get_prompt_template(self, question_type: str) -> str:

        filename = self.prompt_mapping.get(
            question_type.lower(), 
            "dataset_only.txt"  # Default fallback
        )
        
        if filename in self.prompts_cache:
            return self.prompts_cache[filename]
        
        filepath = os.path.join(self.prompts_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                template = f.read()
            
            self.prompts_cache[filename] = template
            return template
            
        except FileNotFoundError:
            print(f" Error: Prompt file not found: {filepath}")
            print(f"   Using empty template for type: {question_type}")
            
            fallback = """You are a medical assistant. 

Question:
{question}

Reference answer from medical dataset:
{dataset_info}

Rewrite the answer in a clear, concise, and student-friendly manner.
Ensure medical accuracy.
Do not add new information beyond the reference answer."""
            
            self.prompts_cache[filename] = fallback
            return fallback
    
    def build_prompt(
        self,
        question: str,
        question_type: str,
        dataset_info: Optional[str] = None,
        api_info: Optional[str] = None
    ) -> str:
        """
        Build a complete prompt by filling the template
        
        Args:
            question: The user's question
            question_type: Type of question 
            dataset_info: Information from medical dataset 
            api_info: Information from external APIs 
            
        Returns:
            str: The complete formatted prompt
        """
        # Get the template
        template = self.get_prompt_template(question_type)
        
        # Prepare the context variables
        # Use consistent variable names as in your prompts
        context_vars = {
            "question": question,
            
            # For qcm.txt and stepwise.txt
            "dataset_answer_or_api_info": dataset_info or "No reference information available.",
            
            # For definition.txt
            "dataset_answer": dataset_info or "No dataset answer available.",
            "api_results": api_info or "No external information available.",
            
            # For dataset_only.txt
            "dataset_answer": dataset_info or "No reference answer available.",
            
            # NEW: Consistent variable names (recommended)
            "dataset_info": dataset_info or "No dataset information available.",
            "api_info": api_info or "No external API information available."
        }
        
        # Apply the template with all variables
        # This handles all your different variable names
        prompt = template
        
        # Replace all possible variable names
        for var_name, var_value in context_vars.items():
            placeholder = "{" + var_name + "}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, var_value)
        
        # Debug: Show which variables were used
        if "DEBUG_PROMPTS" in os.environ:
            print(f"\n Building prompt for type: {question_type}")
            print(f"   Question: {question[:50]}...")
            print(f"   Dataset info length: {len(dataset_info or '')}")
            print(f"   API info length: {len(api_info or '')}")
            print(f"   Final prompt length: {len(prompt)} chars")
        
        return prompt
    
    def list_available_prompts(self) -> Dict[str, str]:
        """List all available prompt types and their files"""
        return {
            question_type: filename 
            for question_type, filename in self.prompt_mapping.items()
        }


# Simple function for easy use
def build_medical_prompt(
    question: str,
    question_type: str,
    dataset_context: Optional[str] = None,
    api_context: Optional[str] = None
) -> str:
    """
    Quick function to build a prompt
    
    Example:
        prompt = build_medical_prompt(
            question="What is diabetes?",
            question_type="definition",
            dataset_context="Diabetes is a chronic condition...",
            api_context="According to recent studies..."
        )
    """
    manager = PromptManager()
    return manager.build_prompt(
        question=question,
        question_type=question_type,
        dataset_info=dataset_context,
        api_info=api_context
    )

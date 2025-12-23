import os
import json
from typing import Dict, Any, Optional
from datetime import datetime

from planning import PromptManager
from llm.llm import LLMClient, generate_response


class LLMHandler:
    """
    Main handler that orchestrates the complete LLM response generation
    """
    
    def __init__(
        self,
        llm_model: str = "bloom-1.7b",  # Changed default from "gpt-3.5-turbo"
        llm_temperature: float = 0.3,
        enable_cache: bool = True,
        prompts_dir: str = "aya_sindel/prompts"
    ):
        """
        Initialize the LLM Handler
        """
        # Initialize modules
        self.prompt_manager = PromptManager(prompts_dir=prompts_dir)
        self.llm_client = LLMClient(
            model=llm_model,
            temperature=llm_temperature,
            cache_enabled=enable_cache
        )
        
        # Track usage statistics
        self.stats = {
            "total_questions": 0,
            "by_type": {},
            "start_time": datetime.now().isoformat()
        }
    
    def process_question(
        self,
        question: str,
        question_type: str,
        dataset_info: Optional[str] = None,
        api_info: Optional[str] = None,
        format_output: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single question through the complete pipeline
        """
        # Update statistics
        self.stats["total_questions"] += 1
        self.stats["by_type"][question_type] = self.stats["by_type"].get(question_type, 0) + 1
        
        try:
            # Step 1: Build the prompt
            prompt = self.prompt_manager.build_prompt(
                question=question,
                question_type=question_type,
                dataset_info=dataset_info,
                api_info=api_info
            )
            
            # Step 2: Generate LLM response
            raw_response = self.llm_client.generate(prompt)
            
            # Step 3: Basic formatting
            if format_output:
                answer = self._format_response(raw_response, question_type)
            else:
                answer = raw_response
            
            # Prepare response metadata
            metadata = {
                "question_type": question_type,
                "prompt_length": len(prompt),
                "response_length": len(answer),
                "has_dataset_info": bool(dataset_info),
                "has_api_info": bool(api_info),
                "timestamp": datetime.now().isoformat(),
                "model": self.llm_client.model
            }
            
            return {
                "answer": answer,
                "metadata": metadata,
                "status": "success"
            }
            
        except Exception as e:
            # Fallback error response
            error_response = self._get_error_response(question, question_type, str(e))
            
            return {
                "answer": error_response,
                "metadata": {
                    "error": str(e),
                    "question_type": question_type,
                    "timestamp": datetime.now().isoformat(),
                    "status": "error"
                },
                "status": "error"
            }
    
    def _format_response(self, response: str, question_type: str) -> str:
        """
        Basic response formatting based on question type
        """
        # Basic cleanup
        response = response.strip()
        
        # Remove any leading/trailing quotes
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        
        # Add formatting based on type
        if question_type == "medical multiple choice question":
            # Ensure clear answer indication for QCM
            if not any(marker in response.lower() for marker in ["correct:", "answer:", "✅", "✓"]):
                response = "✅ Correct answer:\n" + response
        
        elif question_type == "medical stepwise prodcedure question":
            # Ensure step formatting
            if not any(marker in response.lower() for marker in ["step", "étape", "1.", "•"]):
                lines = response.split('\n')
                formatted = []
                for i, line in enumerate(lines, 1):
                    if line.strip():
                        formatted.append(f"Step {i}: {line.strip()}")
                response = '\n'.join(formatted)
        
        return response
    
    def _get_error_response(self, question: str, question_type: str, error: str) -> str:
        """
        Generate a user-friendly error response
        """
        error_templates = {
            "medical multiple choice question": f"I apologize, but I encountered an error while analyzing this multiple choice question.\n\nError: {error}\n\nPlease try again or rephrase your question.",
            "medical definition question": f"I'm unable to provide a definition at the moment due to a technical issue.\n\nError: {error}\n\nYou might try searching medical textbooks or online resources for information about: {question}",
            "medical reasoning question": f"I'm having difficulty analyzing this clinical case right now.\n\nError: {error}\n\nConsider consulting clinical guidelines or speaking with a healthcare professional.",
            "medical stepwise prodcedure question": f"I cannot provide step-by-step instructions at this time due to a system error.\n\nError: {error}\n\nFor procedural guidance, please refer to official medical protocols.",
            "default": f"I apologize, but I'm unable to process your question at the moment.\n\nTechnical issue: {error}\n\nPlease try again later or consult appropriate medical resources."
        }
        
        return error_templates.get(question_type, error_templates["default"])
    
    def batch_process(
        self,
        questions: list,
        question_types: list,
        dataset_infos: Optional[list] = None,
        api_infos: Optional[list] = None
    ) -> list:
        """
        Process multiple questions in sequence
        """
        results = []
        dataset_infos = dataset_infos or [None] * len(questions)
        api_infos = api_infos or [None] * len(questions)
        
        for i, (question, q_type, dataset_info, api_info) in enumerate(
            zip(questions, question_types, dataset_infos, api_infos)
        ):
            result = self.process_question(
                question=question,
                question_type=q_type,
                dataset_info=dataset_info,
                api_info=api_info
            )
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get handler statistics
        """
        llm_stats = self.llm_client.get_stats()
        
        return {
            "handler": {
                "total_questions": self.stats["total_questions"],
                "question_types": self.stats["by_type"],
                "start_time": self.stats["start_time"]
            },
            "llm": llm_stats,
            "prompts": {
                "directory": self.prompt_manager.prompts_dir,
                "cached_templates": len(self.prompt_manager.prompts_cache)
            }
        }
    
    def clear_cache(self):
        """Clear LLM response cache"""
        self.llm_client.clear_cache()
    
    def validate_prompts(self) -> Dict[str, bool]:
        """
        Validate that all prompt files exist and are readable
        """
        prompt_files = ["api_prompt.txt", "dataset_only.txt", "qcm.txt", "stepwise.txt"]
        prompts_dir = self.prompt_manager.prompts_dir
        
        results = {}
        for filename in prompt_files:
            filepath = os.path.join(prompts_dir, filename)
            exists = os.path.exists(filepath)
            results[filename] = exists
        
        return results


def get_llm_answer(
    question: str,
    question_type: str,
    dataset_info: Optional[str] = None,
    api_info: Optional[str] = None
) -> str:
    """
    Quick function to get an LLM answer
    """
    handler = LLMHandler()
    result = handler.process_question(
        question=question,
        question_type=question_type,
        dataset_info=dataset_info,
        api_info=api_info
    )
    return result["answer"]

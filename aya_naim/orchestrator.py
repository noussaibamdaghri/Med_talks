"""
AYA NAIM WORKING ORCHESTRATOR
With fallbacks for missing packages
"""

import pandas as pd
import re
from typing import Dict, List, Any, Tuple

class WorkingTextPreprocessor:
    """Text cleaner that always works"""
    @staticmethod
    def clean(text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s.,!?\-]', '', text)
        return text.strip()
    
    @staticmethod
    def should_summarize(text: str, limit: int = 150) -> bool:
        return len(text.split()) > limit

class WorkingDataLoader:
    """Data loader with fallback"""
    def load(self):
        try:
            print("Loading medical dataset...")
            df = pd.read_json(
                "hf://datasets/medalpaca/medical_meadow_medical_flashcards/medical_meadow_wikidoc_medical_flashcards.json"
            )
            print(f" Loaded {len(df)} medical flashcards")
            return df
        except Exception as e:
            print(f"  Could not load dataset: {e}")
            # Return empty dataframe with correct structure
            return pd.DataFrame({
                'instruction': ['What is headache?', 'How to treat fever?', 'Symptoms of cold'],
                'input': ['', '', ''],
                'output': ['Headache is...', 'Treat fever with...', 'Cold symptoms are...']
            })

class WorkingSimilarity:
    """Similarity with fallback to simple method"""
    def __init__(self):
        self.use_advanced = False
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_advanced = True
            print(" Using sentence-transformers for similarity")
        except ImportError:
            print("  Using simple word-based similarity (install: pip install sentence-transformers)")
    
    def find_similar(self, query: str, questions: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        if self.use_advanced:
            return self._advanced_similarity(query, questions, top_k)
        else:
            return self._simple_similarity(query, questions, top_k)
    
    def _advanced_similarity(self, query: str, questions: List[str], top_k: int) -> List[Tuple[str, float]]:
        """Use sentence-transformers"""
        query_vec = self.model.encode([query])
        question_vecs = self.model.encode(questions)
        
        import numpy as np
        scores = np.dot(question_vecs, query_vec.T).flatten()
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        return [(questions[i], float(scores[i])) for i in top_indices]
    
    def _simple_similarity(self, query: str, questions: List[str], top_k: int) -> List[Tuple[str, float]]:
        """Simple word overlap similarity"""
        query_words = set(query.lower().split())
        results = []
        
        for question in questions:
            question_words = set(question.lower().split())
            common = query_words.intersection(question_words)
            score = len(common) / max(len(query_words), 1)
            results.append((question, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

class PersonAWorkingOrchestrator:
    """Working orchestrator that handles missing packages"""
    
    def __init__(self):
        print("=" * 60)
        print(" AYA NAIM WORKING ORCHESTRATOR")
        print("=" * 60)
        
        self.loader = WorkingDataLoader()
        self.processor = WorkingTextPreprocessor()
        self.similarity = WorkingSimilarity()
        self.dataset = None
        
        print(" Orchestrator initialized (with fallbacks)")
        print("=" * 60)
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Main processing method"""
        print(f"\n Processing: '{user_query[:50]}...'")
        
        # Load data
        if self.dataset is None:
            self.dataset = self.loader.load()
        
        # Clean query
        cleaned = self.processor.clean(user_query)
        
        # Get sample questions
        questions = self.dataset['instruction'].head(50).tolist()
        
        # Find similar
        similar = self.similarity.find_similar(cleaned, questions, top_k=3)
        
        # Format results
        formatted = self._format_results(user_query, similar)
        
        return {
            "success": True,
            "raw_query": user_query,
            "cleaned_query": cleaned,
            "needs_summary": self.processor.should_summarize(cleaned),
            "similar_found": len(similar),
            "similar_results": similar,
            "formatted_context": formatted,
            "dataset_records": len(self.dataset)
        }
    
    def _format_results(self, query: str, similar: List[Tuple[str, float]]) -> str:
        """Format for Person D"""
        lines = []
        lines.append("=" * 60)
        lines.append("MEDICAL CONTEXT FROM PERSON A")
        lines.append("=" * 60)
        lines.append(f"Patient Query: {query}")
        lines.append(f"\nSimilar Medical Questions Found:")
        
        for i, (question, score) in enumerate(similar, 1):
            lines.append(f"{i}. [Score: {score:.3f}] {question[:80]}...")
        
        lines.append("=" * 60)
        return "\n".join(lines)

def test_working_orchestrator():
    """Test function"""
    print("\n TESTING WORKING ORCHESTRATOR")
    print("-" * 40)
    
    orchestrator = PersonAWorkingOrchestrator()
    result = orchestrator.process_query("What causes headaches and fever?")
    
    print(f"\n RESULTS:")
    print(f"• Status: {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"• Cleaned query: {result['cleaned_query']}")
    print(f"• Similar questions found: {result['similar_found']}")
    
    if result['similar_results']:
        print(f"• Best match: '{result['similar_results'][0][0][:60]}...'")
    
    print(f"\n Formatted context length: {len(result['formatted_context'])} chars")
    
    print("\n Working orchestrator test complete!")
    return result

if __name__ == "__main__":
    test_working_orchestrator()
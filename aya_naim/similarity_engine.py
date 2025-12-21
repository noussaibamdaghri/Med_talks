from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple

class MedicalSimilarity:
    """AYA NAIM: Find similar medical questions"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        print(f"AYA: Similarity engine loaded with {model_name}")
    
    def find_similar(self, query: str, questions: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Find questions similar to the query"""
        if not questions:
            return []
        
        # Convert to embeddings
        query_embedding = self.model.encode([query])
        question_embeddings = self.model.encode(questions)
        
        # Calculate similarities
        similarities = np.dot(question_embeddings, query_embedding.T).flatten()
        
        # Get top_k most similar
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((questions[idx], float(similarities[idx])))
        
        return results
    
    def find_similar_in_dataframe(self, query: str, df, text_column='instruction', top_k=3):
        """Find similar items in a pandas DataFrame"""
        texts = df[text_column].tolist()
        similarities = self.find_similar(query, texts, top_k)
        
        # Get full rows for matches
        results = []
        for text, score in similarities:
            matched_row = df[df[text_column] == text].iloc[0]
            results.append({
                'text': text,
                'score': score,
                'full_record': matched_row.to_dict()
            })
        
        return results
import pandas as pd
import re
from typing import Dict, List, Any, Tuple
import os
import sys

class PersonAOrchestrator:
    """
    Orchestrateur complet de Person A (Dataset)
    Version unifiée avec tous les fallbacks intégrés
    """
    
    def __init__(self):
        """Initialise l'orchestrateur avec tous les composants"""
        print("🔧 Initialisation Person A (Dataset)...")
        
        # Components
        self.loader = self._create_data_loader()
        self.processor = self._create_text_processor()
        self.similarity = self._create_similarity_engine()
        
        # Data
        self.dataset = None
        
        print("✅ Person A initialisé")
    
    def _create_data_loader(self):
        """Crée le chargeur de données avec fallback"""
        class DataLoader:
            def load(self):
                try:
                    print("   📂 Chargement dataset médical...")
                    # Essaie différentes sources
                    sources = [
                        "hf://datasets/medalpaca/medical_meadow_medical_flashcards/medical_meadow_wikidoc_medical_flashcards.json",
                        "./medical_data.json",
                        "./dataset/medical_flashcards.json"
                    ]
                    
                    for source in sources:
                        try:
                            if source.startswith("hf://"):
                                # Hugging Face dataset
                                import datasets
                                dataset = datasets.load_dataset(
                                    "medalpaca/medical_meadow_medical_flashcards",
                                    "medical_meadow_wikidoc_medical_flashcards"
                                )
                                df = dataset['train'].to_pandas()
                            else:
                                # Fichier local
                                df = pd.read_json(source)
                            
                            print(f"   ✅ Dataset chargé: {len(df)} enregistrements")
                            return df
                            
                        except Exception as e:
                            continue
                    
                    # Fallback: dataset minimal
                    print("   ⚠️  Utilisation dataset fallback")
                    return pd.DataFrame({
                        'instruction': [
                            'What is headache?',
                            'How to treat fever?', 
                            'Symptoms of common cold',
                            'Treatment for hypertension',
                            'Diagnosis of diabetes',
                            'First aid for burn',
                            'Signs of heart attack',
                            'Management of asthma'
                        ],
                        'input': [''] * 8,
                        'output': [
                            'Headache is pain in head or neck region.',
                            'Treat fever with antipyretics and fluids.',
                            'Cold symptoms: runny nose, cough, sore throat.',
                            'Hypertension treatment includes lifestyle changes and medications.',
                            'Diabetes diagnosed by blood glucose tests.',
                            'First aid for burns: cool with water, cover loosely.',
                            'Heart attack signs: chest pain, shortness of breath.',
                            'Asthma management: inhalers, avoiding triggers.'
                        ]
                    })
                    
                except Exception as e:
                    print(f"   ❌ Erreur chargement dataset: {e}")
                    # Dataset vide minimal
                    return pd.DataFrame({'instruction': [], 'input': [], 'output': []})
        
        return DataLoader()
    
    def _create_text_processor(self):
        """Crée le processeur de texte"""
        class TextProcessor:
            @staticmethod
            def clean(text: str) -> str:
                if not isinstance(text, str):
                    text = str(text)
                # Supprime les espaces multiples
                text = re.sub(r'\s+', ' ', text)
                # Garde la ponctuation de base
                text = re.sub(r'[^\w\s.,!?\-]', '', text)
                return text.strip()
            
            @staticmethod
            def should_summarize(text: str, limit: int = 150) -> bool:
                return len(text.split()) > limit
            
            @staticmethod
            def extract_keywords(text: str, max_words: int = 5) -> List[str]:
                """Extrait les mots-clés"""
                words = text.lower().split()
                # Filtre les mots communs
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                keywords = [w for w in words if w not in stop_words and len(w) > 2]
                return keywords[:max_words]
        
        return TextProcessor()
    
    def _create_similarity_engine(self):
        """Crée le moteur de similarité avec fallbacks"""
        class SimilarityEngine:
            def __init__(self):
                self.use_advanced = False
                try:
                    # Essaie sentence-transformers
                    from sentence_transformers import SentenceTransformer
                    self.model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.use_advanced = True
                    print("   🤖 Similarité avancée activée")
                except ImportError:
                    print("   ⚠️  Similarité simple (installe: pip install sentence-transformers)")
            
            def find_similar(self, query: str, questions: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
                """Trouve les questions similaires"""
                if not questions:
                    return []
                
                if self.use_advanced:
                    return self._advanced_similarity(query, questions, top_k)
                else:
                    return self._simple_similarity(query, questions, top_k)
            
            def _advanced_similarity(self, query: str, questions: List[str], top_k: int) -> List[Tuple[str, float]]:
                """Similarité avec embeddings"""
                query_vec = self.model.encode([query])
                question_vecs = self.model.encode(questions)
                
                import numpy as np
                scores = np.dot(question_vecs, query_vec.T).flatten()
                top_indices = np.argsort(scores)[-top_k:][::-1]
                
                return [(questions[i], float(scores[i])) for i in top_indices]
            
            def _simple_similarity(self, query: str, questions: List[str], top_k: int) -> List[Tuple[str, float]]:
                """Similarité par overlap de mots"""
                query_words = set(query.lower().split())
                results = []
                
                for question in questions:
                    question_words = set(question.lower().split())
                    common = query_words.intersection(question_words)
                    score = len(common) / max(len(query_words), 1)
                    results.append((question, score))
                
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:top_k]
        
        return SimilarityEngine()
    
    def search_dataset(self, user_query: str, max_results: int = 5) -> str:
        """
        Recherche principale - interface pour l'orchestrateur principal
        
        Args:
            user_query: Question de l'utilisateur
            max_results: Nombre max de résultats
            
        Returns:
            str: Contexte formaté pour Person D
        """
        print(f"   🔍 Recherche pour: '{user_query[:50]}...'")
        
        # Charge les données si nécessaire
        if self.dataset is None:
            self.dataset = self.loader.load()
        
        if self.dataset.empty:
            return "Aucune donnée médicale disponible dans le dataset."
        
        # Nettoie la requête
        cleaned_query = self.processor.clean(user_query)
        
        # Extrait les mots-clés
        keywords = self.processor.extract_keywords(cleaned_query)
        
        # Récupère les questions du dataset
        all_questions = self.dataset['instruction'].tolist()
        
        # Limite pour la performance
        sample_size = min(100, len(all_questions))
        sample_questions = all_questions[:sample_size]
        
        # Trouve les questions similaires
        similar_questions = self.similarity.find_similar(
            cleaned_query, 
            sample_questions, 
            top_k=max_results
        )
        
        # Récupère les réponses correspondantes
        context_lines = []
        
        if similar_questions:
            context_lines.append("📚 CONTEXTE MÉDICAL (Person A - Dataset):")
            context_lines.append("=" * 50)
            context_lines.append(f"Requête: {user_query}")
            context_lines.append(f"Mots-clés: {', '.join(keywords)}")
            context_lines.append("")
            context_lines.append("Questions similaires trouvées:")
            
            for i, (question, score) in enumerate(similar_questions, 1):
                # Trouve la réponse correspondante
                match = self.dataset[self.dataset['instruction'] == question]
                if not match.empty:
                    answer = match.iloc[0]['output']
                    context_lines.append(f"{i}. [{score:.2f}] {question}")
                    context_lines.append(f"   Réponse: {answer[:100]}...")
                else:
                    context_lines.append(f"{i}. [{score:.2f}] {question}")
                    context_lines.append(f"   (Pas de réponse correspondante)")
                context_lines.append("")
        
        else:
            context_lines.append("📚 CONTEXTE MÉDICAL (Person A - Dataset):")
            context_lines.append("=" * 50)
            context_lines.append(f"Aucune correspondance exacte pour: {user_query}")
            context_lines.append("")
            context_lines.append("Informations générales médicales:")
            
            # Retourne quelques entrées générales
            for i in range(min(3, len(self.dataset))):
                question = self.dataset.iloc[i]['instruction']
                answer = self.dataset.iloc[i]['output']
                context_lines.append(f"{i+1}. {question}")
                context_lines.append(f"   {answer[:80]}...")
                context_lines.append("")
        
        context_lines.append("=" * 50)
        
        return "\n".join(context_lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques"""
        return {
            "dataset_loaded": self.dataset is not None,
            "dataset_size": len(self.dataset) if self.dataset is not None else 0,
            "similarity_mode": "advanced" if self.similarity.use_advanced else "simple",
            "components": ["loader", "processor", "similarity"]
        }


# Fonction d'interface simple pour l'orchestrateur principal
def create_person_a_orchestrator():
    """Factory function pour créer Person A"""
    return PersonAOrchestrator()


# Test rapide si exécuté directement
if __name__ == "__main__":
    print("🧪 Test de PersonAOrchestrator")
    print("=" * 60)
    
    orchestrator = PersonAOrchestrator()
    
    test_queries = [
        "What causes headache?",
        "How to treat high blood pressure?",
        "Symptoms of diabetes"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Test: {query}")
        context = orchestrator.search_dataset(query)
        print(f"📝 Contexte généré ({len(context)} caractères):")
        print(context[:200] + "..." if len(context) > 200 else context)
        print("-" * 40)
    
    stats = orchestrator.get_stats()
    print(f"\n📊 Statistiques: {stats}")

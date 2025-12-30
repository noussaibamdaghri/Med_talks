"""
Orchestrateur pour Person C - Point d'entrée unique pour Person D
"""
import logging
from typing import List, Optional
from noussaiba_mdaghri.api_client import HTTPClient, http_client
from wikipedia import wikipedia_client
from openfda import openfda_client
from models import APIResult, APIResponse

logger = logging.getLogger(__name__)

class PersonCOrchestrator:
    """
    Orchestrateur principal - Interface unique pour Person D
    
    Usage:
        orchestrator = PersonCOrchestrator()
        results = orchestrator.search_apis("aspirin")
    """
    
    def __init__(self):
        """Initialise tous les clients API"""
        self.http_client = http_client
        self.wikipedia = wikipedia_client
        self.openfda = openfda_client
        
        logger.info(" PersonCOrchestrator initialisé")
    
    def search_apis(self, question: str, max_results: int = 3) -> APIResponse:
        """
        Point d'entrée principal - Recherche sur toutes les APIs
        
        Args:
            question: Question de l'utilisateur
            max_results: Nombre max de résultats par source
            
        Returns:
            APIResponse avec tous les résultats formatés
        """
        logger.info(f" Recherche orchestrée pour: '{question}'")
        
        response = APIResponse(question)
        
        # 1. Toujours chercher sur Wikipedia (définitions générales)
        logger.info(" Interrogation Wikipedia...")
        try:
            wiki_results = self.wikipedia.search(question, max_results=max_results)
            for result in wiki_results:
                response.add_result(result)
            logger.info(f"  Wikipedia: {len(wiki_results)} résultats")
        except Exception as e:
            error_msg = f"Wikipedia error: {str(e)}"
            logger.error(f"   {error_msg}")
            response.add_error(error_msg)
        
        # 2. Chercher sur OpenFDA si question médicale
        if self._looks_medical(question):
            logger.info(" Interrogation OpenFDA (question médicale)...")
            try:
                # Médicaments
                drug_results = self.openfda.search_drugs(question, max_results=max_results)
                for result in drug_results:
                    response.add_result(result)
                
                # Effets secondaires (limités à 1 pour performance)
                if drug_results:  # Seulement si on a trouvé des médicaments
                    adverse_results = self.openfda.search_adverse_effects(question, max_results=1)
                    for result in adverse_results:
                        response.add_result(result)
                
                logger.info(f"  OpenFDA: {len(drug_results) + len(adverse_results)} résultats")
            except Exception as e:
                error_msg = f"OpenFDA error: {str(e)}"
                logger.error(f"   {error_msg}")
                response.add_error(error_msg)
        
        logger.info(f" Recherche terminée: {len(response.results)} résultats totaux")
        return response
    
    def search_specific(self, question: str, source: str = "all", max_results: int = 3) -> APIResponse:
        """
        Recherche sur une source spécifique
        
        Args:
            question: Question utilisateur
            source: 'wikipedia', 'openfda', ou 'all'
            max_results: Nombre max de résultats
            
        Returns:
            APIResponse avec résultats
        """
        logger.info(f" Recherche spécifique [{source}] pour: '{question}'")
        
        response = APIResponse(question)
        
        if source in ["wikipedia", "all"]:
            try:
                results = self.wikipedia.search(question, max_results=max_results)
                for result in results:
                    response.add_result(result)
            except Exception as e:
                response.add_error(f"Wikipedia error: {str(e)}")
        
        if source in ["openfda", "all"]:
            try:
                # Médicaments
                drug_results = self.openfda.search_drugs(question, max_results=max_results)
                for result in drug_results:
                    response.add_result(result)
                
                # Effets secondaires
                adverse_results = self.openfda.search_adverse_effects(question, max_results=1)
                for result in adverse_results:
                    response.add_result(result)
            except Exception as e:
                response.add_error(f"OpenFDA error: {str(e)}")
        
        return response
    
    def format_for_llm(self, question: str, max_results_per_source: int = 2) -> str:
        """
        Formatte directement pour le LLM (utilitaire pour Person D)
        
        Returns:
            Texte formaté prêt pour le prompt
        """
        response = self.search_apis(question)
        
        if not response.results:
            return f"Aucune information externe trouvée pour: {question}"
        
        # Formate joliment
        formatted = f"INFORMATIONS EXTERNES POUR: '{question}'\n\n"
        
        # Groupe par source
        by_source = {}
        for result in response.results:
            if result.source not in by_source:
                by_source[result.source] = []
            by_source[result.source].append(result)
        
        # Format avec limite
        for source, results in by_source.items():
            formatted += f"=== {source.upper()} ===\n"
            for i, result in enumerate(results[:max_results_per_source], 1):
                formatted += f"{i}. {result.title}\n"
                
                # Nettoie le contenu pour le LLM
                clean_content = result.content.replace('\n\n', '\n').strip()
                if len(clean_content) > 200:  # Limite la longueur
                    clean_content = clean_content[:200] + "..."
                
                formatted += f"   {clean_content}\n\n"
        
        return formatted
    
    def clear_cache(self):
        """Vide le cache HTTP"""
        self.http_client.clear_cache()
        logger.info("🧹 Cache HTTP vidé")
    
    def _looks_medical(self, question: str) -> bool:
        """
        Détecte si une question semble médicale
        (Simple heuristique pour Person D)
        """
        medical_keywords = [
            'médicament', 'medicament', 'pillule', 'comprimé', 'traitement',
            'symptôme', 'symptome', 'maladie', 'diagnostic', 'effet secondaire',
            'dose', 'posologie', 'contre-indication', 'interaction',
            'aspirin', 'aspirine', 'paracetamol', 'ibuprofène', 'ibuprofene',
            'vaccin', 'cancer', 'diabète', 'diabete', 'cardiaque', 'foie',
            'rein', 'poumon', 'cerveau', 'sang', 'tension', 'cholestérol',
            'allergie', 'infection', 'virus', 'bactérie', 'bacterie'
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in medical_keywords)

# Instance globale pour import facile
orchestrator = PersonCOrchestrator()

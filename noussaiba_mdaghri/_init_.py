# __init__.py - Version simplifiée
from .api_client import http_client, clear_cache
from .parsers import cleaner
from .models import APIResult, APIResponse
from .wikipedia import wikipedia_client
from .openfda import openfda_client
from .orchestrator import PersonCOrchestrator, orchestrator

class APIManager:
    """Gestionnaire principal pour Person D et B"""
    
    def __init__(self):
        self.clients = {
            "wikipedia": wikipedia_client,
            "openfda": openfda_client
        }
    
    def search(self, query: str, sources: list = None) -> APIResponse:
        """Méthode principale pour les autres personnes"""
        from .models import APIResponse
        
        response = APIResponse(query)
        
        if sources is None:
            sources = ["wikipedia", "openfda"]
        
        for source in sources:
            if source == "wikipedia":
                try:
                    results = self.clients[source].search(query)
                    for result in results:
                        response.add_result(result)
                except Exception as e:
                    response.add_error(f"Wikipedia error: {str(e)}")
            
            elif source == "openfda":
                try:
                    # Médicaments
                    drug_results = self.clients[source].search_drugs(query)
                    for result in drug_results:
                        response.add_result(result)
                    
                    # Effets secondaires
                    adverse_results = self.clients[source].search_adverse_effects(query)
                    for result in adverse_results:
                        response.add_result(result)
                except Exception as e:
                    response.add_error(f"OpenFDA error: {str(e)}")
        
        return response

# Instance globale que les autres utiliseront
api_manager = APIManager()

# Exporte ce qui est important
__all__ = ['api_manager', 'APIResult', 'APIResponse', 'cleaner', 'http_client','orchestrator','clear_cache']


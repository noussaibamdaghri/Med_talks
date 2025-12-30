"""
Module HTTP avec cache et système de réessais
C'est ton moteur pour toutes les communications internet
"""
import time
import logging
from typing import Optional, Dict, Any
import requests
import requests_cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import RequestException, Timeout, ConnectionError
from http.client import IncompleteRead
# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HTTPClient:
    """
    CLIENT HTTP INTELLIGENT
    Fait 3 choses importantes :
    1. Cache les réponses (pour éviter de redemander la même chose)
    2. Réessaie automatiquement si ça échoue
    3. Gère les timeouts
    """
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        """
        Initialise le client HTTP
        
        Args:
            timeout: Temps max d'attente (secondes)
            max_retries: Nombre de réessais en cas d'échec
        """
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Crée une session avec cache
        self.session = requests.Session()
        
        # Configure le cache SQLite (fichier api_cache.sqlite)
        requests_cache.install_cache(
            'api_cache',
            expire_after=3600,  # Cache valide 1 heure
            allowable_methods=['GET', 'POST']
        )
        
        # Headers pour paraître comme un vrai navigateur
        self.session.headers.update({
            'User-Agent': 'MedicalChatbot/1.0 (contact: ton.email@exemple.com)',
            'Accept': 'application/json',
            'Accept-Language': 'fr,en;q=0.9',
            'Cache-Control': 'max-age=0'
        })
    
    @retry(
        stop=stop_after_attempt(3),  # Réessaie 3 fois max
        wait=wait_exponential(multiplier=1, min=2, max=10),  # Attente exponentielle
        retry=retry_if_exception_type((Timeout, ConnectionError)),  # Réessaie seulement sur timeout/connexion
        reraise=True  # Relève l'exception après les réessais
    )
    def get(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Fait une requête GET sécurisée
        
        Args:
            url: L'URL de l'API
            params: Paramètres de recherche (ex: {'q': 'aspirin'})
            headers: Headers spécifiques
            
        Returns:
            Les données JSON de la réponse
            
        Exemple:
            client.get("https://api.fda.gov/drug/label.json", params={'search': 'aspirin'})
        """
        try:
            start_time = time.time()
            
            # Combine les headers par défaut avec les headers spécifiques
            all_headers = self.session.headers.copy()
            if headers:
                all_headers.update(headers)
            
            logger.info(f" GET {url}")
            if params:
                logger.info(f"   Paramètres: {params}")
            
            # Fait la requête
            response = self.session.get(
                url,
                params=params,
                headers=all_headers,
                timeout=self.timeout
            )
            
            # Calcule le temps pris
            elapsed = time.time() - start_time
            
            # Vérifie le statut HTTP
            response.raise_for_status()  # Lève une exception si erreur (400-599)
            
            # Log si c'était du cache ou pas
            if getattr(response, 'from_cache', False):
                logger.info(f" Cache hit ({elapsed:.2f}s)")
            else:
                logger.info(f" API call ({elapsed:.2f}s) - Status: {response.status_code}")
            
            # Essaie de parser en JSON
            try:
                return response.json()
            except ValueError:
                # Si pas du JSON, retourne le texte
                logger.warning(f"  Réponse non-JSON de {url}")
                return {"text": response.text}
                
        except RequestException as e:
            logger.error(f" Erreur HTTP pour {url}: {str(e)}")
            raise
    
    def clear_cache(self):
        """Vide le cache manuellement (utile pour les tests)"""
        requests_cache.clear()
        logger.info("🧹 Cache vidé")

# Instance globale utilisée dans tout ton module
http_client = HTTPClient()


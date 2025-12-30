"""
Client pour l'API Wikipedia (française)
Récupère des définitions et informations générales
"""
import logging
from typing import List, Optional, Dict, Any
from api_client import http_client
from parsers import cleaner
from models import APIResponse,APIResult

logger = logging.getLogger(__name__)

class WikipediaClient:
    """
    CLIENT WIKIPEDIA
    Cherche et récupère des pages Wikipedia
    """
    
    BASE_URL = "https://fr.wikipedia.org/w/api.php"
    
    def search(self, query: str, max_results: int = 3) -> List[APIResult]:
        """
        Cherche des pages Wikipedia correspondant à la requête
        
        Args:
            query: Ce que l'utilisateur cherche
            max_results: Nombre max de résultats
            
        Returns:
            Liste de pages Wikipedia
            
        Exemple:
            client.search("aspirine") → [Page sur l'aspirine]
        """
        try:
            logger.info(f" Wikipedia search: '{query}'")
            
            # Paramètres de l'API Wikipedia
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': max_results,
                'srprop': 'snippet',
                'utf8': 1,
                'srinterwiki': 1
            }
            
            # Appel à l'API
            data = http_client.get(self.BASE_URL, params=params)
            
            # Vérifie s'il y a des résultats
            if 'query' not in data or 'search' not in data['query']:
                logger.warning(f"  Aucun résultat Wikipedia pour: {query}")
                return []
            
            pages = []
            search_results = data['query']['search']
            
            logger.info(f" Wikipedia a trouvé {len(search_results)} résultats")
            
            # Parse chaque résultat
            for item in search_results:
                page = self._parse_search_result(item)
                if page:
                    pages.append(page)
                    logger.debug(f"  ✓ {page.title}")
            
            return pages
            
        except Exception as e:
            logger.error(f" Erreur Wikipedia search: {str(e)}")
            return []
    
    def get_page_content(self, page_id: int) -> Optional[APIResult]:
        """
        Récupère le contenu complet d'une page Wikipedia
        
        Args:
            page_id: ID de la page Wikipedia
            
        Returns:
            La page complète ou None si erreur
        """
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'pageids': page_id,
                'prop': 'extracts|info',
                'inprop': 'url',
                'exintro': True,  # Seulement l'intro
                'explaintext': True,  # Texte seulement, pas HTML
                'exchars': 1000,  # ~1000 caractères max
                'utf8': 1
            }
            
            data = http_client.get(self.BASE_URL, params=params)
            
            if 'query' not in data or 'pages' not in data['query']:
                return None
            
            page_data = data['query']['pages'].get(str(page_id))
            if not page_data:
                return None
            
            return self._parse_full_page(page_data)
            
        except Exception as e:
            logger.error(f" Erreur get_page_content: {str(e)}")
            return None
    
    # Dans wikipedia.py, change _parse_search_result :

    def _parse_search_result(self, item: Dict[str, Any]) -> Optional[APIResult]:
     """Transforme un résultat de recherche en APIResult"""
     try:
         page_id = item.get('pageid')
         title = item.get('title', '')
         snippet = item.get('snippet', '')
         
         if not title:
            return None
        
        # Nettoie le snippet
         clean_snippet = cleaner.clean_html(snippet, max_length=500)
         clean_snippet = cleaner.remove_citations(clean_snippet)
        
        # URL vers la page
         url = f"https://fr.wikipedia.org/?curid={page_id}" if page_id else None
        
         return APIResult(
            source="wikipedia",  # ← Spécifie la source ici
            title=title,
            content=clean_snippet,
            url=url,
            confidence=0.8,
            metadata={"page_id": page_id}  # ← page_id dans metadata
         )
        
     except Exception as e:
         logger.error(f" Erreur parsing Wikipedia: {str(e)}")
         return None
    
    def _parse_full_page(self, page_data: Dict[str, Any]) -> Optional[APIResult]:
        """
        Parse une page Wikipedia complète
        """
        try:
            title = page_data.get('title', '')
            extract = page_data.get('extract', '')
            page_id = page_data.get('pageid')
            fullurl = page_data.get('fullurl')
            
            if not extract:
                return None
            
            # Nettoie et limite le contenu
            clean_content = cleaner.clean_html(extract, max_length=1500)
            clean_content = cleaner.remove_citations(clean_content)
            
            # Si c'est trop long, fait un résumé
            if len(clean_content) > 800:
                clean_content = cleaner.extract_summary(clean_content)
            
            return APIResult(
                title=title,
                content=clean_content,
                url=fullurl,
                page_id=page_id,
                confidence=0.9  # Plus de contenu = plus fiable
            )
            
        except Exception as e:
            logger.error(f" Erreur parsing page Wikipedia: {str(e)}")
            return None

# Instance globale utilisée partout
wikipedia_client = WikipediaClient()


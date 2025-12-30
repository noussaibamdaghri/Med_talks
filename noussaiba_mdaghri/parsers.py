"""
Nettoyeur de données pour les réponses API
Transforme le HTML/markup en texte propre
"""
import re
import html
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class DataCleaner:
    """
    NETTOYEUR DE DONNÉES
    Transforme les données brutes des APIs en texte propre
    """
    
    @staticmethod
    def clean_html(raw_html: str, max_length: int = 1000) -> str:
        """
        Nettoie le HTML pour garder seulement le texte utile
        
        Args:
            raw_html: Texte HTML brut
            max_length: Longueur maximale du résultat
            
        Returns:
            Texte nettoyé sans HTML
            
        Exemple:
            clean_html("<p>Hello <b>world</b></p>") → "Hello world"
        """
        if not raw_html or not isinstance(raw_html, str):
            return ""
        
        try:
            # 1. Décode les entités HTML (&amp; → &, etc.)
            decoded = html.unescape(raw_html)
            
            # 2. Parse avec BeautifulSoup
            soup = BeautifulSoup(decoded, 'lxml')
            
            # 3. Supprime les éléments inutiles
            for element in soup(['script', 'style', 'nav', 'footer', 
                                'header', 'aside', 'iframe', 'noscript',
                                'sup']):  # 'sup' pour les références [1]
                element.decompose()  # Supprime complètement
            
            # 4. Récupère le texte avec des sauts de ligne
            text = soup.get_text(separator='\n')
            
            # 5. Nettoie les espaces multiples et lignes vides
            lines = []
            for line in text.splitlines():
                line = line.strip()
                if line and len(line) > 2:  # Ignore les lignes trop courtes
                    lines.append(line)
            
            clean_text = '\n'.join(lines)
            
            # 6. Tronque si trop long
            if len(clean_text) > max_length:
                clean_text = clean_text[:max_length] + "..."
            
            return clean_text
            
        except Exception as e:
            logger.error(f" Erreur nettoyage HTML: {str(e)}")
            return raw_html[:max_length] if raw_html else ""
    
    @staticmethod
    def extract_summary(text: str, max_sentences: int = 3) -> str:
        """
        Extrait un résumé concis d'un texte
        
        Args:
            text: Texte complet
            max_sentences: Nombre de phrases à garder
            
        Returns:
            Résumé concis
        """
        if not text:
            return ""
        
        # Divise en phrases (simplifié)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Garde les premières phrases
        summary = '. '.join(sentences[:max_sentences])
        if summary and not summary.endswith('.'):
            summary += '.'
        
        return summary
    
    @staticmethod
    def remove_citations(text: str) -> str:
        """
        Supprime les citations [1], [2], etc.
        """
        return re.sub(r'\[\d+\]', '', text)
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Normalise les espaces (supprime les multiples espaces)
        """
        return re.sub(r'\s+', ' ', text).strip()

# Instance globale
cleaner = DataCleaner()


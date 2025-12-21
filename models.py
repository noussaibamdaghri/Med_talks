# models.py - Version corrigée
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

@dataclass
class APIResult:
    """Résultat générique d'API"""
    source: str
    title: str
    content: str
    url: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "source": self.source,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "confidence": self.confidence,
            "metadata": self.metadata
        }

class APIResponse:
    """Conteneur de réponse"""
    
    def __init__(self, query: str):
        self.query = query
        self.results: List[APIResult] = []
        self.timestamp = datetime.now().isoformat()
        self.success = True
        self.errors: List[str] = []
    
    def add_result(self, result: APIResult):
        self.results.append(result)
    
    def add_error(self, error: str):
        self.errors.append(error)
        self.success = False
    
    def __len__(self):
        return len(self.results)
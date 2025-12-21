import re
import unicodedata

class TextPreprocessor:
    """AYA NAIM: Clean medical text"""
    
    def clean(self, text):
        """Basic text cleaning"""
        if not isinstance(text, str):
            text = str(text)
        
        # 1. Unicode fix
        text = unicodedata.normalize('NFKC', text)
        # 2. Fix whitespace
        text = re.sub(r'\s+', ' ', text)
        # 3. Remove weird chars
        text = re.sub(r'[^\w\s.,!?\-:]', '', text)
        return text.strip()
    
    def should_summarize(self, text, word_limit=150):
        """Check if text is too long"""
        return len(text.split()) > word_limit
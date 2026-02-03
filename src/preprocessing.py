"""
Preprocessing module using hazm library for Persian text.
- Normalizer: standardize characters
- Stop-word removal
- Lemmatizer: convert words to main form (e.g., "نوشت" -> "نوشتن")
"""

from hazm import Normalizer, stopwords_list, Lemmatizer
from typing import List
import re


class PersianTextPreprocessor:
    """Preprocess Persian text using hazm."""
    
    def __init__(self):
        self.normalizer = Normalizer()
        self.lemmatizer = Lemmatizer()
        self.stop_words = set(stopwords_list())
    
    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline: normalize -> tokenize -> remove stopwords -> lemmatize.
        
        Args:
            text: Raw Persian text
            
        Returns:
            Preprocessed text (space-joined tokens)
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 1. Normalize: standardize characters (e.g., Arabic/Persian variants)
        normalized = self.normalizer.normalize(text)
        
        # 2. Tokenize (simple word split for Persian)
        tokens = self._tokenize(normalized)
        
        # 3. Remove stop-words
        tokens = [t for t in tokens if t not in self.stop_words]
        
        # 4. Lemmatize
        lemmatized = []
        for token in tokens:
            lemma = self.lemmatizer.lemmatize(token)
            lemmatized.append(lemma if lemma else token)
        
        return " ".join(lemmatized)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize Persian text (keep only word characters)."""
        # Remove extra whitespace and split
        text = re.sub(r"\s+", " ", text.strip())
        # Extract Persian/Arabic/English word tokens
        tokens = re.findall(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFFa-zA-Z0-9]+", text)
        return tokens
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a list of texts."""
        return [self.preprocess(t) for t in texts]

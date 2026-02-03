"""
State-of-the-Art model: ParsBERT embeddings + Logistic Regression.
Uses [CLS] token representation as the text embedding.
"""

import numpy as np
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel
import torch


class ParsBERTClassifier:
    """ParsBERT [CLS] embeddings + Logistic Regression for Persian news classification."""
    
    def __init__(
        self,
        model_name: str = "HooshvareLab/bert-base-parsbert-uncased",
        max_length: int = 256,
        batch_size: int = 16,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.bert_model.to(self.device)
        self.bert_model.eval()
        
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(max_iter=1000, random_state=42)
        self.classes_ = None
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract [CLS] token embeddings from ParsBERT."""
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            with torch.no_grad():
                outputs = self.bert_model(**encoded)
                # [CLS] token is the first token (index 0)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_embeddings)
        
        return np.vstack(all_embeddings)
    
    def fit(self, X: List[str], y: np.ndarray) -> "ParsBERTClassifier":
        """Fit Logistic Regression on ParsBERT embeddings."""
        embeddings = self._get_embeddings(X)
        self.lr.fit(embeddings, y)
        self.classes_ = self.lr.classes_
        return self
    
    def predict(self, X: List[str]) -> np.ndarray:
        """Predict labels for texts."""
        embeddings = self._get_embeddings(X)
        return self.lr.predict(embeddings)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Predict class probabilities."""
        embeddings = self._get_embeddings(X)
        return self.lr.predict_proba(embeddings)

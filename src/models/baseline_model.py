"""
Baseline model: TF-IDF + SVM with linear kernel.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np
from typing import List, Tuple


class BaselineClassifier:
    """TF-IDF + Linear SVM for Persian news classification."""
    
    def __init__(self, max_features: int = 10000, max_df: float = 0.95, min_df: int = 2):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                max_df=max_df,
                min_df=min_df,
                ngram_range=(1, 2),
                sublinear_tf=True,
            )),
            ("svm", SVC(kernel="linear", C=1.0, random_state=42)),
        ])
        self.classes_ = None
    
    def fit(self, X: List[str], y: np.ndarray) -> "BaselineClassifier":
        """Fit the model on preprocessed texts."""
        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.classes_
        return self
    
    def predict(self, X: List[str]) -> np.ndarray:
        """Predict labels for texts."""
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Predict probabilities (SVM with linear kernel uses decision function)."""
        # SVC doesn't have predict_proba by default for large datasets;
        # enable probability=True in SVC if needed. For now we use decision_function
        return self.pipeline.decision_function(X)

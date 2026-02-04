#!/usr/bin/env python3
"""
Part One: Persian News Text Classification

Pipeline:
1. Load data (body -> text, category from tags/category)
2. Preprocess with hazm (Normalizer, stopwords, Lemmatizer)
3. Train/Test split 80/20
4. Baseline: TF-IDF + SVM (linear kernel)
5. SOTA: ParsBERT [CLS] + Logistic Regression
6. Evaluate: Accuracy, F1-Score, Confusion Matrix

Usage:
    python train_classifier.py --data path/to/dataset.csv
    python train_classifier.py --data path/to/dataset_folder/

Dataset: https://www.kaggle.com/datasets/amirzenoozi/persian-news-dataset
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data_loader import load_persian_news_dataset
from src.preprocessing import PersianTextPreprocessor
from src.models.baseline_model import BaselineClassifier
from src.models.sota_model import ParsBERTClassifier
from src.evaluation import evaluate_model, plot_confusion_matrix, print_evaluation_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Persian News Classification - Part 1"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset CSV or folder containing CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for saving confusion matrices and reports",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--skip-sota",
        action="store_true",
        help="Skip ParsBERT model (faster, for testing)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("Loading Persian News Dataset...")
    df = load_persian_news_dataset(args.data)
    print(f"Loaded {len(df)} samples")
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    class_names = list(le.classes_)
    print(f"Classes: {class_names}")
    
    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        df["body"].tolist(),
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=None if len(class_names) < 2 else y,
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Preprocessing with hazm
    print("\nPreprocessing with hazm (Normalizer, stopwords, Lemmatizer)...")
    preprocessor = PersianTextPreprocessor()
    X_train_preprocessed = preprocessor.preprocess_batch(X_train)
    X_test_preprocessed = preprocessor.preprocess_batch(X_test)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Baseline: TF-IDF + SVM ---
    print("\n" + "="*50)
    print("Training Baseline Model (TF-IDF + SVM linear kernel)")
    print("="*50)
    baseline = BaselineClassifier()
    baseline.fit(X_train_preprocessed, y_train)
    y_pred_baseline = baseline.predict(X_test_preprocessed)
    
    metrics_baseline = evaluate_model(y_test, y_pred_baseline)
    print_evaluation_report(metrics_baseline, "Baseline (TF-IDF + SVM)")
    
    plot_confusion_matrix(
        y_test,
        y_pred_baseline,
        class_names,
        title="Baseline (TF-IDF + SVM) - Confusion Matrix",
        save_path=str(output_dir / "confusion_matrix_baseline.png"),
    )
    
    # --- SOTA: ParsBERT + Logistic Regression ---
    if not args.skip_sota:
        print("\n" + "="*50)
        print("Training SOTA Model (ParsBERT + Logistic Regression)")
        print("="*50)
        # Use raw text for BERT (no hazm preprocessing - BERT tokenizer handles it)
        sota = ParsBERTClassifier()
        sota.fit(X_train, y_train)  # Raw text for BERT
        y_pred_sota = sota.predict(X_test)
        
        metrics_sota = evaluate_model(y_test, y_pred_sota)
        print_evaluation_report(metrics_sota, "SOTA (ParsBERT + LR)")
        
        plot_confusion_matrix(
            y_test,
            y_pred_sota,
            class_names,
            title="SOTA (ParsBERT + Logistic Regression) - Confusion Matrix",
            save_path=str(output_dir / "confusion_matrix_sota.png"),
        )
    
    print("\nDone.")


if __name__ == "__main__":
    main()

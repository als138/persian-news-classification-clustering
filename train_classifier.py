import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


RANDOM_STATE = 42


def load_dataset(data_dir: Path):
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("❌ No CSV file found in data directory")

    df = pd.read_csv(csv_files[0])

    # ===============================
    # Explicit column mapping (FIXED)
    # ===============================
    label_col = "subgroup"   # or "service" for higher-level classes
    text_col = "body"        # main news text

    if label_col not in df.columns:
        raise ValueError(f"❌ Label column '{label_col}' not found in CSV")

    if text_col not in df.columns:
        raise ValueError(f"❌ Text column '{text_col}' not found in CSV")

    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]

    return df


def clean_rare_classes(df: pd.DataFrame, min_samples: int = 2):
    counts = Counter(df["label"])
    valid_labels = {c for c, n in counts.items() if n >= min_samples}

    removed = set(counts.keys()) - valid_labels
    if removed:
        print(f"⚠️ Removing rare classes (<{min_samples} samples): {removed}")

    df = df[df["label"].isin(valid_labels)].copy()
    return df


def can_stratify(y):
    return min(Counter(y).values()) >= 2


def plot_confusion(y_true, y_pred, labels, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_baseline(df: pd.DataFrame, output_dir: Path):
    X = df["text"].values
    y = df["label"].values

    stratify = y if can_stratify(y) else None
    if stratify is None:
        print("⚠️ Stratified split disabled (some classes have < 2 samples)")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify
    )

    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=2
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LinearSVC()
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print("\n📊 Classification Report:\n")
    print(report)

    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "classification_report.txt").write_text(report)

    labels = sorted(set(y))
    plot_confusion(
        y_test,
        y_pred,
        labels,
        output_dir / "confusion_matrix.png"
    )

    print(f"✅ Baseline results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--skip-sota", action="store_true", help="Skip ParsBERT model")

    args = parser.parse_args()

    data_dir = Path(args.data)
    output_dir = Path(args.output_dir)

    print("📥 Loading dataset...")
    df = load_dataset(data_dir)

    print(f"Initial dataset size: {len(df)}")

    df = clean_rare_classes(df, min_samples=2)

    print(f"Dataset size after cleaning: {len(df)}")
    print("Class distribution:")
    print(df["label"].value_counts())

    print("\n🚀 Running baseline TF-IDF + LinearSVC model...")
    run_baseline(df, output_dir)

    if not args.skip_sota:
        print("\nℹ️ SOTA (ParsBERT) part is not implemented in this script.")
        print("ℹ️ Use --skip-sota for now or integrate SOTA separately.")


if __name__ == "__main__":
    main()

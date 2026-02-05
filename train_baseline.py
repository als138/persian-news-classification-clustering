# train_baseline.py
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================
# Utils
# =====================================================
def load_csv(data_dir: Path):
    csvs = list(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError("❌ No CSV file found in data directory")
    return pd.read_csv(csvs[0])


def keep_top_k_classes(df, label_col="label", k=20, other_label="other"):
    counts = df[label_col].value_counts()
    top_k = counts.head(k).index.tolist()
    df = df.copy()
    df[label_col] = df[label_col].apply(
        lambda x: x if x in top_k else other_label
    )
    return df


def plot_confusion(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="data directory with CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Loading dataset...")
    df = load_csv(Path(args.data))

    # Explicit column mapping for Persian News dataset
    df = df[["body", "subgroup"]].dropna()
    df.columns = ["text", "label"]

    print(f"Initial samples: {len(df)} | labels: {df['label'].nunique()}")

    # Option B: Top-K + other
    df = keep_top_k_classes(df, label_col="label", k=args.top_k, other_label="other")
    print("After Top-K filtering:")
    print(df["label"].value_counts())

    # Encode labels
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])

    X = df["text"].values
    y = df["label_id"].values

    counts = Counter(y)
    stratify = y if min(counts.values()) >= 2 else None
    if stratify is None:
        print("⚠️ Stratified split disabled (some classes have <2 samples)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    print("🟦 Running Baseline: TF-IDF + LinearSVC")

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LinearSVC()
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    labels = list(le.classes_)
    report = classification_report(
        y_test, y_pred, target_names=labels, zero_division=0
    )

    print(report)
    (out_dir / "classification_report.txt").write_text(report)

    plot_confusion(
        y_test, y_pred, labels,
        out_dir / "confusion_matrix.png"
    )

    print("✅ Baseline results saved to:", out_dir)


if __name__ == "__main__":
    main()

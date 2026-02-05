import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

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
    """
    Keep only top-k most frequent classes.
    All remaining classes are mapped to `other_label`.
    """
    counts = df[label_col].value_counts()
    top_k_labels = counts.head(k).index.tolist()

    df = df.copy()
    df[label_col] = df[label_col].apply(
        lambda x: x if x in top_k_labels else other_label
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
# Baseline: TF-IDF + LinearSVC (CPU)
# =====================================================
def run_baseline(df, output_dir: Path):
    print("\n🟦 Running BASELINE (TF-IDF + LinearSVC)")
    out_dir = output_dir / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    X = df["text"].values
    y = df["label_id"].values

    counts = Counter(y)
    stratify = y if min(counts.values()) >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

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

    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)
    (out_dir / "classification_report.txt").write_text(report)

    labels = list(df["label"].unique())
    plot_confusion(
        y_test, y_pred, labels,
        out_dir / "confusion_matrix.png"
    )

    print("✅ Baseline results saved to:", out_dir)


# =====================================================
# SOTA: ParsBERT (GPU)
# =====================================================
def run_sota(df, output_dir: Path):
    print("\n🔥 Running SOTA (ParsBERT + GPU)")
    out_dir = output_dir / "sota"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    X = df["text"].tolist()
    y = df["label_id"].tolist()

    counts = Counter(y)
    stratify = y if min(counts.values()) >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    model_name = "HooshvareLab/bert-base-parsbert-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(texts):
        return tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=384
        )

    train_enc = tokenize(X_train)
    test_enc = tokenize(X_test)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, enc, labels):
            self.enc = enc
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_ds = Dataset(train_enc, y_train)
    test_ds = Dataset(test_enc, y_test)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(df["label"].unique())
    ).to(device)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,   # effective batch = 16
        num_train_epochs=5,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro")
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    preds = trainer.predict(test_ds)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)

    labels = list(df["label"].unique())
    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    print(report)
    (out_dir / "classification_report.txt").write_text(report)

    plot_confusion(
        y_true, y_pred, labels,
        out_dir / "confusion_matrix.png"
    )

    trainer.save_model(out_dir / "model")
    tokenizer.save_pretrained(out_dir / "model")

    print("✅ SOTA results saved to:", out_dir)


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--run-baseline", action="store_true")
    parser.add_argument("--run-sota", action="store_true")
    args = parser.parse_args()

    # default: run both
    run_baseline_flag = args.run_baseline or (not args.run_baseline and not args.run_sota)
    run_sota_flag = args.run_sota or (not args.run_baseline and not args.run_sota)

    df = load_csv(Path(args.data))

    # Explicit column mapping
    df = df[["body", "subgroup"]].dropna()
    df.columns = ["text", "label"]

    print("Original label count:", df["label"].nunique())

    # =========================
    # OPTION B: Top-K classes
    # =========================
    df = keep_top_k_classes(df, label_col="label", k=args.top_k, other_label="other")

    print("\nAfter Top-K filtering:")
    print(df["label"].value_counts())

    # Encode labels
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_baseline_flag:
        run_baseline(df, output_dir)

    if run_sota_flag:
        run_sota(df, output_dir)


if __name__ == "__main__":
    main()

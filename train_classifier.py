# train_classifier.py
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

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

MODEL_NAME = "HooshvareLab/bert-base-parsbert-uncased"

def load_csv(data_dir: Path):
    csvs = list(Path(data_dir).glob("*.csv"))
    if not csvs:
        raise FileNotFoundError("No CSV in data dir")
    return pd.read_csv(csvs[0])

def keep_top_k_classes(df, label_col="label", k=20, other_label="other"):
    counts = df[label_col].value_counts()
    top_k = counts.head(k).index.tolist()
    df = df.copy()
    df[label_col] = df[label_col].apply(lambda x: x if x in top_k else other_label)
    return df

def plot_confusion(y_true, y_pred, labels, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro")
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="data dir with CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20, help="keep top-k labels (others -> other)")
    parser.add_argument("--max-length", type=int, default=256, help="tokenizer max length")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per-device-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--do-sample", type=int, default=0, help="if >0 sample this many examples for quick test")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(Path(args.data))

    # explicit mapping
    df = df[["body", "subgroup"]].dropna().rename(columns={"body":"text", "subgroup":"label"})
    print("Original samples:", len(df), "unique labels:", df["label"].nunique())

    # Top-K mapping
    df = keep_top_k_classes(df, label_col="label", k=args.top_k, other_label="other")
    print("After Top-K:", df["label"].nunique())
    print(df["label"].value_counts().head(args.top_k + 2))

    # optional sampling for quick tests
    if args.do_sample and args.do_sample > 0:
        df = df.sample(min(args.do_sample, len(df)), random_state=42).reset_index(drop=True)
        print("Sampling applied, new size:", len(df))

    # encode labels
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    labels = list(le.classes_)
    print("Encoded labels:", len(labels))

    # stratify only if every class has >=2 samples
    counts = Counter(df["label_id"])
    can_stratify = min(counts.values()) >= 2
    stratify_arr = df["label_id"].values if can_stratify else None
    if not can_stratify:
        print("Warning: stratified split disabled (some classes <2 samples)")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label_id"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=stratify_arr
    )

    print("Train size:", len(X_train), "Eval size:", len(X_test))

    # tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(texts):
        return tokenizer(texts, truncation=True, padding=False, max_length=args.max_length)

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

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(labels))
    # gradient checkpointing reduces memory at cost of some compute
    try:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    except Exception:
        pass
    if device == "cuda":
        model.to("cuda")

    # TrainingArguments tuned for practical Colab runs
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=50,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.06,
        weight_decay=0.01,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=42,
        report_to="none"
    )

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

    # Train
    trainer.train()

    # Evaluate & save
    preds = trainer.predict(test_ds)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)

    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    (out_dir / "classification_report.txt").write_text(report)
    print(report)

    plot_confusion(y_true, y_pred, labels, out_dir / "confusion_matrix.png")
    trainer.save_model(out_dir / "model")
    tokenizer.save_pretrained(out_dir / "model")
    print("All outputs saved to:", out_dir)

if __name__ == "__main__":
    main()

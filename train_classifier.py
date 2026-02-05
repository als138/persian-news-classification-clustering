# train_classifier.py  (SAFE, ParsBERT, automatic handling of rare classes)
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

# -----------------------
# Utilities
# -----------------------
def load_csv(data_dir: Path):
    csvs = list(Path(data_dir).glob("*.csv"))
    if not csvs:
        raise FileNotFoundError("❌ No CSV file found in data directory")
    df = pd.read_csv(csvs[0])
    return df

def handle_rare_classes(df, label_col="subgroup", min_samples=2, mode="auto", rare_frac_threshold=0.10, merge_label="other"):
    """
    mode: "auto" | "drop" | "merge" | "none"
      - auto: drop rare samples if they are < rare_frac_threshold of data, else merge into `merge_label`
      - drop: remove samples of classes with < min_samples
      - merge: merge rare classes into `merge_label`
      - none: do nothing
    Returns cleaned df and info dict.
    """
    counts = Counter(df[label_col])
    rare_labels = [lbl for lbl, cnt in counts.items() if cnt < min_samples]
    rare_total = sum(counts[l] for l in rare_labels) if rare_labels else 0
    total = len(df)
    frac = rare_total / total if total > 0 else 0

    info = {"rare_labels": rare_labels, "rare_total": rare_total, "fraction": frac, "action": None}

    if not rare_labels:
        info["action"] = "none_needed"
        return df.copy(), info

    if mode == "none":
        info["action"] = "none"
        return df.copy(), info

    if mode == "drop":
        df2 = df[~df[label_col].isin(rare_labels)].copy()
        info["action"] = "drop"
        return df2, info

    if mode == "merge":
        df2 = df.copy()
        df2.loc[df2[label_col].isin(rare_labels), label_col] = merge_label
        info["action"] = "merge"
        return df2, info

    # auto:
    if mode == "auto":
        if frac < rare_frac_threshold:
            df2 = df[~df[label_col].isin(rare_labels)].copy()
            info["action"] = "drop_auto"
            return df2, info
        else:
            df2 = df.copy()
            df2.loc[df2[label_col].isin(rare_labels), label_col] = merge_label
            info["action"] = "merge_auto"
            return df2, info

    # fallback
    info["action"] = "none"
    return df.copy(), info

def plot_and_save_confusion(y_true, y_pred, labels, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# -----------------------
# Metrics for Trainer
# -----------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro}

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to data directory containing CSV")
    parser.add_argument("--output-dir", required=True, help="output directory (e.g. /content/drive/... )")
    parser.add_argument("--min-samples", type=int, default=2, help="min samples per class to allow stratify")
    parser.add_argument("--rare-mode", choices=["auto","drop","merge","none"], default="auto",
                        help="how to handle rare classes: auto/drop/merge/none")
    parser.add_argument("--rare-threshold", type=float, default=0.10,
                        help="if mode=auto: fraction threshold under which rare samples are dropped")
    parser.add_argument("--merge-label", type=str, default="other", help="label name for merged rare classes")
    args = parser.parse_args()

    # require GPU for ParsBERT training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device != "cuda":
        print("⚠️ Warning: CUDA is not available. Training will run on CPU (very slow).")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Loading CSV...")
    df = load_csv(Path(args.data))

    # explicit mapping for Persian News
    label_col = "subgroup"
    text_col = "body"

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV columns: {list(df.columns)}")
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in CSV columns: {list(df.columns)}")

    df = df[[text_col, label_col]].dropna().rename(columns={text_col: "text", label_col: "label"})
    print(f"Initial samples: {len(df)} | unique labels: {df['label'].nunique()}")

    # handle rare classes per user's choice/auto
    df_clean, info = handle_rare_classes(df, label_col="label",
                                         min_samples=args.min_samples,
                                         mode=args.rare_mode,
                                         rare_frac_threshold=args.rare_threshold,
                                         merge_label=args.merge_label)
    print("Rare classes info:", info)
    print(f"After handling rare classes: samples={len(df_clean)} | unique labels={df_clean['label'].nunique()}")

    # if still any labels with < min_samples, drop them forcibly (safety)
    counts_after = Counter(df_clean["label"])
    still_rare = [l for l,c in counts_after.items() if c < args.min_samples]
    if still_rare:
        print("⚠️ Some labels still have < min_samples after handling. Dropping them:", still_rare)
        df_clean = df_clean[~df_clean["label"].isin(still_rare)].copy()

    print("Final distribution (label:count):")
    print(pd.Series(df_clean["label"]).value_counts().sort_index())

    # encode labels
    le = LabelEncoder()
    df_clean["label_id"] = le.fit_transform(df_clean["label"])
    labels = list(le.classes_)
    print("Encoded", len(labels), "labels.")

    # decide stratify
    label_counts = Counter(df_clean["label_id"])
    can_stratify = min(label_counts.values()) >= args.min_samples if label_counts else False
    stratify_arr = df_clean["label_id"].values if can_stratify else None
    if not can_stratify:
        print("⚠️ Stratified split disabled: some classes have fewer than", args.min_samples, "samples.")

    # train/test split
    X = df_clean["text"].tolist()
    y = df_clean["label_id"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arr
    )
    print("Train size:", len(X_train), "Test size:", len(X_test))

    # ------------------------
    # Tokenizer & model
    # ------------------------
    MODEL_NAME = "HooshvareLab/bert-base-parsbert-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(texts):
        return tokenizer(texts, truncation=True, padding=False, max_length=384)

    train_enc = tokenize(X_train)
    test_enc = tokenize(X_test)

    # dataset wrapper
    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, enc, labels):
            self.enc = enc
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k,v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_ds = DictDataset(train_enc, y_train)
    test_ds = DictDataset(test_enc, y_test)

    # model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(labels))
    if device == "cuda":
        model.to("cuda")

    # ------------------------
    # TrainingArguments (Best practice)
    # ------------------------
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,   # effective batch size = 16
        num_train_epochs=5,
        learning_rate=2e-5,
        warmup_ratio=0.1,
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

    print("🔥 Starting training...")
    trainer.train()

    print("📊 Running final evaluation...")
    preds = trainer.predict(test_ds)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)

    # save report + confusion
    report_txt = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    (out_dir / "classification_report.txt").write_text(report_txt)
    print(report_txt)

    plot_and_save_confusion(y_true, y_pred, labels, out_dir / "confusion_matrix.png")

    trainer.save_model(out_dir / "final_model")
    tokenizer.save_pretrained(out_dir / "final_model")

    print("✅ All outputs saved to:", out_dir)

if __name__ == "__main__":
    main()

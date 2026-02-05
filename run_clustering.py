"""
Run K-Means clustering over ParsBERT embeddings and generate reports/plots.

Usage:
    python run_clustering.py --data data/ --output-dir outputs/clustering
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data_loader import load_persian_news_dataset
from src.clustering import run_clustering_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="ParsBERT embeddings + KMeans clustering")
    parser.add_argument("--data", required=True, help="Path to CSV file or folder containing dataset")
    parser.add_argument("--output-dir", default="outputs/clustering", help="Directory to save clustering artifacts")
    parser.add_argument("--model-name", default="HooshvareLab/bert-base-parsbert-uncased",
                        help="ParsBERT model name")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length for tokenizer")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for embedding extraction")
    parser.add_argument("--sample", type=int, default=None,
                        help="Optional: randomly sample N rows for quick runs")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device != "cuda":
        print("⚠️ CUDA not detected; embedding extraction will run on CPU (slower).")

    print("📥 Loading dataset...")
    df = load_persian_news_dataset(args.data)
    print(f"Loaded {len(df)} rows with {df['label'].nunique()} unique labels.")

    if args.sample is not None and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"Sampled down to {len(df)} rows for faster experimentation.")

    texts = df["body"].astype(str).tolist()
    true_labels = df["label"].astype(str).tolist()
    n_clusters = df["label"].nunique()

    print(f"➡️ Running clustering with K = {n_clusters} (unique labels).")
    outputs = run_clustering_pipeline(
        texts=texts,
        true_labels=true_labels,
        n_clusters=n_clusters,
        output_dir=out_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    summary_lines = [
        "# Clustering Summary",
        "",
        f"- Samples: {len(texts)}",
        f"- Clusters (K): {n_clusters}",
        f"- Embedding dim: {outputs.embeddings.shape[1] if outputs.embeddings.size else 0}",
        f"- KMeans inertia: {outputs.kmeans_inertia:.2f}",
        f"- PCA explained variance (2D): {outputs.pca_explained_var:.4f}",
        "",
        "Artifacts:",
        "- parsbert_embeddings.npy",
        "- pca_clusters.png",
        "- cluster_profiles.md",
        "- alignment_report.md (if labels provided)",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_lines))

    print("✅ Clustering complete. Artifacts saved to", out_dir.resolve())


if __name__ == "__main__":
    main()

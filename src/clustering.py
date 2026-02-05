"""
Clustering utilities for Persian news embeddings.

Pipeline:
- Compute ParsBERT embeddings (uses [CLS]).
- Run K-Means with K equal to the number of gold categories (or user-provided).
- Reduce to 2D with PCA for visualization.
- Build cluster profiles using average TF-IDF scores per cluster.
- Evaluate alignment to ground-truth labels (purity, ARI, AMI, V-measure, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    v_measure_score,
)
from transformers import AutoModel, AutoTokenizer


@dataclass
class ClusteringOutputs:
    embeddings: np.ndarray
    cluster_labels: np.ndarray
    pca_points: np.ndarray
    pca_explained_var: float
    kmeans_inertia: float


def compute_parsbert_embeddings(
    texts: List[str],
    model_name: str = "HooshvareLab/bert-base-parsbert-uncased",
    max_length: int = 256,
    batch_size: int = 16,
    device: Optional[str] = None,
) -> np.ndarray:
    """Return [CLS] embeddings for texts using ParsBERT."""

    if len(texts) == 0:
        return np.empty((0, 768))

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    batches = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        batches.append(cls_embeddings)

    return np.vstack(batches)


def run_kmeans(
    embeddings: np.ndarray, n_clusters: int, random_state: int = 42, n_init: int = 10
) -> Tuple[KMeans, np.ndarray]:
    """Fit K-Means and return model and labels."""
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    labels = kmeans.fit_predict(embeddings)
    return kmeans, labels


def reduce_pca(
    embeddings: np.ndarray, n_components: int = 2, random_state: int = 42
) -> Tuple[PCA, np.ndarray]:
    """Reduce embeddings to 2D with PCA."""
    pca = PCA(n_components=n_components, random_state=random_state)
    points = pca.fit_transform(embeddings)
    return pca, points


def plot_clusters(
    pca_points: np.ndarray,
    cluster_labels: np.ndarray,
    save_path: Path,
    true_labels: Optional[Iterable[str]] = None,
) -> None:
    """Scatter plot colored by cluster id. If true labels provided, show hull markers."""
    df = pd.DataFrame({
        "pc1": pca_points[:, 0],
        "pc2": pca_points[:, 1],
        "cluster": cluster_labels,
    })
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x="pc1",
        y="pc2",
        hue="cluster",
        palette="tab10",
        s=35,
        alpha=0.85,
        edgecolor="none",
    )
    plt.title("K-Means clusters (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def compute_cluster_profiles(
    texts: List[str],
    cluster_labels: np.ndarray,
    top_k: int = 10,
    max_features: int = 20000,
) -> List[Dict[str, List[str]]]:
    """Return top-k TF-IDF terms for each cluster."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(texts)
    vocab = np.array(vectorizer.get_feature_names_out())

    profiles = []
    for cid in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == cid
        if mask.sum() == 0:
            continue
        mean_scores = tfidf[mask].mean(axis=0).A1
        top_idx = mean_scores.argsort()[::-1][:top_k]
        profiles.append({
            "cluster": int(cid),
            "top_terms": vocab[top_idx].tolist(),
        })
    return profiles


def alignment_metrics(
    true_labels: Iterable[str], cluster_labels: np.ndarray
) -> Dict[str, float]:
    """Compute unsupervised alignment scores."""
    y_true = np.array(list(true_labels))
    return {
        "purity": _purity(y_true, cluster_labels),
        "ari": adjusted_rand_score(y_true, cluster_labels),
        "ami": adjusted_mutual_info_score(y_true, cluster_labels),
        "homogeneity": homogeneity_score(y_true, cluster_labels),
        "completeness": completeness_score(y_true, cluster_labels),
        "v_measure": v_measure_score(y_true, cluster_labels),
    }


def build_contingency(true_labels: Iterable[str], cluster_labels: np.ndarray) -> pd.DataFrame:
    """Return cluster x true label contingency table."""
    return pd.crosstab(cluster_labels, true_labels)


def _purity(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    table = pd.crosstab(cluster_labels, true_labels)
    majority = table.max(axis=1).sum()
    return float(majority) / len(true_labels) if len(true_labels) else 0.0


def save_profiles_markdown(
    profiles: List[Dict[str, List[str]]], save_path: Path
) -> None:
    lines = ["# Cluster Profiles (Top TF-IDF terms)", ""]
    for profile in profiles:
        terms = ", ".join(profile["top_terms"])
        lines.append(f"- Cluster {profile['cluster']}: {terms}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines))


def save_alignment_report(
    metrics: Dict[str, float],
    contingency: pd.DataFrame,
    save_path: Path,
) -> None:
    lines = ["# Cluster ↔ Label Alignment", ""]
    for key, val in metrics.items():
        lines.append(f"- {key}: {val:.4f}")
    lines.append("")
    lines.append("## Contingency Table (rows=clusters, cols=true labels)")
    lines.append(contingency.to_markdown())

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines))


def run_clustering_pipeline(
    texts: List[str],
    true_labels: Optional[Iterable[str]],
    n_clusters: int,
    output_dir: Path,
    model_name: str = "HooshvareLab/bert-base-parsbert-uncased",
    max_length: int = 256,
    batch_size: int = 16,
) -> ClusteringOutputs:
    """Full clustering flow: embeddings -> K-Means -> PCA -> reports."""

    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = compute_parsbert_embeddings(
        texts,
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
    )
    np.save(output_dir / "parsbert_embeddings.npy", embeddings)

    kmeans, cluster_labels = run_kmeans(embeddings, n_clusters=n_clusters)
    pca, pca_points = reduce_pca(embeddings)

    plot_clusters(pca_points, cluster_labels, output_dir / "pca_clusters.png")

    profiles = compute_cluster_profiles(texts, cluster_labels)
    save_profiles_markdown(profiles, output_dir / "cluster_profiles.md")

    if true_labels is not None:
        contingency = build_contingency(true_labels, cluster_labels)
        metrics = alignment_metrics(true_labels, cluster_labels)
        save_alignment_report(metrics, contingency, output_dir / "alignment_report.md")

        # also persist as CSV for convenience
        contingency.to_csv(output_dir / "contingency_table.csv")
        pd.DataFrame([metrics]).to_csv(output_dir / "alignment_scores.csv", index=False)

    # capture PCA explained variance for quick reference
    pca_ev = float(pca.explained_variance_ratio_.sum())

    return ClusteringOutputs(
        embeddings=embeddings,
        cluster_labels=cluster_labels,
        pca_points=pca_points,
        pca_explained_var=pca_ev,
        kmeans_inertia=float(kmeans.inertia_),
    )

# Persian News Classification and Clustering

Persian news classification and clustering with **hazm** preprocessing, **TF-IDF** vectorization, and **ParsBERT** — including source code and evaluation reports.

## Project Overview

This project applies NLP to formal and structured Persian news texts for:
1. **Part One (Supervised):** Topic classification using body text
2. **Part Two (Unsupervised):** Discovering hidden patterns and clustering

## Dataset

**Persian News Dataset** from Kaggle:  
https://www.kaggle.com/datasets/amirzenoozi/persian-news-dataset

- News from official agencies (Fars, Mehr)
- Fields: `id`, `title`, `abstract`, `body`, `tags`
- Text for classification: **body**
- Labels: from **tags** or **category** column

### Download Dataset

```bash
# Using Kaggle API (requires ~/.kaggle/kaggle.json)
python scripts/download_dataset.py --output data/

# Or download manually from Kaggle and place CSV in data/
```

---

## Part One: Text Classification

Given news body text, predict subject category (sports, economics, politics, etc.).

### Implementation

| Step | Description |
|------|-------------|
| **Preprocessing** | hazm: Normalizer, stop-word removal, Lemmatizer |
| **Baseline** | TF-IDF + SVM (linear kernel) |
| **SOTA** | ParsBERT [CLS] embeddings + Logistic Regression |
| **Evaluation** | 80/20 split, Accuracy, F1-Score, Confusion Matrix |

### Setup

```bash
pip install -r requirements.txt
```

### Run

```bash
python train_classifier.py --data data/your_dataset.csv
# or
python train_classifier.py --data data/  # auto-detect CSV in folder
```

**Options:**
- `--output-dir outputs` — Save confusion matrices
- `--test-size 0.2` — Test fraction (default 80/20)
- `--skip-sota` — Skip ParsBERT (faster for testing)

### Output

- Console: Accuracy, F1 (macro), F1 (weighted)
- `outputs/confusion_matrix_baseline.png`
- `outputs/confusion_matrix_sota.png`

---

## Part Two: Clustering & Pattern Analysis

Unsupervised grouping of news when labels are hidden.

### Steps
- ParsBERT [CLS] embeddings (768-dim)
- K-Means with **K = number of gold categories**
- PCA → 2D for visualization
- Cluster profiles: top 10 TF‑IDF terms per cluster
- Alignment metrics vs. gold labels (purity, ARI, AMI, homogeneity, completeness, V-measure)

### Run

```bash
python run_clustering.py --data data/ --output-dir outputs/clustering
# optional flags:
#   --sample 2000      # subsample for quicker runs
#   --batch-size 8     # lower if GPU memory is tight
#   --max-length 256   # adjust token length
```

### Clustering Artifacts
- `outputs/clustering/parsbert_embeddings.npy`
- `outputs/clustering/pca_clusters.png`
- `outputs/clustering/cluster_profiles.md`
- `outputs/clustering/alignment_report.md` (plus `alignment_scores.csv`, `contingency_table.csv`)
- `outputs/clustering/summary.md`

### Reporting Checklist (Deliverables)
- **Classification Performance Analysis:** compare SVM vs. ParsBERT (accuracy/F1) and explain the winner.
- **Error Analysis:** use confusion matrix to note overlapping categories (e.g., politics vs. economy) and why.
- **Clustering Report:** include PCA scatter plot and top terms per cluster.
- **Alignment Evaluation:** discuss purity / ARI / V-measure and how clusters map to true categories.

---

## Project Structure

```
├── train_classifier.py      # Part 1 main pipeline
├── run_clustering.py        # Part 2 clustering pipeline
├── requirements.txt
├── src/
│   ├── data_loader.py       # Load Persian News Dataset
│   ├── preprocessing.py     # hazm preprocessing
│   ├── evaluation.py        # Metrics & confusion matrix
│   └── models/
│       ├── baseline_model.py   # TF-IDF + SVM
│       └── sota_model.py       # ParsBERT + Logistic Regression
│   └── clustering.py        # Embeddings -> KMeans -> PCA -> profiles & alignment
├── scripts/
│   └── download_dataset.py  # Kaggle download helper
└── outputs/                 # Generated reports
```

---

## License

MIT License — see [LICENSE](LICENSE).

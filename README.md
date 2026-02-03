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

## Project Structure

```
├── train_classifier.py      # Part 1 main pipeline
├── requirements.txt
├── src/
│   ├── data_loader.py       # Load Persian News Dataset
│   ├── preprocessing.py     # hazm preprocessing
│   ├── evaluation.py        # Metrics & confusion matrix
│   └── models/
│       ├── baseline_model.py   # TF-IDF + SVM
│       └── sota_model.py       # ParsBERT + Logistic Regression
├── scripts/
│   └── download_dataset.py  # Kaggle download helper
└── outputs/                 # Generated reports
```

---

## License

MIT License — see [LICENSE](LICENSE).

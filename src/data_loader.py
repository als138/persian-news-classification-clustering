"""
Data loading utilities for Persian News Dataset.
Supports CSV and SQLite from Kaggle: https://www.kaggle.com/datasets/amirzenoozi/persian-news-dataset
"""

import pandas as pd
import sqlite3
from pathlib import Path


def load_persian_news_dataset(data_path: str) -> pd.DataFrame:
    """
    Load Persian News Dataset from CSV or directory.
    
    Expected columns: id, title, abstract, body, tags (and optionally category)
    The dataset may use 'tags' or 'category' for subject labels.
    
    Args:
        data_path: Path to CSV file or directory containing the dataset
        
    Returns:
        DataFrame with 'body' (text) and 'label' (category) columns
    """
    path = Path(data_path)
    
    if path.is_dir():
        # Look for CSV or SQLite in directory
        csv_files = list(path.glob("*.csv"))
        sqlite_files = list(path.glob("*.db")) + list(path.glob("*.sqlite"))
        if csv_files:
            data_path = str(max(csv_files, key=lambda f: f.stat().st_size))
        elif sqlite_files:
            data_path = str(sqlite_files[0])
        else:
            raise FileNotFoundError(f"No CSV or SQLite files found in {data_path}")
    
    path = Path(data_path)
    if path.suffix.lower() in (".db", ".sqlite"):
        conn = sqlite3.connect(path)
        df = pd.read_sql("SELECT * FROM news", conn)
        conn.close()
    else:
        df = pd.read_csv(data_path, encoding="utf-8-sig")
    
    # Normalize column names (handle variations)
    df.columns = df.columns.str.strip().str.lower()
    
    # Map common column names
    col_map = {
        "body": ["body", "text", "content", "news_body"],
        "category": ["category", "label", "subject", "cat", "tags"],
    }
    
    text_col = None
    for candidate in col_map["body"]:
        if candidate in df.columns:
            text_col = candidate
            break
    
    label_col = None
    for candidate in col_map["category"]:
        if candidate in df.columns:
            label_col = candidate
            break
    
    if text_col is None:
        # Fallback: combine title + abstract + body if available
        if "title" in df.columns and "abstract" in df.columns:
            df["body"] = df["title"].fillna("") + " " + df["abstract"].fillna("")
            text_col = "body"
        elif "title" in df.columns:
            df["body"] = df["title"]
            text_col = "body"
        else:
            raise ValueError(
                f"Could not find text column. Available: {list(df.columns)}"
            )
    
    if label_col is None:
        raise ValueError(
            f"Could not find category/label column. Available: {list(df.columns)}"
        )
    
    # Handle tags: if tags is a string with multiple values, take first as category
    if label_col == "tags" and df[label_col].dtype == object:
        # Tags might be "sport,economics" or "ورزش" - use first tag as category
        df["label"] = df[label_col].astype(str).str.split(",").str[0].str.strip()
    else:
        df["label"] = df[label_col]
    
    # Filter valid rows
    df = df[df["body"].notna() & (df["body"].astype(str).str.len() > 0)]
    df = df[df["label"].notna() & (df["label"].astype(str).str.len() > 0)]
    
    return df[["body", "label"]].reset_index(drop=True)

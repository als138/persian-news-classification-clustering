#!/usr/bin/env python3
"""
Download Persian News Dataset from Kaggle.
Requires Kaggle API credentials: ~/.kaggle/kaggle.json

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --output data/
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory for dataset",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        subprocess.run(
            [
                sys.executable, "-m", "kaggle", "datasets", "download",
                "-d", "amirzenoozi/persian-news-dataset",
                "-p", str(output_dir),
                "--unzip",
            ],
            check=True,
        )
        print(f"Dataset downloaded to {output_dir}")
    except subprocess.CalledProcessError as e:
        print("Kaggle download failed. Ensure:")
        print("  1. pip install kaggle")
        print("  2. Kaggle API key at ~/.kaggle/kaggle.json")
        print("  3. Or download manually from:")
        print("     https://www.kaggle.com/datasets/amirzenoozi/persian-news-dataset")
        sys.exit(1)


if __name__ == "__main__":
    main()

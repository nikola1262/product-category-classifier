"""Treniranje i evaluacija klasifikatora proizvoda zasnovanog SAMO na naslovu.

Komande:
    python train_model.py
    python train_model.py --data data/products.csv --output models/product_category_model.pkl
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Grafici se kreiraju i bez otvorenog prozora.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.data_utils import clean_data, read_raw_data

ROOT = Path(__file__).resolve().parent


def word_vectorizer() -> TfidfVectorizer:
    # Jedna/dve reci prepoznaju npr. 'washing machine', 'iphone 7'.
    return TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=120_000,
        sublinear_tf=True, strip_accents="unicode",
    )


def char_vectorizer() -> TfidfVectorizer:
    # Karakterski n-grami prepoznaju delove sifara i razlike u pisanju modela.
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        max_features=180_000, sublinear_tf=True, strip_accents="unicode",
    )


def build_models() -> dict[str, Pipeline]:
    """Sve ce biti fitovano SAMO na trening skupu (bez TF-IDF leakage-a)."""
    return {
        "Logistic Regression / word": Pipeline([
            ("tfidf", word_vectorizer()), ("classifier", LogisticRegression(max_iter=1000)),
        ]),
        "Multinomial Naive Bayes / word": Pipeline([
            ("tfidf", word_vectorizer()), ("classifier", MultinomialNB(alpha=0.3)),
        ]),
        "Linear SVC / char": Pipeline([
            ("tfidf", char_vectorizer()), ("classifier", LinearSVC(random_state=42)),
        ]),
        "Linear SVC / word+char": Pipeline([
            ("tfidf", FeatureUnion([
                ("word", word_vectorizer()), ("char", char_vectorizer()),
            ])),
            ("classifier", LinearSVC(random_state=42)),
        ]),
    }


def evaluate_models(data: pd.DataFrame):
    """Uporedi vise algoritama na identicnom stratifikovanom test skupu."""
    x_train, x_test, y_train, y_test = train_test_split(
        data["title"].astype(str), data["category"].astype(str),
        test_size=0.20, stratify=data["category"], random_state=42,
    )
    rows, fitted, predictions = [], {}, {}
    for name, model in build_models().items():
        print(f"Treniram: {name}", flush=True)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        rows.append({
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        })
        fitted[name] = model
        predictions[name] = pred
    ranking = pd.DataFrame(rows).sort_values(["f1_macro", "accuracy"], ascending=False)
    best_name = str(ranking.iloc[0]["model"])
    return ranking, fitted, predictions, best_name, x_test, y_test


def run_training(data_path: Path, output_path: Path, reports_dir: Path) -> None:
    raw = read_raw_data(str(data_path))
    data, summary = clean_data(raw)
    print(f"Ocisceno: {summary['clean_rows']} naslova / {summary['n_categories']} kategorija")
    rankings, fitted, predictions, best_name, x_test, y_test = evaluate_models(data)
    print("\n", rankings.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    reports_dir.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(reports_dir / "model_comparison.csv", index=False)
    y_pred = predictions[best_name]
    labels = sorted(data["category"].unique())
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(reports_dir / "classification_report.csv")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(reports_dir / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(9, 6.6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=labels, xticks_rotation=48, cmap="Blues", ax=ax,
        values_format="d", colorbar=False,
    )
    ax.set_title(f"Matrica zabune — {best_name}", pad=13)
    fig.tight_layout()
    fig.savefig(reports_dir / "confusion_matrix.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    order = rankings.iloc[::-1]
    y = np.arange(len(order))
    ax.barh(y - .18, order["accuracy"], height=.34, label="Accuracy")
    ax.barh(y + .18, order["f1_macro"], height=.34, label="Macro F1")
    ax.set_yticks(y, order["model"])
    ax.set_xlim(.8, 1.005)
    ax.set_xlabel("Rezultat na izdvojenom test skupu")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(reports_dir / "model_comparison.png", dpi=170)
    plt.close(fig)

    # Tek posle evaluacije ponovo treniramo pobednika na svim cistim podacima.
    final = build_models()[best_name]
    final.fit(data["title"].astype(str), data["category"].astype(str))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, output_path, compress=3)
    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path.name),
        "best_model": best_name,
        "test_samples": int(len(x_test)),
        "metrics": rankings.iloc[0].to_dict(),
        "cleaning": summary,
        "packages": {"sklearn": sklearn.__version__, "pandas": pd.__version__},
        "note": "Evaluacija na 20% naslova; finalni .pkl treniran na svih 100% ociscenih podataka.",
    }
    (reports_dir / "training_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nPobednik: {best_name}. Sacuvan: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Treniraj model za kategorije proizvoda")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "products.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "product_category_model.pkl")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    run_training(args.data, args.output, args.reports)


if __name__ == "__main__":
    main()

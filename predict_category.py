"""Konzolni program: predlozi kategoriju na osnovu naziva proizvoda.

Pokretanje:
    python predict_category.py
    python predict_category.py --text "bosch serie 4 kgv39vl31g"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "product_category_model.pkl"


def predict_title(model, title: str) -> str:
    """Model prima listu tekstova i vraca naziv kategorije."""
    if not title.strip():
        raise ValueError("Naziv proizvoda ne sme biti prazan.")
    return str(model.predict([title.strip()])[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Klasifikacija proizvoda na osnovu naziva")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--text", help="Jednokratna predikcija, bez interaktivnog unosa")
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Model ne postoji: {args.model}\nPokreni: python train_model.py")
    # .pkl/.joblib ucitavaj samo iz pouzdanih izvora (pickle moze izvrsiti kod).
    model = joblib.load(args.model)

    if args.text is not None:
        print(predict_title(model, args.text))
        return

    print("Klasifikacija proizvoda | unesi naziv proizvoda ('izlaz' za kraj)")
    while True:
        try:
            title = input("\nNaziv proizvoda: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKraj programa.")
            break
        if title.casefold() in {"izlaz", "exit", "quit", "q"}:
            print("Kraj programa.")
            break
        if not title:
            print("Unesi naziv koji nije prazan.")
            continue
        print(f"Predlozena kategorija: {predict_title(model, title)}")


if __name__ == "__main__":
    main()

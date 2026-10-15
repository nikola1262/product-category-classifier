"""Ucitavanje i ciscenje kataloga proizvoda za ML klasifikaciju po naslovu."""
from __future__ import annotations

import pandas as pd

# Razliciti zapisi iste kategorije nisu razlicite ciljne klase.
CATEGORY_MAP = {
    "mobile phones": "Mobile Phones", "mobile phone": "Mobile Phones",
    "digital cameras": "Digital Cameras",
    "microwaves": "Microwaves",
    "washing machines": "Washing Machines",
    "fridge freezers": "Fridge Freezers",
    "fridges": "Fridges", "fridge": "Fridges",
    "freezers": "Freezers", "dishwashers": "Dishwashers",
    "cpus": "CPUs", "cpu": "CPUs", "tvs": "TVs",
}


def read_raw_data(path: str) -> pd.DataFrame:
    """Ucitaj CSV; strip uklanja suvisne razmake iz naziva kolona."""
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = frame.columns.str.strip()
    required = {"Product Title", "Category Label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Nedostaju obavezne kolone: {sorted(missing)}")
    return frame


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Vrati jedan proizvod po jedinstvenom normalizovanom naslovu.

    Ako je *isti* naslov oznacen razlicitim kategorijama, ukloni sve njegove
    redove. Time izbegavamo kontradiktorne oznake i leakage izmedju skupova.
    """
    df = raw.copy()
    df.columns = df.columns.str.strip()
    starting = len(df)

    df["title"] = df["Product Title"].astype("string").str.strip().str.replace(
        r"\s+", " ", regex=True
    )
    df["title_key"] = df["title"].str.casefold()
    df["category"] = (
        df["Category Label"].astype("string").str.strip().str.casefold().map(CATEGORY_MAP)
    )

    valid = df["title"].notna() & df["category"].notna() & df["title"].str.len().gt(0)
    df = df.loc[valid.fillna(False)].copy()
    after_invalid = len(df)

    # Naslov sa vise razlicitih kategorija ima nejasnu ciljnu oznaku.
    conflicting = df.groupby("title_key")["category"].nunique()
    ambiguous = set(conflicting[conflicting > 1].index)
    conflicting_rows = int(df["title_key"].isin(ambiguous).sum())
    df = df.loc[~df["title_key"].isin(ambiguous)].copy()

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["title_key"], keep="first").reset_index(drop=True)
    df = df[["title", "category", "title_key"]]
    summary = {
        "raw_rows": starting,
        "invalid_or_unknown_rows": starting - after_invalid,
        "ambiguous_title_groups": len(ambiguous),
        "ambiguous_rows_removed": conflicting_rows,
        "duplicate_title_rows_removed": before_dedup - len(df),
        "clean_rows": len(df),
        "n_categories": int(df["category"].nunique()),
        "categories": df["category"].value_counts().to_dict(),
    }
    if df["category"].nunique() < 2:
        raise ValueError("Potrebne su bar dve kategorije za klasifikaciju.")
    return df, summary

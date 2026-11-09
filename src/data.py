"""Chargement et nettoyage de la base Auto MPG."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LOCAL_CSV = Path(__file__).resolve().parent.parent / "data" / "auto_mpg.csv"


def load_raw(offline: bool = False) -> tuple[pd.DataFrame, str]:
    """Charge les 398 lignes brutes.

    Source officielle de l'énoncé : seaborn.load_dataset('mpg') (téléchargement
    depuis le dépôt GitHub « seaborn-data »). Si Internet n'est pas disponible,
    on utilise la copie locale data/auto_mpg.csv (copie identique, enregistrée
    depuis seaborn).
    """
    if not offline:
        try:
            import seaborn as sns
            return sns.load_dataset("mpg"), "seaborn.load_dataset('mpg')"
        except Exception as exc:  # pas de réseau, proxy, etc.
            print(f"[data] seaborn indisponible ({type(exc).__name__}) -> copie locale")
    return pd.read_csv(LOCAL_CSV), f"copie locale {LOCAL_CSV.name}"


def load_clean(offline: bool = False) -> tuple[pd.DataFrame, dict]:
    """Nettoyage :
    1. suppression des 6 véhicules sans valeur de 'horsepower' -> 392 lignes
       (c'est le nombre d'échantillons annoncé par l'énoncé) ;
    2. suppression de 'name' : identifiant quasi unique (pas une caractéristique
       physique), il ne généraliserait pas à un nouveau véhicule.
    """
    raw, source = load_raw(offline)
    info = {"source": source, "n_raw": len(raw),
            "n_missing_horsepower": int(raw["horsepower"].isna().sum())}
    df = raw.dropna(subset=["horsepower"]).drop(columns=["name"]).reset_index(drop=True)
    df["origin"] = df["origin"].astype(str)
    df["cylinders"] = df["cylinders"].astype(int)
    info["n_clean"] = len(df)
    return df, info

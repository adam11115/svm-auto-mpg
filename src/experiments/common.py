"""Préparation commune aux deux problèmes (même découpage train/test partout)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import load_clean
from ..preprocessing import TabularPreprocessor, TargetDiscretizer, stratified_train_test_split

SEED = 42          # graine unique : tous les résultats sont reproductibles
TEST_SIZE = 0.20   # 20 % des véhicules mis de côté pour l'évaluation finale
N_FOLDS = 5        # validation croisée à 5 plis sur l'entraînement

TABLE_DIR = Path(__file__).resolve().parents[2] / "results" / "tables"


@dataclass
class Context:
    df: pd.DataFrame
    info: dict
    idx_train: np.ndarray
    idx_test: np.ndarray
    prep: TabularPreprocessor
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray        # mpg (continu)
    y_test: np.ndarray
    summary: dict = field(default_factory=dict)


def build_context(offline: bool = False) -> Context:
    df, info = load_clean(offline)
    y = df["mpg"].to_numpy(dtype=float)
    # Stratification du découpage sur les terciles de mpg : le train et le test
    # couvrent la même plage de consommation (utile aux DEUX problèmes).
    strata = TargetDiscretizer("quantile").fit_transform(y)
    idx_tr, idx_te = stratified_train_test_split(strata, TEST_SIZE, SEED)
    prep = TabularPreprocessor().fit(df.iloc[idx_tr])      # appris sur le TRAIN seulement
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    return Context(df=df, info=info, idx_train=idx_tr, idx_test=idx_te, prep=prep,
                   X_train=prep.transform(df.iloc[idx_tr]), X_test=prep.transform(df.iloc[idx_te]),
                   y_train=y[idx_tr], y_test=y[idx_te])


def save_table(df: pd.DataFrame, name: str) -> str:
    path = TABLE_DIR / name
    df.to_csv(path, index=False, float_format="%.4f")
    return f"results/tables/{name}"

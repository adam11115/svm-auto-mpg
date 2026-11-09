"""Prétraitement implémenté depuis zéro (NumPy / pandas).

Contenu :
* StandardScalerScratch   : centrage-réduction  x' = (x - moyenne) / écart-type
* OneHotEncoderScratch    : encodage disjonctif des variables catégorielles
* TabularPreprocessor     : les deux combinés (équivalent d'un ColumnTransformer)
* TargetDiscretizer       : transformation de mpg (continu) en 3 classes
* stratified_train_test_split, stratified_kfold, kfold : découpages des données

RÈGLE D'OR (anti-fuite de données) : tous les paramètres (moyennes, écarts-types,
catégories, seuils de discrétisation) sont appris sur l'échantillon
d'ENTRAÎNEMENT uniquement, puis appliqués tels quels au test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = ["displacement", "horsepower", "weight", "acceleration", "model_year"]
CATEGORICAL_FEATURES = ["cylinders", "origin"]

# Ordre des classes : indice 0 = mpg le plus bas = CONSOMMATION ÉLEVÉE (attention :
# beaucoup de miles par gallon = peu de carburant consommé).
CLASS_NAMES = ["élevée", "moyenne", "faible"]


class StandardScalerScratch:
    def fit(self, X: np.ndarray) -> "StandardScalerScratch":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)          # écart-type « population » (ddof=0), comme sklearn
        self.scale_[self.scale_ == 0] = 1.0  # colonne constante : on ne divise pas par 0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class OneHotEncoderScratch:
    """Une colonne 0/1 par modalité vue à l'entraînement, SAUF la modalité de référence.

    drop='most_frequent' : on supprime la colonne de la modalité la plus fréquente
    (4 cylindres, origine USA). Sans cela, les colonnes d'une même variable
    somment toujours à 1 et sont colinéaires avec l'intercept b (« piège des
    variables indicatrices ») : les prédictions sont inchangées mais les
    coefficients ne sont plus interprétables. Chaque coefficient se lit alors
    comme un écart par rapport à la modalité de référence.
    Modalité inconnue au test -> que des 0 (traitée comme la référence).
    """

    def __init__(self, drop: str | None = "most_frequent"):
        self.drop = drop

    def fit(self, X: np.ndarray) -> "OneHotEncoderScratch":
        X = np.asarray(X, dtype=object)
        self.categories_, self.reference_ = [], []
        for j in range(X.shape[1]):
            values, counts = np.unique(X[:, j].astype(str), return_counts=True)
            cats = np.array(sorted(set(X[:, j])), dtype=object)
            ref = None
            if self.drop == "most_frequent":
                # en cas d'égalité d'effectif, np.argmax prend la première (ordre trié)
                ref_str = values[np.argmax(counts)]
                ref = next(c for c in cats if str(c) == ref_str)
                cats = np.array([c for c in cats if c != ref], dtype=object)
            self.categories_.append(cats)
            self.reference_.append(ref)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=object)
        blocks = [(X[:, [j]] == cats[None, :]).astype(float)
                  for j, cats in enumerate(self.categories_)]
        return np.hstack(blocks)

    def feature_names(self, input_names):
        return [f"{name}_{c}" for name, cats in zip(input_names, self.categories_) for c in cats]


class TabularPreprocessor:
    """Variables continues -> standardisées ; variables catégorielles -> one-hot (0/1)."""

    def __init__(self, numeric=NUMERIC_FEATURES, categorical=CATEGORICAL_FEATURES):
        self.numeric = list(numeric)
        self.categorical = list(categorical)

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        self.scaler_ = StandardScalerScratch().fit(df[self.numeric].to_numpy(dtype=float))
        self.encoder_ = OneHotEncoderScratch().fit(df[self.categorical].to_numpy(dtype=object))
        self.feature_names_ = self.numeric + self.encoder_.feature_names(self.categorical)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num = self.scaler_.transform(df[self.numeric].to_numpy(dtype=float))
        cat = self.encoder_.transform(df[self.categorical].to_numpy(dtype=object))
        return np.hstack([num, cat])

    def fit_transform(self, df):
        return self.fit(df).transform(df)


class TargetDiscretizer:
    """Transforme mpg en 3 classes ordonnées (0 = élevée, 1 = moyenne, 2 = faible).

    strategy :
      * 'quantile' : seuils = terciles de mpg (classes de même effectif)
      * 'uniform'  : 3 intervalles de même largeur entre min et max de mpg
      * 'domain'   : seuils fixés « métier » (par défaut 20 et 30 mpg,
                     soit environ 11,8 et 7,8 L/100 km)
    Règle d'affectation : classe 0 si mpg < s1, classe 1 si s1 <= mpg < s2, sinon 2.
    """

    def __init__(self, strategy: str = "quantile", thresholds=(20.0, 30.0)):
        self.strategy = strategy
        self.thresholds = thresholds

    def fit(self, y: np.ndarray) -> "TargetDiscretizer":
        y = np.asarray(y, dtype=float)
        if self.strategy == "quantile":
            t = np.quantile(y, [1 / 3, 2 / 3])
        elif self.strategy == "uniform":
            t = y.min() + (y.max() - y.min()) * np.array([1 / 3, 2 / 3])
        elif self.strategy == "domain":
            t = np.asarray(self.thresholds, dtype=float)
        else:
            raise ValueError(f"Stratégie inconnue : {self.strategy!r}")
        self.thresholds_ = np.asarray(t, dtype=float)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        # np.digitize(right=False) : y < t1 -> 0 ; t1 <= y < t2 -> 1 ; y >= t2 -> 2
        return np.digitize(np.asarray(y, dtype=float), self.thresholds_)

    def fit_transform(self, y):
        return self.fit(y).transform(y)


# ------------------------------------------------------------------ découpages
def stratified_train_test_split(strata: np.ndarray, test_size: float = 0.2, seed: int = 42):
    """Renvoie (idx_train, idx_test) en conservant la proportion de chaque strate."""
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata)
    train, test = [], []
    for s in np.unique(strata):
        idx = rng.permutation(np.flatnonzero(strata == s))
        n_test = int(round(test_size * idx.size))
        test.extend(idx[:n_test])
        train.extend(idx[n_test:])
    return np.sort(np.array(train)), np.sort(np.array(test))


def stratified_kfold(y: np.ndarray, k: int = 5, seed: int = 42):
    """Liste de k couples (idx_train, idx_validation), chaque classe répartie sur les k plis."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    fold_of = np.empty(y.size, dtype=int)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        fold_of[idx] = np.arange(idx.size) % k
    return [(np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)) for f in range(k)]


def kfold(n: int, k: int = 5, seed: int = 42):
    """Validation croisée simple (régression) : permutation puis k blocs."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    return [(np.sort(np.concatenate([folds[g] for g in range(k) if g != f])), np.sort(folds[f]))
            for f in range(k)]

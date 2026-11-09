"""Recherche d'hyperparamètres par validation croisée (grille), écrite depuis zéro.

Fonctionne avec n'importe quel objet possédant fit(X, y) / predict(X) :
nos modèles « maison » comme les modèles scikit-learn. Les deux sont donc
réglés avec EXACTEMENT les mêmes plis et la même grille.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd


def expand_grid(grid: dict) -> list:
    """{'C': [1, 10], 'gamma': [0.1]} -> [{'C': 1, 'gamma': 0.1}, {'C': 10, 'gamma': 0.1}]"""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def grid_search_cv(make_model, grid: dict, X: np.ndarray, y: np.ndarray, folds,
                   score_fn, greater_is_better: bool = True, tolerance: float = 0.01):
    """Évalue chaque combinaison sur chaque pli ; renvoie (tableau, paramètres retenus).

    make_model(**params) doit renvoyer un modèle NON entraîné.
    score_fn(y_true, y_pred) -> float.

    Règle de choix « parcimonieuse » :
      1. on garde les combinaisons dont le score moyen est à moins de
         `tolerance` (1 % en relatif) du meilleur score ;
      2. parmi elles, on prend le plus petit C (régularisation la plus forte,
         modèle le plus simple, le moins sujet au sur-apprentissage) ;
      3. à C égal, on prend le meilleur score (puis la première de la grille).
    Raison : avec 5 plis, l'écart-type du score CV vaut ~0,03 en accuracy et
    ~0,6 mpg en RMSE ; des écarts < 1 % ne sont pas significatifs, et un C plus
    petit généralise au moins aussi bien. tolerance=0 redonne le simple argmax.
    La colonne 'retenu' du tableau indique la combinaison choisie et
    'meilleur_brut' celle du simple argmax.
    """
    rows = []
    combos = expand_grid(grid)
    for params in combos:
        scores = []
        t0 = time.perf_counter()
        for tr, va in folds:
            model = make_model(**params).fit(X[tr], y[tr])
            scores.append(score_fn(y[va], model.predict(X[va])))
        rows.append({**params, "cv_mean": float(np.mean(scores)), "cv_std": float(np.std(scores)),
                     "fit_time_s": time.perf_counter() - t0})
    table = pd.DataFrame(rows)
    sign = 1 if greater_is_better else -1
    score = sign * table["cv_mean"].to_numpy()          # « plus grand = meilleur »
    raw_best = int(np.argmax(score))
    best_val = table["cv_mean"].iloc[raw_best]
    if greater_is_better:
        ok = table["cv_mean"].to_numpy() >= best_val - tolerance * abs(best_val)
    else:
        ok = table["cv_mean"].to_numpy() <= best_val + tolerance * abs(best_val)
    cand = np.flatnonzero(ok)
    if "C" in table:
        c_min = table["C"].iloc[cand].min()
        cand = cand[table["C"].iloc[cand].to_numpy() == c_min]
    chosen = int(cand[np.argmax(score[cand])])          # argmax : 1re occurrence en cas d'égalité
    table["meilleur_brut"] = False
    table.loc[raw_best, "meilleur_brut"] = True
    table["retenu"] = False
    table.loc[chosen, "retenu"] = True
    best = dict(combos[chosen])  # types d'origine conservés (degree reste un int)
    return table, best

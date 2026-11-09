"""Point d'entrée unique : exécute TOUT le projet et régénère results/.

Utilisation (depuis le dossier du projet) :
    python main.py                 # exécution standard (~3 à 4 minutes)
    python main.py --offline       # sans Internet : utilise data/auto_mpg.csv
    python main.py --full-scratch  # études de bruit aussi avec le SVR maison (plus long)
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn

from src import plotting
from src.experiments.classification import run_classification
from src.experiments.common import SEED, build_context
from src.experiments.discretization import run_discretization, run_exploration
from src.experiments.regression import run_regression

ROOT = Path(__file__).resolve().parent


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main():
    parser = argparse.ArgumentParser(description="SVM (SVC / SVR) sur Auto MPG — maison vs scikit-learn")
    parser.add_argument("--offline", action="store_true", help="utiliser la copie locale du jeu de données")
    parser.add_argument("--full-scratch", action="store_true",
                        help="faire aussi les 560 entraînements des études de bruit avec le SVR maison")
    args = parser.parse_args()

    t_start = time.perf_counter()
    plotting.setup()
    np.random.seed(SEED)

    print("[1/5] Chargement et préparation des données ...")
    ctx = build_context(offline=args.offline)
    print(f"      source : {ctx.info['source']} | {ctx.info['n_raw']} lignes brutes, "
          f"{ctx.info['n_missing_horsepower']} sans horsepower -> {ctx.info['n_clean']} véhicules")
    print(f"      train = {len(ctx.idx_train)} | test = {len(ctx.idx_test)} | "
          f"{ctx.X_train.shape[1]} variables après prétraitement")

    summary = {"environnement": {"python": platform.python_version(), "numpy": np.__version__,
                                 "scikit-learn": sklearn.__version__},
               "donnees": {**ctx.info, "n_train": len(ctx.idx_train), "n_test": len(ctx.idx_test),
                           "variables": ctx.prep.feature_names_}}

    print("[2/5] Exploration ...")
    summary["exploration"] = run_exploration(ctx)
    print("[3/5] Comparaison des stratégies de discrétisation de mpg ...")
    summary["discretisation"] = run_discretization(ctx)
    print("[4/5] Classification SVC (linéaire, RBF, polynomial) : maison vs scikit-learn ...")
    t = time.perf_counter()
    summary["classification"] = run_classification(ctx)
    print(f"      terminé en {time.perf_counter() - t:.0f} s")
    print("[5/5] Régression SVR linéaire, influence de epsilon, bruit ...")
    t = time.perf_counter()
    summary["regression"] = run_regression(ctx, full_scratch=args.full_scratch)
    print(f"      terminé en {time.perf_counter() - t:.0f} s")

    summary["duree_totale_s"] = round(time.perf_counter() - t_start, 1)
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_jsonable)
    write_results_md(summary)
    print(f"\nTerminé en {summary['duree_totale_s']} s. Résultats : results/RESULTS.md, "
          f"results/tables/*.csv, results/figures/*.png")


def write_results_md(s: dict):
    """Résumé lisible généré automatiquement (les chiffres viennent de CETTE exécution)."""
    c, r = s["classification"], s["regression"]
    L = ["# Résultats (générés automatiquement par `python main.py`)", "",
         f"Environnement : Python {s['environnement']['python']}, NumPy {s['environnement']['numpy']}, "
         f"scikit-learn {s['environnement']['scikit-learn']} — durée {s['duree_totale_s']} s", "",
         f"Données : {s['donnees']['n_clean']} véhicules ({s['donnees']['source']}), "
         f"train {s['donnees']['n_train']} / test {s['donnees']['n_test']}", "",
         "## Discrétisation de mpg", "",
         "| stratégie | seuils (mpg) | effectifs train (élevée/moyenne/faible) | accuracy CV | accuracy équilibrée CV | F1 macro CV |",
         "|---|---|---|---|---|---|"]
    for row in s["discretisation"]["table"]:
        L.append(f"| {row['strategie']} | {row['seuil_1_mpg']:.2f} / {row['seuil_2_mpg']:.2f} | "
                 f"{row['n_train_elevee']} / {row['n_train_moyenne']} / {row['n_train_faible']} | "
                 f"{row['cv_accuracy']:.3f} | {row['cv_balanced_accuracy']:.3f} | {row['cv_f1_macro']:.3f} |")
    L += ["", "## Classification (test, 78 véhicules)", "",
          "| noyau | implémentation | hyperparamètres | accuracy | acc. équilibrée | F1 macro | n VS | accord maison/sklearn |",
          "|---|---|---|---|---|---|---|---|"]
    for row in c["test"]:
        nsv = "-" if row.get("n_vecteurs_support") is None or row["n_vecteurs_support"] != row["n_vecteurs_support"] else int(row["n_vecteurs_support"])
        acc_ = row.get("accord_avec_sklearn_memes_params")
        acc_ = "-" if acc_ is None or acc_ != acc_ else f"{100 * acc_:.0f} %"
        f1 = row["f1_macro_test"]
        f1 = "-" if f1 is None or f1 != f1 else f"{f1:.3f}"
        L.append(f"| {row['noyau']} | {row['implementation']} | {row['hyperparametres']} | {row['accuracy_test']:.3f} | "
                 f"{row['balanced_accuracy_test']:.3f} | {f1} | {nsv} | {acc_} |")
    L += ["", "## Régression (test)", "", f"Hyperparamètres retenus (CV) : {r['cv']['best_scratch']}", "",
          "| modèle | RMSE | MAE | R² | n VS |", "|---|---|---|---|---|"]
    for row in r["test"]:
        nsv = row.get("n_vecteurs_support")
        nsv = "-" if nsv is None or nsv != nsv else int(nsv)
        L.append(f"| {row['modele']} | {row['rmse_test']:.3f} | {row['mae_test']:.3f} | {row['r2_test']:.3f} | {nsv} |")
    L += ["", "## Influence de epsilon (C fixé)", "",
          "| epsilon | n VS maison | n VS sklearn | % VS | % train dans le tube | RMSE CV | RMSE test |",
          "|---|---|---|---|---|---|---|"]
    for row in r["epsilon_study"]:
        L.append(f"| {row['epsilon']:g} | {row['n_SV_maison']} | {row['n_SV_sklearn']} | {row['pct_SV']:.1f} | "
                 f"{row['pct_train_dans_tube']:.1f} | {row['rmse_cv']:.3f} | {row['rmse_test']:.3f} |")
    L += ["", f"Moteur des études de bruit : {r['noise_engine']}", ""]
    with open(ROOT / "results" / "RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

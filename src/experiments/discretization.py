"""Partie 0 (exploration) et Partie 1a (choix de la discrétisation de mpg)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import plotting as P
from ..metrics import accuracy, balanced_accuracy, f1_macro
from ..preprocessing import CLASS_NAMES, NUMERIC_FEATURES, TargetDiscretizer, stratified_kfold
from ..svm_scratch.svc import SVCScratch
from .common import N_FOLDS, SEED, Context, save_table

STRATEGIES = {
    "quantile": "Terciles (effectifs égaux)",
    "uniform": "Largeurs égales",
    "domain": "Seuils métier 20 / 30 mpg",
}


def run_exploration(ctx: Context) -> dict:
    df = ctx.df
    plt = P.plt
    # Corrélations de mpg avec les variables continues (+ cylinders en tant que nombre)
    corr = df[["mpg", "cylinders"] + NUMERIC_FEATURES].corr()["mpg"].drop("mpg")
    save_table(corr.rename("corr_avec_mpg").reset_index().rename(columns={"index": "variable"}),
               "00_correlations_mpg.csv")
    desc = df[["mpg"] + NUMERIC_FEATURES].describe().T.reset_index().rename(columns={"index": "variable"})
    save_table(desc, "00_statistiques_descriptives.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, col in zip(axes, ["weight", "horsepower", "model_year"]):
        ax.scatter(df[col], df["mpg"], s=14, color=P.BLUE, alpha=0.6, edgecolor="white", linewidth=0.4)
        ax.set_xlabel(col)
        ax.set_ylabel("mpg")
        ax.set_title(f"mpg vs {col}  (r = {corr[col]:+.2f})")
    fig.suptitle("Relations entre mpg et quelques variables explicatives (392 véhicules)", y=1.03)
    figs = [P.save(fig, "00_mpg_vs_variables.png")]
    return {"correlations": corr.round(3).to_dict(), "figures": figs,
            "n_cylinders": df["cylinders"].value_counts().sort_index().to_dict(),
            "n_origin": df["origin"].value_counts().to_dict()}


def run_discretization(ctx: Context) -> dict:
    """Compare 3 stratégies avec le MÊME modèle de référence (SVC RBF maison,
    C=1, gamma='scale' : valeurs par défaut, non réglées, pour juger la cible
    elle-même et non le réglage)."""
    plt = P.plt
    rows, thresholds = [], {}
    for strat, label in STRATEGIES.items():
        disc = TargetDiscretizer(strat).fit(ctx.y_train)   # seuils appris sur le train
        t = disc.thresholds_
        thresholds[strat] = t
        c_tr, c_te = disc.transform(ctx.y_train), disc.transform(ctx.y_test)
        counts = np.bincount(c_tr, minlength=3)
        # « zone ambiguë » : véhicules à moins de 1 mpg d'un seuil (difficiles à classer)
        near = np.min(np.abs(ctx.y_train[:, None] - t[None, :]), axis=1) < 1.0
        folds = stratified_kfold(c_tr, N_FOLDS, SEED)
        acc, bacc, f1 = [], [], []
        for tr, va in folds:
            m = SVCScratch(C=1.0, kernel="rbf", gamma="scale").fit(ctx.X_train[tr], c_tr[tr])
            pred = m.predict(ctx.X_train[va])
            acc.append(accuracy(c_tr[va], pred))
            bacc.append(balanced_accuracy(c_tr[va], pred, 3))
            f1.append(f1_macro(c_tr[va], pred, 3))
        majority = np.bincount(c_tr).argmax()
        rows.append({
            "strategie": strat, "description": label,
            "seuil_1_mpg": t[0], "seuil_2_mpg": t[1],
            "seuil_1_L100km": 235.215 / t[0], "seuil_2_L100km": 235.215 / t[1],
            "n_train_elevee": counts[0], "n_train_moyenne": counts[1], "n_train_faible": counts[2],
            "ratio_desequilibre": counts.max() / max(counts.min(), 1),
            "pct_zone_ambigue": 100 * near.mean(),
            "baseline_classe_majoritaire_test": accuracy(c_te, np.full_like(c_te, majority)),
            "cv_accuracy": np.mean(acc), "cv_balanced_accuracy": np.mean(bacc), "cv_f1_macro": np.mean(f1),
        })
    table = pd.DataFrame(rows)
    path = save_table(table, "01_discretisation_comparaison.csv")

    # Figure : histogramme de mpg (train) + seuils des 3 stratégies
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    colors = [P.ORANGE, P.AQUA, P.BLUE]  # élevée / moyenne / faible (identité de classe)
    bins = np.arange(8, 48, 1.0)
    for ax, (strat, label) in zip(axes, STRATEGIES.items()):
        t = thresholds[strat]
        edges = [-np.inf, t[0], t[1], np.inf]
        for k in range(3):
            sel = (ctx.y_train >= edges[k]) & (ctx.y_train < edges[k + 1])
            ax.hist(ctx.y_train[sel], bins=bins, color=colors[k], edgecolor="white",
                    linewidth=1, label=f"{CLASS_NAMES[k]} (n={sel.sum()})")
        for v in t:
            ax.axvline(v, color=P.INK, linewidth=1, linestyle="--")
        ax.set_title(f"{label}\nseuils : {t[0]:.1f} et {t[1]:.1f} mpg")
        ax.set_xlabel("mpg (entraînement)")
        ax.legend(title="consommation", fontsize=8, title_fontsize=8)
    axes[0].set_ylabel("nombre de véhicules")
    fig2 = P.save(fig, "01_discretisation_histogrammes.png")
    return {"table": table.round(4).to_dict(orient="records"), "files": [path, fig2]}

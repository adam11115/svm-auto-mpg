"""Partie 1b : classification (SVC linéaire, RBF, polynomial) — maison vs scikit-learn."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from .. import plotting as P
from ..metrics import accuracy, balanced_accuracy, confusion_matrix, f1_macro
from ..model_selection import grid_search_cv
from ..preprocessing import CLASS_NAMES, TargetDiscretizer, stratified_kfold
from ..svm_scratch.primal import LinearSVCPrimal
from ..svm_scratch.svc import OneVsOneClassifier, SVCScratch
from .common import N_FOLDS, SEED, Context, save_table

STRATEGY = "quantile"  # stratégie retenue à l'issue de la partie 1a

# Grilles d'hyperparamètres (échelle logarithmique ; voir README pour la justification)
GRIDS = {
    "linear": {"C": [0.001, 0.01, 0.1, 1, 10, 100]},
    "rbf": {"C": [0.1, 1, 10, 100, 1000], "gamma": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
    "poly": {"C": [0.1, 1, 10, 100, 1000], "degree": [2, 3, 4], "gamma": [0.01, 0.03, 0.1, 0.3]},
}
POLY_COEF0 = 1.0  # noyau (gamma <x,z> + 1)^d : contient aussi les termes de degré < d


def _make(kernel: str, impl: str):
    def factory(**params):
        kw = dict(params)
        if kernel == "poly":
            kw["coef0"] = POLY_COEF0
        if impl == "scratch":
            return SVCScratch(kernel=kernel, **kw)
        return SVC(kernel=kernel, **kw)   # sklearn : même problème, même tol=1e-3
    return factory


def _n_sv(model) -> int:
    return model.n_support_total_ if hasattr(model, "n_support_total_") else int(model.support_.size)


def run_classification(ctx: Context) -> dict:
    plt = P.plt
    disc = TargetDiscretizer(STRATEGY).fit(ctx.y_train)
    c_tr, c_te = disc.transform(ctx.y_train), disc.transform(ctx.y_test)
    folds = stratified_kfold(c_tr, N_FOLDS, SEED)
    Xtr, Xte = ctx.X_train, ctx.X_test

    out = {"thresholds": disc.thresholds_.round(3).tolist(),
           "n_train_by_class": np.bincount(c_tr).tolist(), "n_test_by_class": np.bincount(c_te).tolist(),
           "cv": {}, "test": [], "files": []}
    test_rows, preds = [], {}

    for kernel, grid in GRIDS.items():
        cv_tables, best = {}, {}
        for impl in ("scratch", "sklearn"):
            table, best[impl] = grid_search_cv(_make(kernel, impl), grid, Xtr, c_tr, folds, accuracy)
            cv_tables[impl] = table
        merged = cv_tables["scratch"].merge(cv_tables["sklearn"], on=list(grid),
                                            suffixes=("_scratch", "_sklearn"))
        merged["ecart_abs"] = (merged["cv_mean_scratch"] - merged["cv_mean_sklearn"]).abs()
        out["files"].append(save_table(merged, f"02_cv_{kernel}.csv"))
        out["cv"][kernel] = {"best_scratch": best["scratch"], "best_sklearn": best["sklearn"],
                             "cv_acc_retenu_scratch": float(merged.loc[merged["retenu_scratch"], "cv_mean_scratch"].iloc[0]),
                             "cv_acc_max_scratch": float(merged["cv_mean_scratch"].max()),
                             "params_max_brut_scratch": str({k: merged.loc[merged["meilleur_brut_scratch"], k].iloc[0].item() for k in grid}),
                             "cv_acc_retenu_sklearn": float(merged.loc[merged["retenu_sklearn"], "cv_mean_sklearn"].iloc[0]),
                             "max_ecart_cv": float(merged["ecart_abs"].max())}
        if kernel == "rbf":
            pivot = cv_tables["scratch"].pivot(index="C", columns="gamma", values="cv_mean")
            fig, ax = plt.subplots(figsize=(5.6, 3.8))
            im = ax.imshow(pivot.values, cmap=P.SEQ_CMAP, aspect="auto", vmin=0.5, vmax=0.9)
            ax.set_xticks(range(pivot.shape[1]), [f"{g:g}" for g in pivot.columns])
            ax.set_yticks(range(pivot.shape[0]), [f"{c:g}" for c in pivot.index])
            ax.set_xlabel("gamma")
            ax.set_ylabel("C")
            ax.grid(False)
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    v = pivot.values[i, j]
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                            color="white" if v > 0.5 + 0.5 * (0.9 - 0.5) else P.INK)
            fig.colorbar(im, ax=ax, label="accuracy moyenne (CV 5 plis)")
            ax.set_title("SVC RBF maison : accuracy en validation croisée")
            out["files"].append(P.save(fig, "02_cv_rbf_heatmap.png"))

        # Ré-entraînement sur tout le train avec les MEILLEURS paramètres de chaque implémentation
        for impl in ("scratch", "sklearn"):
            model = _make(kernel, impl)(**best[impl])
            t0 = time.perf_counter()
            model.fit(Xtr, c_tr)
            fit_t = time.perf_counter() - t0
            pred = model.predict(Xte)
            preds[(kernel, impl)] = pred
            test_rows.append({"noyau": kernel, "implementation": impl,
                              "hyperparametres": str(best[impl]),
                              "accuracy_test": accuracy(c_te, pred),
                              "balanced_accuracy_test": balanced_accuracy(c_te, pred, 3),
                              "f1_macro_test": f1_macro(c_te, pred, 3),
                              "n_vecteurs_support": _n_sv(model), "temps_fit_s": fit_t})
        # accord maison / sklearn à hyperparamètres IDENTIQUES (ceux retenus par la version maison)
        same = _make(kernel, "sklearn")(**best["scratch"]).fit(Xtr, c_tr).predict(Xte)
        test_rows[-2]["accord_avec_sklearn_memes_params"] = float(np.mean(same == preds[(kernel, "scratch")]))

    # Variante PRIMALE (sous-gradient, perte charnière explicite), avec le C du linéaire maison
    C_lin = out["cv"]["linear"]["best_scratch"]["C"]
    prim = OneVsOneClassifier(lambda: LinearSVCPrimal(C=C_lin))
    t0 = time.perf_counter()
    prim.fit(Xtr, c_tr)
    fit_t = time.perf_counter() - t0
    pred = prim.predict(Xte)
    test_rows.append({"noyau": "linear", "implementation": "scratch_primal_sous_gradient",
                      "hyperparametres": str({"C": C_lin}), "accuracy_test": accuracy(c_te, pred),
                      "balanced_accuracy_test": balanced_accuracy(c_te, pred, 3),
                      "f1_macro_test": f1_macro(c_te, pred, 3), "n_vecteurs_support": np.nan,
                      "temps_fit_s": fit_t,
                      "accord_avec_sklearn_memes_params": float(np.mean(
                          pred == SVC(kernel="linear", C=C_lin).fit(Xtr, c_tr).predict(Xte)))})
    # Référence naïve
    maj = np.bincount(c_tr).argmax()
    test_rows.append({"noyau": "-", "implementation": "baseline_classe_majoritaire",
                      "hyperparametres": "-", "accuracy_test": accuracy(c_te, np.full_like(c_te, maj)),
                      "balanced_accuracy_test": 1 / 3, "f1_macro_test": np.nan,
                      "n_vecteurs_support": np.nan, "temps_fit_s": np.nan})
    test = pd.DataFrame(test_rows)
    out["files"].append(save_table(test, "02_classification_test.csv"))
    out["test"] = test.round(4).to_dict(orient="records")

    # Matrices de confusion (implémentation maison)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, kernel in zip(axes, GRIDS):
        M = confusion_matrix(c_te, preds[(kernel, "scratch")], 3)
        ax.imshow(M, cmap=P.SEQ_CMAP)
        ax.grid(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, M[i, j], ha="center", va="center",
                        color="white" if M[i, j] > M.max() / 2 else P.INK, fontsize=11)
        ax.set_xticks(range(3), CLASS_NAMES)
        ax.set_yticks(range(3), CLASS_NAMES)
        ax.set_xlabel("classe prédite")
        acc = accuracy(c_te, preds[(kernel, "scratch")])
        ax.set_title(f"SVC {kernel} (maison) — accuracy test {acc:.3f}")
        out.setdefault("confusion", {})[kernel] = M.tolist()
    axes[0].set_ylabel("vraie classe")
    fig.subplots_adjust(wspace=0.35)
    out["files"].append(P.save(fig, "02_matrices_confusion.png"))

    # Barres : accuracy test maison vs sklearn
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(GRIDS))
    for k, (impl, color, lab) in enumerate([("scratch", P.BLUE, "maison (NumPy)"),
                                            ("sklearn", P.ORANGE, "scikit-learn")]):
        vals = [test.loc[(test["noyau"] == kn) & (test["implementation"] == impl),
                         "accuracy_test"].iloc[0] for kn in GRIDS]
        bars = ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=color, label=lab)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, color=P.INK2)
    ax.axhline(test.iloc[-1]["accuracy_test"], color=P.INK2, linestyle=":", linewidth=1.2,
               label="classe majoritaire")
    ax.set_xticks(x, list(GRIDS))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy sur le test (78 véhicules)")
    ax.set_title("Classification : maison vs scikit-learn")
    ax.legend(loc="lower right", fontsize=8)
    out["files"].append(P.save(fig, "02_accuracy_maison_vs_sklearn.png"))

    out["files"].append(_decision_regions(ctx, disc))
    return out


def _decision_regions(ctx: Context, disc) -> str:
    """Illustration 2D (poids, année) : frontière linéaire vs RBF. Visualisation seulement."""
    plt = P.plt
    from ..preprocessing import StandardScalerScratch
    cols = ["weight", "model_year"]
    tr = ctx.df.iloc[ctx.idx_train]
    sc = StandardScalerScratch().fit(tr[cols].to_numpy(float))
    X2 = sc.transform(tr[cols].to_numpy(float))
    c = disc.transform(tr["mpg"].to_numpy(float))
    gx, gy = np.meshgrid(np.linspace(X2[:, 0].min() - .3, X2[:, 0].max() + .3, 250),
                         np.linspace(X2[:, 1].min() - .3, X2[:, 1].max() + .3, 250))
    grid = np.c_[gx.ravel(), gy.ravel()]
    colors = [P.ORANGE, P.AQUA, P.BLUE]
    from matplotlib.colors import ListedColormap
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (title, model) in zip(axes, [("noyau linéaire (C=1)", SVCScratch(C=1, kernel="linear")),
                                         ("noyau RBF (C=10, gamma=1)", SVCScratch(C=10, kernel="rbf", gamma=1.0))]):
        model.fit(X2, c)
        Z = model.predict(grid).reshape(gx.shape)
        ax.contourf(gx, gy, Z, levels=[-0.5, 0.5, 1.5, 2.5], cmap=ListedColormap(colors), alpha=0.18)
        for k in range(3):
            ax.scatter(X2[c == k, 0], X2[c == k, 1], s=16, color=colors[k], edgecolor="white",
                       linewidth=0.5, label=CLASS_NAMES[k])
        ax.set_title(f"Frontières de décision — {title}")
        ax.set_xlabel("weight (standardisé)")
        ax.grid(False)
    axes[0].set_ylabel("model_year (standardisé)")
    axes[1].legend(title="consommation", fontsize=8, loc="upper right")
    return P.save(fig, "02_frontieres_2D.png")

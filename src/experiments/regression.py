"""Partie 2 : régression (SVR linéaire, perte epsilon-insensible) — maison vs scikit-learn,
influence de epsilon sur les vecteurs de support et sur la robustesse au bruit."""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVR, LinearSVR

from .. import plotting as P
from ..metrics import mae, r2, rmse
from ..model_selection import grid_search_cv
from ..preprocessing import kfold
from ..svm_scratch.primal import LinearSVRPrimal
from ..svm_scratch.svr import SVRScratch
from .common import N_FOLDS, SEED, Context, save_table

GRID = {"C": [0.01, 0.1, 1, 10, 100], "epsilon": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]}
EPS_STUDY = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10]
NOISE_SIGMAS = [0.0, 1.0, 2.0, 4.0]       # écart-type du bruit gaussien ajouté (en mpg)
NOISE_EPS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]
OUTLIER_RATES = [0.0, 0.05, 0.10, 0.20]   # part des véhicules d'entraînement corrompus
OUTLIER_SHIFT = 15.0                      # erreur grossière ajoutée (mpg), ex. erreur de saisie
N_REPEATS = 20                            # tirages aléatoires par configuration


def _ols(X, y):
    """Moindres carrés ordinaires (référence, perte quadratique) — depuis zéro."""
    A = np.c_[np.ones(len(X)), X]
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def _ols_predict(beta, X):
    return beta[0] + X @ beta[1:]


def run_regression(ctx: Context, full_scratch: bool = False) -> dict:
    """full_scratch=False : les études de bruit (560 entraînements répétés) utilisent
    sklearn.svm.SVR(kernel='linear'), qui résout EXACTEMENT le même problème que
    SVRScratch (vérifié plus haut : mêmes vecteurs de support, mêmes prédictions)
    mais en C compilé. full_scratch=True : tout est fait avec SVRScratch (plus lent)."""
    plt = P.plt
    if full_scratch:
        def fast_svr(**p):
            return SVRScratch(kernel="linear", **p)
    else:
        def fast_svr(**p):
            return SVR(kernel="linear", **p)
    Xtr, Xte, ytr, yte = ctx.X_train, ctx.X_test, ctx.y_train, ctx.y_test
    folds = kfold(len(ytr), N_FOLDS, SEED)
    out = {"files": [], "noise_engine": "SVRScratch (maison)" if full_scratch else "sklearn.svm.SVR(kernel='linear')"}

    # ---------------------------------------------------------- 1. réglage (C, epsilon)
    tables, best = {}, {}
    makers = {"scratch": lambda **p: SVRScratch(kernel="linear", **p),
              "sklearn": lambda **p: SVR(kernel="linear", **p)}
    for impl, mk in makers.items():
        tables[impl], best[impl] = grid_search_cv(mk, GRID, Xtr, ytr, folds, rmse, greater_is_better=False)
    merged = tables["scratch"].merge(tables["sklearn"], on=list(GRID), suffixes=("_scratch", "_sklearn"))
    merged["ecart_abs"] = (merged["cv_mean_scratch"] - merged["cv_mean_sklearn"]).abs()
    out["files"].append(save_table(merged.rename(columns=lambda c: c.replace("cv_mean", "cv_rmse")),
                                   "03_cv_svr.csv"))
    C_best, eps_best = best["scratch"]["C"], best["scratch"]["epsilon"]
    out["cv"] = {"best_scratch": best["scratch"], "best_sklearn": best["sklearn"],
                 "cv_rmse_retenu_scratch": float(merged.loc[merged["retenu_scratch"], "cv_mean_scratch"].iloc[0]),
                 "cv_rmse_min_scratch": float(merged["cv_mean_scratch"].min()),
                 "params_min_brut_scratch": str({k: merged.loc[merged["meilleur_brut_scratch"], k].iloc[0].item() for k in GRID}),
                 "max_ecart_cv": float(merged["ecart_abs"].max())}

    pivot = tables["scratch"].pivot(index="C", columns="epsilon", values="cv_mean")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    lo, hi = float(np.nanmin(pivot.values)), float(np.nanmax(pivot.values))
    im = ax.imshow(pivot.values, cmap=P.SEQ_CMAP + "_r", aspect="auto", vmin=lo, vmax=hi)
    ax.set_xticks(range(pivot.shape[1]), [f"{e:g}" for e in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), [f"{c:g}" for c in pivot.index])
    ax.grid(False)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v < lo + 0.5 * (hi - lo) else P.INK)
    ax.set_xlabel("epsilon (mpg)")
    ax.set_ylabel("C")
    fig.colorbar(im, ax=ax, label="RMSE moyenne CV (mpg) — plus foncé = meilleur")
    ax.set_title("SVR linéaire maison : RMSE en validation croisée")
    out["files"].append(P.save(fig, "03_cv_svr_heatmap.png"))

    # ---------------------------------------------------------- 2. comparaison sur le test
    rows, models = [], {}
    candidates = {
        "scratch_dual_SMO": lambda: SVRScratch(C=C_best, epsilon=eps_best),
        "scratch_primal_sous_gradient": lambda: LinearSVRPrimal(C=C_best, epsilon=eps_best),
        "sklearn_SVR_kernel_linear": lambda: SVR(kernel="linear", C=C_best, epsilon=eps_best),
        "sklearn_LinearSVR": lambda: LinearSVR(C=C_best, epsilon=eps_best, loss="epsilon_insensitive",
                                               max_iter=200_000, random_state=SEED),
    }
    for name, mk in candidates.items():
        m = mk()
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m.fit(Xtr, ytr)
        fit_t = time.perf_counter() - t0
        pred = m.predict(Xte)
        models[name] = m
        coef = np.ravel(m.coef_)
        intercept = float(np.ravel(m.intercept_)[0]) if np.ndim(m.intercept_) else float(m.intercept_)
        rows.append({"modele": name, "C": C_best, "epsilon": eps_best,
                     "rmse_test": rmse(yte, pred), "mae_test": mae(yte, pred), "r2_test": r2(yte, pred),
                     "n_vecteurs_support": int(m.support_.size) if hasattr(m, "support_") else np.nan,
                     "objectif_primal_train": 0.5 * coef @ coef + C_best * np.maximum(
                         0, np.abs(ytr - (Xtr @ coef + intercept)) - eps_best).sum(),
                     "temps_fit_s": fit_t})
    beta = _ols(Xtr, ytr)
    p_ols = _ols_predict(beta, Xte)
    rows.append({"modele": "reference_moindres_carres_OLS", "C": np.nan, "epsilon": np.nan,
                 "rmse_test": rmse(yte, p_ols), "mae_test": mae(yte, p_ols), "r2_test": r2(yte, p_ols),
                 "n_vecteurs_support": np.nan, "objectif_primal_train": np.nan, "temps_fit_s": np.nan})
    rows.append({"modele": "reference_moyenne_constante", "C": np.nan, "epsilon": np.nan,
                 "rmse_test": rmse(yte, np.full_like(yte, ytr.mean())),
                 "mae_test": mae(yte, np.full_like(yte, ytr.mean())), "r2_test": r2(yte, np.full_like(yte, ytr.mean())),
                 "n_vecteurs_support": np.nan, "objectif_primal_train": np.nan, "temps_fit_s": np.nan})
    test = pd.DataFrame(rows)
    out["files"].append(save_table(test, "03_regression_test.csv"))
    out["test"] = test.round(4).to_dict(orient="records")
    dual = models["scratch_dual_SMO"]
    out["dual_objective_scratch"] = float(dual.dual_objective_)
    out["max_abs_pred_diff_scratch_vs_sklearn"] = float(np.max(np.abs(
        dual.predict(Xte) - models["sklearn_SVR_kernel_linear"].predict(Xte))))
    out["ols_train_residual_std"] = float(np.std(ytr - _ols_predict(beta, Xtr)))

    coefs = pd.DataFrame({"variable": ctx.prep.feature_names_,
                          **{k: np.ravel(models[k].coef_) for k in candidates},
                          "OLS": beta[1:]})
    out["files"].append(save_table(coefs, "03_coefficients.csv"))
    out["coefficients"] = coefs.round(3).to_dict(orient="records")
    fig, ax = plt.subplots(figsize=(8, 4))
    yy = np.arange(len(coefs))
    ax.barh(yy + 0.2, coefs["scratch_dual_SMO"], height=0.38, color=P.BLUE, label="maison (dual SMO)")
    ax.barh(yy - 0.2, coefs["sklearn_SVR_kernel_linear"], height=0.38, color=P.ORANGE, label="scikit-learn SVR")
    ax.set_yticks(yy, coefs["variable"])
    ax.invert_yaxis()
    ax.axvline(0, color=P.INK2, linewidth=1)
    ax.set_xlabel("coefficient w_j (mpg par écart-type, ou par présence de la modalité)")
    ax.set_title(f"SVR linéaire (C={C_best:g}, epsilon={eps_best:g}) : coefficients appris")
    ax.legend(fontsize=8)
    out["files"].append(P.save(fig, "03_coefficients.png"))

    # ---------------------------------------------------------- 3. influence de epsilon
    rows = []
    tube_models = {}
    for eps in EPS_STUDY:
        m = SVRScratch(C=C_best, epsilon=eps).fit(Xtr, ytr)
        s = SVR(kernel="linear", C=C_best, epsilon=eps).fit(Xtr, ytr)
        res = ytr - m.predict(Xtr)
        coef_abs = np.abs(m.dual_coef_)
        cv = [rmse(ytr[va], SVRScratch(C=C_best, epsilon=eps).fit(Xtr[tr], ytr[tr]).predict(Xtr[va]))
              for tr, va in folds]
        pred = m.predict(Xte)
        rows.append({"epsilon": eps,
                     "n_SV_maison": m.support_.size, "n_SV_sklearn": s.support_.size,
                     "pct_SV": 100 * m.support_.size / len(ytr),
                     "n_SV_bornes_hors_tube": int(np.sum(coef_abs >= C_best - 1e-9)),
                     "n_SV_libres_sur_bord": int(np.sum(coef_abs < C_best - 1e-9)),
                     # strictement dans le tube : |r| < eps - 1e-3 (1e-3 = tolérance du solveur ;
                     # les VS « libres » sont SUR le bord, à +/- 1e-3 mpg près)
                     "pct_train_dans_tube": 100 * np.mean(np.abs(res) < eps - 1e-3),
                     "rmse_cv": float(np.mean(cv)), "rmse_test": rmse(yte, pred),
                     "mae_test": mae(yte, pred), "r2_test": r2(yte, pred),
                     "norme_w": float(np.linalg.norm(m.coef_))})
        tube_models[eps] = m
    eps_table = pd.DataFrame(rows)
    out["files"].append(save_table(eps_table, "03_influence_epsilon.csv"))
    out["epsilon_study"] = eps_table.round(4).to_dict(orient="records")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.plot(eps_table["epsilon"], eps_table["n_SV_maison"], color=P.BLUE, marker="o", label="maison (dual SMO)")
    ax.plot(eps_table["epsilon"], eps_table["n_SV_sklearn"], color=P.ORANGE, marker="x", markersize=8,
            linestyle="none", label="scikit-learn SVR")
    ax.set_xlabel("epsilon (mpg)")
    ax.set_ylabel(f"nombre de vecteurs de support (sur {len(ytr)})")
    ax.set_title("Plus epsilon est grand, moins il y a de vecteurs de support")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(eps_table["epsilon"], eps_table["rmse_test"], color=P.BLUE, marker="o", label="RMSE test")
    ax.plot(eps_table["epsilon"], eps_table["rmse_cv"], color=P.AQUA, marker="s", label="RMSE validation croisée")
    ax.axvline(eps_best, color=P.INK2, linestyle=":", linewidth=1.2, label=f"epsilon retenu = {eps_best:g}")
    ax.set_xlabel("epsilon (mpg)")
    ax.set_ylabel("RMSE (mpg)")
    ax.set_title("Erreur de prédiction en fonction de epsilon")
    ax.legend(fontsize=8)
    out["files"].append(P.save(fig, "03_influence_epsilon.png"))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, eps in zip(axes, [0.5, 3, 6]):
        m = tube_models[eps]
        fitted = m.predict(Xtr)
        res = ytr - fitted
        is_sv = np.zeros(len(ytr), dtype=bool)
        is_sv[m.support_] = True
        ax.axhspan(-eps, eps, color=P.AQUA, alpha=0.15, label=f"tube ±{eps:g} mpg (perte nulle)")
        ax.scatter(fitted[~is_sv], res[~is_sv], s=14, color=P.INK2, alpha=0.5,
                   label=f"dans le tube : ignorés ({(~is_sv).sum()})")
        ax.scatter(fitted[is_sv], res[is_sv], s=16, color=P.ORANGE, edgecolor="white", linewidth=0.4,
                   label=f"vecteurs de support ({is_sv.sum()})")
        ax.axhline(0, color=P.INK, linewidth=1)
        ax.set_title(f"epsilon = {eps:g}")
        ax.set_xlabel("mpg prédit (entraînement)")
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("résidu  y - f(x)  (mpg)")
    fig.suptitle("Le tube epsilon : seuls les points sur son bord ou en dehors portent le modèle", y=1.02)
    out["files"].append(P.save(fig, "03_tube_epsilon.png"))

    # ---------------------------------------------------------- 4. robustesse au bruit
    rng = np.random.default_rng(SEED)
    rows = []
    for sigma in NOISE_SIGMAS:
        noises = [rng.normal(0.0, sigma, size=len(ytr)) for _ in range(N_REPEATS)]
        for eps in NOISE_EPS:
            r_test, n_sv, ws = [], [], []
            for z in noises:
                m = fast_svr(C=C_best, epsilon=eps).fit(Xtr, ytr + z)
                r_test.append(rmse(yte, m.predict(Xte)))   # évalué sur les VRAIES valeurs du test
                n_sv.append(m.support_.size)
                ws.append(np.ravel(m.coef_))
            rows.append({"sigma_bruit": sigma, "modele": "SVR", "epsilon": eps,
                         "rmse_test_moy": np.mean(r_test), "rmse_test_std": np.std(r_test),
                         "pct_SV_moy": 100 * np.mean(n_sv) / len(ytr),
                         "instabilite_coef": float(np.mean(np.std(ws, axis=0)))})
        r_test, ws = [], []
        for z in noises:
            b = _ols(Xtr, ytr + z)
            r_test.append(rmse(yte, _ols_predict(b, Xte)))
            ws.append(b[1:])
        rows.append({"sigma_bruit": sigma, "modele": "OLS", "epsilon": np.nan,
                     "rmse_test_moy": np.mean(r_test), "rmse_test_std": np.std(r_test),
                     "pct_SV_moy": np.nan, "instabilite_coef": float(np.mean(np.std(ws, axis=0)))})
    noise = pd.DataFrame(rows)
    out["files"].append(save_table(noise, "03_bruit_gaussien.csv"))
    out["noise_study"] = noise.round(4).to_dict(orient="records")

    rows = []
    for rate in OUTLIER_RATES:
        for rep in range(N_REPEATS):
            y_bad = ytr.copy()
            k = int(round(rate * len(ytr)))
            idx = rng.choice(len(ytr), size=k, replace=False)
            y_bad[idx] += OUTLIER_SHIFT
            m = fast_svr(C=C_best, epsilon=eps_best).fit(Xtr, y_bad)
            b = _ols(Xtr, y_bad)
            rows.append({"taux_aberrants": rate, "rep": rep,
                         "rmse_test_SVR": rmse(yte, m.predict(Xte)),
                         "rmse_test_OLS": rmse(yte, _ols_predict(b, Xte))})
    raw = pd.DataFrame(rows)
    outl = pd.DataFrame([{"taux_aberrants": rate,
                          "rmse_SVR_moy": g["rmse_test_SVR"].mean(), "rmse_SVR_std": g["rmse_test_SVR"].std(ddof=0),
                          "rmse_OLS_moy": g["rmse_test_OLS"].mean(), "rmse_OLS_std": g["rmse_test_OLS"].std(ddof=0)}
                         for rate, g in raw.groupby("taux_aberrants")])
    out["files"].append(save_table(outl, "03_valeurs_aberrantes.csv"))
    out["outlier_study"] = outl.round(4).to_dict(orient="records")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    shades = ["#b7d3f2", "#6fa6e6", "#2a78d6", "#174f94"]  # une teinte, du clair au foncé = bruit croissant
    svr_rows = noise[noise["modele"] == "SVR"]
    for sigma, col in zip(NOISE_SIGMAS, shades):
        sub = svr_rows[svr_rows["sigma_bruit"] == sigma]
        axes[0].plot(sub["epsilon"], sub["pct_SV_moy"], marker="o", color=col, label=f"sigma = {sigma:g}")
        axes[1].plot(sub["epsilon"], sub["rmse_test_moy"], marker="o", color=col, label=f"sigma = {sigma:g}")
    axes[0].set_xlabel("epsilon (mpg)")
    axes[0].set_ylabel("% de vecteurs de support")
    axes[0].set_title("Bruit ajouté au train : plus de points sortent du tube")
    axes[1].set_xlabel("epsilon (mpg)")
    axes[1].set_ylabel("RMSE sur le test non bruité (mpg)")
    axes[1].set_title(f"Erreur de test (moyenne sur {N_REPEATS} tirages)")
    for ax in axes[:2]:
        ax.legend(title="bruit gaussien", fontsize=8, title_fontsize=8)
    ax = axes[2]
    ax.errorbar(100 * outl["taux_aberrants"], outl["rmse_OLS_moy"], yerr=outl["rmse_OLS_std"],
                color=P.ORANGE, marker="s", capsize=3, label="moindres carrés (perte quadratique)")
    ax.errorbar(100 * outl["taux_aberrants"], outl["rmse_SVR_moy"], yerr=outl["rmse_SVR_std"],
                color=P.BLUE, marker="o", capsize=3, label=f"SVR (epsilon = {eps_best:g})")
    ax.set_xlabel(f"% de véhicules d'entraînement avec mpg faussé de +{OUTLIER_SHIFT:g}")
    ax.set_ylabel("RMSE sur le test (mpg)")
    ax.set_title("Valeurs aberrantes : la perte epsilon-insensible résiste")
    ax.legend(fontsize=8)
    out["files"].append(P.save(fig, "03_bruit_et_aberrants.png"))
    return out

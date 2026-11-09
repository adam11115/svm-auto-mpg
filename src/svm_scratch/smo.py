"""Solveur SMO (Sequential Minimal Optimization) écrit depuis zéro avec NumPy.

Il résout le problème quadratique « générique » utilisé par LIBSVM (et donc
par sklearn.svm.SVC / sklearn.svm.SVR) :

        min_a   f(a) = 1/2 a^T Q a + p^T a
        s.c.    y^T a = constante        (y_t dans {-1, +1})
                0 <= a_t <= C

* C-SVC      : Q_ij = y_i y_j K(x_i, x_j),  p = -1,  y = étiquettes (+1/-1)
* epsilon-SVR: 2n variables (a, a*), voir svr.py pour la construction de Q et p.

Principe de SMO : à chaque itération on choisit DEUX variables (a_i, a_j)
(il en faut au moins deux pour respecter la contrainte d'égalité y^T a = cte),
on résout analytiquement le sous-problème 2D, puis on met à jour le gradient.

Sélection de la paire (i, j) : règle « WSS2 » du second ordre de
Fan, Chen & Lin (2005), « Working Set Selection Using Second Order
Information for Training SVM », JMLR 6 — c'est la règle de LIBSVM.

Critère d'arrêt : m(a) - M(a) < tol (violation maximale des conditions KKT),
avec tol = 1e-3 comme la valeur par défaut de scikit-learn.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

TAU = 1e-12  # remplace une courbure <= 0 (noyau non défini positif / doublons)


@dataclass
class SMOResult:
    alpha: np.ndarray      # solution duale
    rho: float             # seuil : f(x) = sum(coef_i K(x_i, x)) - rho
    n_iter: int            # nombre d'itérations SMO effectuées
    converged: bool        # True si le critère KKT a été atteint
    objective: float       # valeur de 1/2 a^T Q a + p^T a à la solution
    gap: float             # m(a) - M(a) final (violation KKT)


def smo_solve(Q: np.ndarray, p: np.ndarray, y: np.ndarray, C: float,
              tol: float = 1e-3, max_iter: int = 1_000_000) -> SMOResult:
    """Résout le QP dual ci-dessus. Départ : a = 0 (toujours admissible)."""
    n = p.shape[0]
    y = y.astype(float)
    alpha = np.zeros(n)
    G = p.astype(float).copy()     # gradient : G = Q a + p, et a = 0 au départ
    QD = np.diag(Q).copy()         # diagonale de Q (Q_ii)

    converged = False
    gap = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        # --- 1. Ensembles I_up / I_low (variables qu'on peut augmenter/diminuer
        #        dans la direction y_t sans violer 0 <= a_t <= C)
        pos = y > 0
        up = (pos & (alpha < C)) | (~pos & (alpha > 0))
        low = (pos & (alpha > 0)) | (~pos & (alpha < C))

        v = -y * G                                  # -y_t * grad_t
        v_up = np.where(up, v, -np.inf)
        i = int(np.argmax(v_up))
        g_max = v_up[i]                             # m(a)
        g_min = np.min(np.where(low, v, np.inf))    # M(a)
        gap = g_max - g_min
        if gap < tol:                               # conditions KKT satisfaites
            converged = True
            break

        # --- 2. Choix de j : gain maximal de l'objectif (information du 2nd ordre)
        b = g_max - v                               # > 0 pour les candidats
        a = QD[i] + QD - 2.0 * y[i] * y * Q[i]      # courbure = K_ii + K_jj - 2 K_ij
        a = np.where(a > 0, a, TAU)
        cand = low & (b > 0)
        j = int(np.argmin(np.where(cand, -(b * b) / a, np.inf)))

        # --- 3. Résolution analytique du sous-problème à 2 variables
        #        (transcription fidèle de Solver::Solve de LIBSVM)
        Ci = Cj = C
        old_ai, old_aj = alpha[i], alpha[j]
        Qij = Q[i, j]
        if y[i] != y[j]:
            quad = QD[i] + QD[j] + 2.0 * Qij
            if quad <= 0:
                quad = TAU
            delta = (-G[i] - G[j]) / quad
            diff = alpha[i] - alpha[j]
            alpha[i] += delta
            alpha[j] += delta
            if diff > 0:
                if alpha[j] < 0:
                    alpha[j] = 0.0
                    alpha[i] = diff
            else:
                if alpha[i] < 0:
                    alpha[i] = 0.0
                    alpha[j] = -diff
            if diff > Ci - Cj:
                if alpha[i] > Ci:
                    alpha[i] = Ci
                    alpha[j] = Ci - diff
            else:
                if alpha[j] > Cj:
                    alpha[j] = Cj
                    alpha[i] = Cj + diff
        else:
            quad = QD[i] + QD[j] - 2.0 * Qij
            if quad <= 0:
                quad = TAU
            delta = (G[i] - G[j]) / quad
            s = alpha[i] + alpha[j]
            alpha[i] -= delta
            alpha[j] += delta
            if s > Ci:
                if alpha[i] > Ci:
                    alpha[i] = Ci
                    alpha[j] = s - Ci
            else:
                if alpha[j] < 0:
                    alpha[j] = 0.0
                    alpha[i] = s
            if s > Cj:
                if alpha[j] > Cj:
                    alpha[j] = Cj
                    alpha[i] = s - Cj
            else:
                if alpha[i] < 0:
                    alpha[i] = 0.0
                    alpha[j] = s

        # --- 4. Mise à jour incrémentale du gradient (O(n) au lieu de O(n^2))
        d_ai = alpha[i] - old_ai
        d_aj = alpha[j] - old_aj
        G += Q[i] * d_ai + Q[j] * d_aj
    else:
        warnings.warn(f"SMO : max_iter={max_iter} atteint sans convergence "
                      f"(violation KKT = {gap:.2e})", RuntimeWarning)

    rho = _compute_rho(alpha, G, y, C)
    objective = 0.5 * float(alpha @ (G + p))   # = 1/2 a^T Q a + p^T a
    return SMOResult(alpha=alpha, rho=rho, n_iter=it, converged=converged,
                     objective=objective, gap=float(gap))


def _compute_rho(alpha: np.ndarray, G: np.ndarray, y: np.ndarray, C: float) -> float:
    """Calcul du seuil rho (= -b), comme LIBSVM (Solver::calculate_rho).

    Pour les variables « libres » (0 < a_t < C) les conditions KKT imposent
    y_t G_t = rho : on fait la moyenne. S'il n'y en a aucune, rho est pris au
    milieu de l'intervalle admissible [lb, ub].
    """
    yG = y * G
    at_upper = alpha >= C
    at_lower = alpha <= 0
    free = ~at_upper & ~at_lower
    if np.any(free):
        return float(np.mean(yG[free]))
    pos = y > 0
    ub_mask = (at_upper & ~pos) | (at_lower & pos)
    lb_mask = (at_upper & pos) | (at_lower & ~pos)
    ub = np.min(yG[ub_mask]) if np.any(ub_mask) else np.inf
    lb = np.max(yG[lb_mask]) if np.any(lb_mask) else -np.inf
    return float((ub + lb) / 2.0)

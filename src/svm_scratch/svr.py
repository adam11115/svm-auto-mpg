"""Régression à vecteurs de support (epsilon-SVR) implémentée depuis zéro.

Perte epsilon-insensible :  L_eps(y, f(x)) = max(0, |y - f(x)| - eps)
    -> les erreurs plus petites que eps (à l'intérieur du « tube ») coûtent 0.

Problème primal :
    min_{w,b}  1/2 ||w||^2 + C * sum_i max(0, |y_i - (w . x_i + b)| - eps)

Problème dual (2n variables a_i, a_i*) :
    min  1/2 (a - a*)^T K (a - a*) + eps * sum(a + a*) - sum y_i (a_i - a_i*)
    s.c. sum (a_i - a_i*) = 0,   0 <= a_i, a_i* <= C

On l'écrit sous la forme générique de smo.py avec la variable empilée
beta = [a ; a*] (longueur 2n), z = [+1 ... +1, -1 ... -1] :
    Q = [[K, -K], [-K, K]]   (Q_st = z_s z_t K_st),
    p = [eps - y ; eps + y].

Prédiction : f(x) = sum_i (a_i - a_i*) K(x_i, x) + b.
Vecteurs de support : points avec a_i - a_i* != 0, c'est-à-dire les points
SUR le bord du tube ou EN DEHORS. Les points strictement à l'intérieur du tube
ont a_i = a_i* = 0 : ils n'influencent pas le modèle (c'est ainsi que le
modèle « ignore » les petites erreurs / le bruit).
"""
from __future__ import annotations

import numpy as np

from .kernels import compute_kernel, resolve_gamma
from .smo import smo_solve


class SVRScratch:
    def __init__(self, C: float = 1.0, epsilon: float = 0.1, kernel: str = "linear",
                 gamma="scale", degree: int = 3, coef0: float = 0.0, tol: float = 1e-3,
                 max_iter: int = 1_000_000):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVRScratch":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.shape[0]
        self.gamma_ = resolve_gamma(self.gamma, X)
        K = self._kernel(X, X)
        z = np.concatenate([np.ones(n), -np.ones(n)])
        K2 = np.block([[K, K], [K, K]])
        Q = (z[:, None] * z[None, :]) * K2
        p = np.concatenate([self.epsilon - y, self.epsilon + y])
        res = smo_solve(Q, p, z, self.C, tol=self.tol, max_iter=self.max_iter)

        coef = res.alpha[:n] - res.alpha[n:]          # a_i - a_i*
        sv = coef != 0
        self.support_ = np.flatnonzero(sv)
        self.support_vectors_ = X[sv]
        self.dual_coef_ = coef[sv]
        self.intercept_ = -res.rho
        self.n_iter_ = res.n_iter
        self.converged_ = res.converged
        self.dual_objective_ = -res.objective
        if self.kernel == "linear":
            self.coef_ = self.dual_coef_ @ self.support_vectors_   # w = sum (a_i - a_i*) x_i
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.support_vectors_.shape[0] == 0:
            return np.full(X.shape[0], self.intercept_)
        return self._kernel(X, self.support_vectors_) @ self.dual_coef_ + self.intercept_

    def primal_objective(self, X: np.ndarray, y: np.ndarray) -> float:
        """1/2||w||^2 + C * somme des pertes epsilon-insensibles (noyau linéaire)."""
        if self.kernel != "linear":
            raise ValueError("Objectif primal explicite seulement pour le noyau linéaire")
        r = np.asarray(y) - (np.asarray(X) @ self.coef_ + self.intercept_)
        return 0.5 * float(self.coef_ @ self.coef_) + self.C * float(
            np.maximum(0.0, np.abs(r) - self.epsilon).sum())

    def _kernel(self, A, B):
        return compute_kernel(A, B, self.kernel, self.gamma_, self.degree, self.coef0)


def epsilon_insensitive_loss(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float) -> np.ndarray:
    """max(0, |y - f(x)| - eps), point par point."""
    return np.maximum(0.0, np.abs(np.asarray(y_true) - np.asarray(y_pred)) - epsilon)

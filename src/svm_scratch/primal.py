"""Versions LINÉAIRES résolues directement dans le PRIMAL par descente de sous-gradient.

Intérêt pédagogique : la fonction de perte apparaît explicitement dans le code
(perte charnière pour la classification, perte epsilon-insensible pour la
régression). Ces pertes ne sont pas dérivables partout (« coude » en 0 ou en
|r| = eps) : on utilise donc un SOUS-gradient.

Objectif minimisé (on divise par C*n, ce qui ne change pas le minimiseur mais
rend les gradients de l'ordre de 1 quel que soit n) :

    J(w, b) = 1/(2 C n) ||w||^2 + (1/n) * sum_i perte_i(w, b)

Pas d'apprentissage décroissant eta_t = eta0 / sqrt(t) (garantie de
convergence classique pour les fonctions convexes non lisses) et on renvoie la
moyenne des itérés de la seconde moitié (moyennage de Polyak-Ruppert), plus
stable que le dernier itéré.

Ces solveurs donnent une solution APPROCHÉE ; la solution exacte est fournie
par le solveur dual SMO (svc.py / svr.py). La comparaison des deux est un test
de cohérence (voir README).
"""
from __future__ import annotations

import numpy as np


class _LinearPrimalBase:
    def __init__(self, C: float = 1.0, n_iter: int = 30_000, eta0: float = 10.0):
        self.C = C
        self.n_iter = n_iter
        self.eta0 = eta0

    def _subgradient(self, X, y, w, b):  # pragma: no cover - défini dans les sous-classes
        raise NotImplementedError

    def _fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        lam = 1.0 / (self.C * n)
        w = np.zeros(d)
        b = 0.0
        w_avg = np.zeros(d)
        b_avg = 0.0
        n_avg = 0
        start_avg = self.n_iter // 2
        for t in range(1, self.n_iter + 1):
            gw_loss, gb_loss = self._subgradient(X, y, w, b)
            eta = self.eta0 / np.sqrt(t)
            w -= eta * (lam * w + gw_loss)
            b -= eta * gb_loss
            if t > start_avg:
                n_avg += 1
                w_avg += (w - w_avg) / n_avg
                b_avg += (b - b_avg) / n_avg
        self.coef_ = w_avg
        self.intercept_ = b_avg
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


class LinearSVCPrimal(_LinearPrimalBase):
    """SVM linéaire binaire (étiquettes -1/+1), perte charnière max(0, 1 - y f(x))."""

    def _subgradient(self, X, y, w, b):
        margins = y * (X @ w + b)
        active = margins < 1                       # points qui violent la marge
        coeff = np.where(active, -y, 0.0) / X.shape[0]
        return X.T @ coeff, coeff.sum()

    def fit(self, X, y):
        return self._fit(X, y)

    def predict(self, X):
        return np.where(self.decision_function(X) > 0, 1, -1)

    def primal_objective(self, X, y):
        m = y * self.decision_function(X)
        return 0.5 * float(self.coef_ @ self.coef_) + self.C * float(np.maximum(0, 1 - m).sum())


class LinearSVRPrimal(_LinearPrimalBase):
    """SVR linéaire, perte epsilon-insensible max(0, |y - f(x)| - eps)."""

    def __init__(self, C: float = 1.0, epsilon: float = 0.1, n_iter: int = 30_000,
                 eta0: float = 10.0):
        super().__init__(C=C, n_iter=n_iter, eta0=eta0)
        self.epsilon = epsilon

    def _subgradient(self, X, y, w, b):
        r = y - (X @ w + b)                        # résidus
        # dérivée de max(0, |r| - eps) par rapport à f(x) : -signe(r) hors du tube, 0 dedans
        coeff = np.where(np.abs(r) > self.epsilon, -np.sign(r), 0.0) / X.shape[0]
        return X.T @ coeff, coeff.sum()

    def fit(self, X, y):
        return self._fit(X, y)

    def predict(self, X):
        return self.decision_function(X)

    def primal_objective(self, X, y):
        r = np.asarray(y) - self.predict(X)
        return 0.5 * float(self.coef_ @ self.coef_) + self.C * float(
            np.maximum(0, np.abs(r) - self.epsilon).sum())

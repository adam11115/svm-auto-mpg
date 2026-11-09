"""Classifieurs SVM (SVC) implémentés depuis zéro.

* BinarySVC      : SVM binaire à marge souple, résolu dans le DUAL par SMO,
                   avec noyau linéaire, RBF ou polynomial.
* SVCScratch     : extension multiclasse « un-contre-un » (One-vs-One), la même
                   stratégie que sklearn.svm.SVC, avec vote majoritaire.

Problème primal (marge souple, perte charnière / hinge loss) :

    min_{w,b}  1/2 ||w||^2 + C * sum_i max(0, 1 - y_i (w . phi(x_i) + b))

Problème dual (résolu par SMO) :

    max_a  sum_i a_i - 1/2 sum_ij a_i a_j y_i y_j K(x_i, x_j)
    s.c.   0 <= a_i <= C,  sum_i a_i y_i = 0

Fonction de décision : f(x) = sum_i a_i y_i K(x_i, x) + b.
Les vecteurs de support sont les points avec a_i > 0.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .kernels import compute_kernel, resolve_gamma
from .smo import smo_solve


class BinarySVC:
    """SVM binaire (étiquettes -1 / +1) résolu dans le dual par SMO."""

    def __init__(self, C: float = 1.0, kernel: str = "linear", gamma="scale",
                 degree: int = 3, coef0: float = 0.0, tol: float = 1e-3,
                 max_iter: int = 1_000_000):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

    # -------------------------------------------------------------- entraînement
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BinarySVC":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if not set(np.unique(y)) <= {-1.0, 1.0}:
            raise ValueError("BinarySVC attend des étiquettes dans {-1, +1}")
        self.gamma_ = resolve_gamma(self.gamma, X)
        K = self._kernel(X, X)
        Q = (y[:, None] * y[None, :]) * K           # Q_ij = y_i y_j K_ij
        p = -np.ones(X.shape[0])                    # terme linéaire : -sum(a_i)
        res = smo_solve(Q, p, y, self.C, tol=self.tol, max_iter=self.max_iter)

        sv = res.alpha > 0                          # vecteurs de support
        self.alpha_ = res.alpha
        self.support_ = np.flatnonzero(sv)
        self.support_vectors_ = X[sv]
        self.dual_coef_ = res.alpha[sv] * y[sv]     # a_i * y_i
        self.intercept_ = -res.rho                  # b = -rho
        self.n_iter_ = res.n_iter
        self.converged_ = res.converged
        self.dual_objective_ = -res.objective       # valeur du dual (à maximiser)
        if self.kernel == "linear":                 # w explicite : sum a_i y_i x_i
            self.coef_ = self.dual_coef_ @ self.support_vectors_
        return self

    # -------------------------------------------------------------- prédiction
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.support_vectors_.shape[0] == 0:
            return np.full(X.shape[0], self.intercept_)
        return self._kernel(X, self.support_vectors_) @ self.dual_coef_ + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) > 0, 1, -1)

    def primal_objective(self, X: np.ndarray, y: np.ndarray) -> float:
        """1/2||w||^2 + C * somme des pertes charnières (noyau linéaire seulement)."""
        if self.kernel != "linear":
            raise ValueError("Objectif primal explicite seulement pour le noyau linéaire")
        margins = y * (np.asarray(X) @ self.coef_ + self.intercept_)
        return 0.5 * float(self.coef_ @ self.coef_) + self.C * float(np.maximum(0, 1 - margins).sum())

    def _kernel(self, A, B):
        return compute_kernel(A, B, self.kernel, self.gamma_, self.degree, self.coef0)


class OneVsOneClassifier:
    """Stratégie multiclasse un-contre-un (K(K-1)/2 classifieurs binaires).

    Pour chaque paire de classes (a, b) avec a < b on entraîne un classifieur
    binaire où a -> +1 et b -> -1. En prédiction, chaque classifieur vote ;
    la classe la plus votée gagne (en cas d'égalité : la plus petite étiquette,
    comme LIBSVM).
    """

    def __init__(self, binary_factory):
        self.binary_factory = binary_factory  # fonction sans argument -> modèle binaire

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OneVsOneClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.estimators_ = []
        self.pairs_ = []
        sv_union = np.zeros(X.shape[0], dtype=bool)
        for a, b in combinations(range(len(self.classes_)), 2):
            idx = np.flatnonzero((y == self.classes_[a]) | (y == self.classes_[b]))
            y_bin = np.where(y[idx] == self.classes_[a], 1, -1)
            est = self.binary_factory().fit(X[idx], y_bin)
            if hasattr(est, "support_"):
                sv_union[idx[est.support_]] = True
            self.estimators_.append(est)
            self.pairs_.append((a, b))
        # un point est vecteur de support s'il l'est dans AU MOINS un sous-problème
        self.support_ = np.flatnonzero(sv_union)
        self.n_support_total_ = int(sv_union.sum())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        votes = np.zeros((X.shape[0], len(self.classes_)), dtype=int)
        for (a, b), est in zip(self.pairs_, self.estimators_):
            positive = est.decision_function(X) > 0
            votes[positive, a] += 1
            votes[~positive, b] += 1
        return self.classes_[np.argmax(votes, axis=1)]  # argmax : 1er max = plus petite classe


class SVCScratch(OneVsOneClassifier):
    """SVC multiclasse « maison » (dual + SMO + un-contre-un)."""

    def __init__(self, C: float = 1.0, kernel: str = "rbf", gamma="scale",
                 degree: int = 3, coef0: float = 0.0, tol: float = 1e-3):
        self.C, self.kernel, self.gamma = C, kernel, gamma
        self.degree, self.coef0, self.tol = degree, coef0, tol
        super().__init__(lambda: BinarySVC(C=self.C, kernel=self.kernel, gamma=self.gamma_,
                                           degree=self.degree, coef0=self.coef0, tol=self.tol))

    def fit(self, X, y):
        # gamma='scale' est calculé UNE fois sur toutes les données d'entraînement
        # (comme sklearn), puis transmis à chaque sous-problème binaire.
        self.gamma_ = resolve_gamma(self.gamma, np.asarray(X, dtype=float))
        return super().fit(X, y)

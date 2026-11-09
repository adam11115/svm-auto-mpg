"""Fonctions noyau (kernels) implémentées avec NumPy uniquement.

Les définitions sont EXACTEMENT celles de scikit-learn / LIBSVM, afin que la
comparaison avec scikit-learn porte sur le même problème mathématique :

    linéaire     : K(x, z) = <x, z>
    RBF (gauss.) : K(x, z) = exp(-gamma * ||x - z||^2)
    polynomial   : K(x, z) = (gamma * <x, z> + coef0) ** degree
"""
from __future__ import annotations

import numpy as np


def linear_kernel(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Produit scalaire : matrice (n_X, n_Z) des <x_i, z_j>."""
    return X @ Z.T


def rbf_kernel(X: np.ndarray, Z: np.ndarray, gamma: float) -> np.ndarray:
    """Noyau gaussien. ||x - z||^2 = ||x||^2 + ||z||^2 - 2 <x, z>."""
    sq_dists = (
        np.sum(X ** 2, axis=1)[:, None]
        + np.sum(Z ** 2, axis=1)[None, :]
        - 2.0 * (X @ Z.T)
    )
    np.maximum(sq_dists, 0.0, out=sq_dists)  # supprime les -1e-15 dus aux arrondis
    return np.exp(-gamma * sq_dists)


def polynomial_kernel(X: np.ndarray, Z: np.ndarray, degree: int, gamma: float,
                      coef0: float) -> np.ndarray:
    """Noyau polynomial inhomogène."""
    return (gamma * (X @ Z.T) + coef0) ** degree


def resolve_gamma(gamma, X: np.ndarray) -> float:
    """Traduit gamma='scale' / 'auto' en valeur numérique (mêmes règles que sklearn).

    'scale' : 1 / (n_features * Var(X))   (Var calculée sur toute la matrice)
    'auto'  : 1 / n_features
    """
    n_features = X.shape[1]
    if gamma == "scale":
        var = X.var()
        return 1.0 / (n_features * var) if var > 0 else 1.0
    if gamma == "auto":
        return 1.0 / n_features
    return float(gamma)


def compute_kernel(X: np.ndarray, Z: np.ndarray, kernel: str, gamma: float = 1.0,
                   degree: int = 3, coef0: float = 0.0) -> np.ndarray:
    """Point d'entrée unique utilisé par les modèles."""
    if kernel == "linear":
        return linear_kernel(X, Z)
    if kernel == "rbf":
        return rbf_kernel(X, Z, gamma)
    if kernel == "poly":
        return polynomial_kernel(X, Z, degree, gamma, coef0)
    raise ValueError(f"Noyau inconnu : {kernel!r} (attendu 'linear', 'rbf' ou 'poly')")

"""Métriques d'évaluation implémentées depuis zéro."""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------ classification
def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """M[i, j] = nombre d'exemples de vraie classe i prédits en classe j."""
    M = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(M, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return M


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def balanced_accuracy(y_true, y_pred, n_classes: int) -> float:
    """Moyenne des rappels par classe : insensible au déséquilibre des classes."""
    M = confusion_matrix(y_true, y_pred, n_classes)
    support = M.sum(axis=1)
    recalls = np.diag(M)[support > 0] / support[support > 0]
    return float(np.mean(recalls))


def f1_macro(y_true, y_pred, n_classes: int) -> float:
    """Moyenne non pondérée des F1 par classe (chaque classe compte autant)."""
    M = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(M).astype(float)
    pred = M.sum(axis=0).astype(float)
    true = M.sum(axis=1).astype(float)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(np.mean(f1))


# ------------------------------------------------------------ régression
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    ss_res = np.sum((y_true - np.asarray(y_pred)) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)

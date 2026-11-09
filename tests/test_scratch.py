"""Tests automatiques : chaque brique « maison » est comparée à scikit-learn.

Lancer :  python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

from src.data import load_clean
from src.metrics import accuracy, balanced_accuracy, confusion_matrix, f1_macro, mae, r2, rmse
from src.preprocessing import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, TabularPreprocessor,
                               TargetDiscretizer, stratified_kfold, stratified_train_test_split)
from src.svm_scratch import kernels as K
from src.svm_scratch.primal import LinearSVCPrimal, LinearSVRPrimal
from src.svm_scratch.svc import BinarySVC, SVCScratch
from src.svm_scratch.svr import SVRScratch


@pytest.fixture(scope="module")
def data():
    df, _ = load_clean(offline=True)
    X = TabularPreprocessor().fit_transform(df)
    y = df["mpg"].to_numpy()
    cls = TargetDiscretizer("quantile").fit_transform(y)
    return df, X, y, cls


def test_dataset_has_392_rows(data):
    df, X, _, _ = data
    assert len(df) == 392
    # 5 continues + (5 - 1) modalités cylinders + (3 - 1) origines (une référence par variable)
    assert X.shape == (392, 5 + 4 + 2)


def test_kernels_match_sklearn():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(20, 4)), rng.normal(size=(15, 4))
    assert np.allclose(K.linear_kernel(A, B), linear_kernel(A, B))
    assert np.allclose(K.rbf_kernel(A, B, 0.3), rbf_kernel(A, B, gamma=0.3))
    assert np.allclose(K.polynomial_kernel(A, B, 3, 0.2, 1.0),
                       polynomial_kernel(A, B, degree=3, gamma=0.2, coef0=1.0))


def test_preprocessing_matches_sklearn(data):
    df = data[0]
    ref = [df[c].mode().iloc[0] for c in CATEGORICAL_FEATURES]       # 4 cylindres, usa
    assert ref == [4, "usa"]
    ct = ColumnTransformer([("num", StandardScaler(), NUMERIC_FEATURES),
                            ("cat", OneHotEncoder(drop=ref, handle_unknown="ignore"),
                             CATEGORICAL_FEATURES)], sparse_threshold=0)
    ours = TabularPreprocessor().fit(df)
    assert ours.feature_names_[-6:] == ["cylinders_3", "cylinders_5", "cylinders_6",
                                        "cylinders_8", "origin_europe", "origin_japan"]
    assert np.allclose(ours.transform(df), ct.fit_transform(df))


def test_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    yt, yp = rng.integers(0, 3, 100), rng.integers(0, 3, 100)
    assert accuracy(yt, yp) == pytest.approx(skm.accuracy_score(yt, yp))
    assert balanced_accuracy(yt, yp, 3) == pytest.approx(skm.balanced_accuracy_score(yt, yp))
    assert f1_macro(yt, yp, 3) == pytest.approx(skm.f1_score(yt, yp, average="macro"))
    assert (confusion_matrix(yt, yp, 3) == skm.confusion_matrix(yt, yp)).all()
    a, b = rng.normal(size=50), rng.normal(size=50)
    assert rmse(a, b) == pytest.approx(np.sqrt(skm.mean_squared_error(a, b)))
    assert mae(a, b) == pytest.approx(skm.mean_absolute_error(a, b))
    assert r2(a, b) == pytest.approx(skm.r2_score(a, b))


def test_splits_are_stratified_and_disjoint(data):
    cls = data[3]
    tr, te = stratified_train_test_split(cls, 0.2, seed=42)
    assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == len(cls)
    for f_tr, f_va in stratified_kfold(cls[tr], k=5, seed=42):
        assert len(np.intersect1d(f_tr, f_va)) == 0


@pytest.mark.parametrize("kernel,kw", [("linear", {}), ("rbf", {"gamma": 0.1}),
                                       ("poly", {"gamma": 0.1, "degree": 3, "coef0": 1.0})])
def test_svc_matches_sklearn(data, kernel, kw):
    _, X, _, cls = data
    ours = SVCScratch(C=1.0, kernel=kernel, **kw).fit(X, cls)
    ref = SVC(C=1.0, kernel=kernel, **kw).fit(X, cls)
    assert np.mean(ours.predict(X) == ref.predict(X)) >= 0.99
    assert abs(ours.n_support_total_ - ref.support_.size) <= 2


def test_binary_svc_kkt_and_duality(data):
    _, X, _, cls = data
    y = np.where(cls == 0, 1, -1)
    m = BinarySVC(C=1.0, kernel="linear").fit(X, y)
    assert m.converged_
    assert np.all((m.alpha_ >= 0) & (m.alpha_ <= 1.0))           # 0 <= a <= C
    assert abs(m.alpha_ @ y) < 1e-8                                # sum a_i y_i = 0
    primal, dual = m.primal_objective(X, y), m.dual_objective_
    assert (primal - dual) / primal < 1e-3                         # saut de dualité ~ 0


@pytest.mark.parametrize("eps", [0.0, 1.0, 3.0])
def test_svr_matches_sklearn(data, eps):
    _, X, y, _ = data
    ours = SVRScratch(C=1.0, epsilon=eps).fit(X, y)
    ref = SVR(kernel="linear", C=1.0, epsilon=eps).fit(X, y)
    assert np.max(np.abs(ours.predict(X) - ref.predict(X))) < 0.05
    assert abs(ours.support_.size - ref.support_.size) <= 2
    assert np.allclose(ours.coef_, ref.coef_.ravel(), atol=0.05)


def test_primal_solvers_reach_dual_optimum(data):
    _, X, y, cls = data
    yb = np.where(cls == 0, 1, -1)
    d, p = BinarySVC(C=1.0).fit(X, yb), LinearSVCPrimal(C=1.0).fit(X, yb)
    assert p.primal_objective(X, yb) <= d.primal_objective(X, yb) * 1.01
    d, p = SVRScratch(C=1.0, epsilon=1.0).fit(X, y), LinearSVRPrimal(C=1.0, epsilon=1.0).fit(X, y)
    assert p.primal_objective(X, y) <= d.primal_objective(X, y) * 1.01


def test_discretizer_orders_classes_by_consumption():
    y = np.array([10.0, 15.0, 22.0, 25.0, 35.0, 40.0])
    c = TargetDiscretizer("quantile").fit_transform(y)
    assert c[0] == 0 and c[-1] == 2       # 10 mpg -> consommation élevée (0), 40 mpg -> faible (2)
    assert list(TargetDiscretizer("domain").fit_transform(np.array([19.9, 20.0, 30.0]))) == [0, 1, 2]

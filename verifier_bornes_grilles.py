"""Vérification facultative : les hyperparamètres retenus ne sont-ils pas au BORD des grilles ?

Si le meilleur C (ou gamma) était la valeur extrême de la grille, le vrai
optimum pourrait se trouver au-delà. On relance donc la même sélection (même
découpage, mêmes plis, même règle parcimonieuse) avec des grilles beaucoup plus
larges et on vérifie que le choix ne change pas.

Pour aller vite on utilise scikit-learn (même problème que nos modèles maison,
cf. tests/ et results/tables/02_cv_*.csv où les deux colonnes sont identiques).

    python verifier_bornes_grilles.py            (~30 secondes)
"""
from sklearn.svm import SVC, SVR

from src.experiments.classification import GRIDS, POLY_COEF0, STRATEGY
from src.experiments.common import N_FOLDS, SEED, build_context
from src.experiments.regression import GRID as SVR_GRID
from src.metrics import accuracy, rmse
from src.model_selection import grid_search_cv
from src.preprocessing import TargetDiscretizer, kfold, stratified_kfold

WIDE = {
    "linear": {"C": [1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000]},
    "rbf": {"C": [0.1, 1, 10, 100, 1000, 1e4, 1e5], "gamma": [3e-4, 1e-3, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]},
    "poly": {"C": [0.1, 1, 10, 100, 1000, 1e4], "degree": [2, 3, 4, 5], "gamma": [0.003, 0.01, 0.03, 0.1, 0.3]},
}
SVR_WIDE = {"C": [1e-3, 0.01, 0.1, 1, 10, 100, 1000], "epsilon": [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8]}

ctx = build_context(offline=True)
c_tr = TargetDiscretizer(STRATEGY).fit(ctx.y_train).transform(ctx.y_train)
folds = stratified_kfold(c_tr, N_FOLDS, SEED)
for kernel in GRIDS:
    def mk(**p):
        return SVC(kernel=kernel, coef0=POLY_COEF0 if kernel == "poly" else 0.0, **p)
    _, normal = grid_search_cv(mk, GRIDS[kernel], ctx.X_train, c_tr, folds, accuracy)
    _, wide = grid_search_cv(mk, WIDE[kernel], ctx.X_train, c_tr, folds, accuracy)
    print(f"SVC {kernel:6s} grille du projet -> {normal} | grille large -> {wide} | "
          f"{'IDENTIQUE' if normal == wide else 'DIFFÉRENT'}")

rf = kfold(len(ctx.y_train), N_FOLDS, SEED)
def mk_svr(**p):
    return SVR(kernel="linear", **p)
_, normal = grid_search_cv(mk_svr, SVR_GRID, ctx.X_train, ctx.y_train, rf, rmse, greater_is_better=False)
_, wide = grid_search_cv(mk_svr, SVR_WIDE, ctx.X_train, ctx.y_train, rf, rmse, greater_is_better=False)
print(f"SVR linéaire grille du projet -> {normal} | grille large -> {wide} | "
      f"{'IDENTIQUE' if normal == wide else 'DIFFÉRENT'}")

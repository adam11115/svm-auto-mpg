# Résultats (générés automatiquement par `python main.py`)

Environnement : Python 3.13.18, NumPy 2.5.3, scikit-learn 1.9.1 — durée 121.7 s

Données : 392 véhicules (seaborn.load_dataset('mpg')), train 314 / test 78

## Discrétisation de mpg

| stratégie | seuils (mpg) | effectifs train (élevée/moyenne/faible) | accuracy CV | accuracy équilibrée CV | F1 macro CV |
|---|---|---|---|---|---|
| quantile | 18.73 / 26.93 | 105 / 104 / 105 | 0.834 | 0.834 | 0.835 |
| uniform | 21.53 / 34.07 | 149 / 132 / 33 | 0.844 | 0.728 | 0.743 |
| domain | 20.00 / 30.00 | 123 / 116 / 75 | 0.831 | 0.826 | 0.827 |

## Classification (test, 78 véhicules)

| noyau | implémentation | hyperparamètres | accuracy | acc. équilibrée | F1 macro | n VS | accord maison/sklearn |
|---|---|---|---|---|---|---|---|
| linear | scratch | {'C': 0.1} | 0.821 | 0.821 | 0.822 | 173 | 100 % |
| linear | sklearn | {'C': 0.1} | 0.821 | 0.821 | 0.822 | 173 | - |
| rbf | scratch | {'C': 1000, 'gamma': 0.01} | 0.846 | 0.846 | 0.848 | 107 | 100 % |
| rbf | sklearn | {'C': 1000, 'gamma': 0.01} | 0.846 | 0.846 | 0.848 | 107 | - |
| poly | scratch | {'C': 100, 'degree': 4, 'gamma': 0.03} | 0.821 | 0.821 | 0.819 | 110 | 100 % |
| poly | sklearn | {'C': 100, 'degree': 4, 'gamma': 0.03} | 0.821 | 0.821 | 0.819 | 110 | - |
| linear | scratch_primal_sous_gradient | {'C': 0.1} | 0.821 | 0.821 | 0.822 | - | 100 % |
| - | baseline_classe_majoritaire | - | 0.333 | 0.333 | - | - | - |

## Régression (test)

Hyperparamètres retenus (CV) : {'C': 10, 'epsilon': 2.5}

| modèle | RMSE | MAE | R² | n VS |
|---|---|---|---|---|
| scratch_dual_SMO | 3.717 | 2.659 | 0.783 | 109 |
| scratch_primal_sous_gradient | 3.716 | 2.660 | 0.784 | - |
| sklearn_SVR_kernel_linear | 3.717 | 2.659 | 0.783 | 109 |
| sklearn_LinearSVR | 3.708 | 2.647 | 0.784 | - |
| reference_moindres_carres_OLS | 3.581 | 2.616 | 0.799 | - |
| reference_moyenne_constante | 8.014 | 6.455 | -0.006 | - |

## Influence de epsilon (C fixé)

| epsilon | n VS maison | n VS sklearn | % VS | % train dans le tube | RMSE CV | RMSE test |
|---|---|---|---|---|---|---|
| 0 | 314 | 314 | 100.0 | 0.0 | 3.038 | 3.809 |
| 0.25 | 282 | 281 | 89.8 | 10.2 | 3.036 | 3.789 |
| 0.5 | 256 | 256 | 81.5 | 18.5 | 3.027 | 3.860 |
| 0.75 | 232 | 232 | 73.9 | 26.1 | 3.029 | 3.831 |
| 1 | 212 | 212 | 67.5 | 32.5 | 3.034 | 3.816 |
| 1.5 | 174 | 174 | 55.4 | 44.6 | 3.018 | 3.794 |
| 2 | 142 | 142 | 45.2 | 54.8 | 2.976 | 3.727 |
| 2.5 | 109 | 109 | 34.7 | 65.3 | 2.966 | 3.717 |
| 3 | 81 | 81 | 25.8 | 74.2 | 2.989 | 3.698 |
| 3.5 | 64 | 64 | 20.4 | 79.6 | 2.998 | 3.607 |
| 4 | 45 | 45 | 14.3 | 85.7 | 3.003 | 3.611 |
| 5 | 30 | 30 | 9.6 | 90.4 | 3.158 | 3.456 |
| 6 | 25 | 25 | 8.0 | 92.0 | 3.310 | 3.442 |
| 7 | 15 | 15 | 4.8 | 95.2 | 3.601 | 3.498 |
| 8 | 10 | 10 | 3.2 | 96.8 | 3.944 | 3.620 |
| 10 | 6 | 6 | 1.9 | 98.1 | 4.514 | 4.177 |

Moteur des études de bruit : sklearn.svm.SVR(kernel='linear')


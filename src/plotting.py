"""Style commun des figures (matplotlib, export PNG, sans affichage à l'écran)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sans fenêtre : fonctionne partout (Windows, Linux, serveur)
import matplotlib.pyplot as plt  # noqa: E402

# Palette catégorielle (ordre fixe) : 1 = implémentation maison, 2 = scikit-learn, 3 = autre
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ_CMAP = "Blues"  # séquentiel : une seule teinte, du clair au foncé

FIG_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"


def setup():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
    })


def save(fig, name: str) -> str:
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return f"results/figures/{name}"

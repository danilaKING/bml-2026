"""Общие параметры оформления рисунков и вспомогательные функции вывода."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

RC_PARAMS = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "DejaVu Sans", "Liberation Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.labelsize": 11,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "figure.dpi": 120,
}


def use_style():
    """Применить единые параметры оформления matplotlib."""
    plt.rcParams.update(RC_PARAMS)


use_style()


def save(fig, name):
    """Сохранить рисунок в figures/<name>.png и закрыть его."""
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  сохранено: {path.relative_to(Path(__file__).resolve().parent)}")
    return path


def section(title):
    """Вывести заголовок раздела."""
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


def subsection(title):
    """Вывести заголовок подраздела."""
    print(f"\n--- {title} " + "-" * max(0, 68 - len(title)))


def raz(n):
    """Согласовать форму слова «раз» с числительным."""
    n = int(round(n))
    if 11 <= n % 100 <= 14:
        return "раз"
    return "раза" if n % 10 in (2, 3, 4) else "раз"

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from coral import coral_transform
from dann import fit_dann
from data import make_domains
from models import fit_classifier, predict, predict_proba, features
from self_training import fit_self_training

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
RES = ROOT / "results"

C0, C1 = "#2a78d6", "#eb6834"                 # класс 0 / класс 1
METHOD_COLORS = ["#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 12, "axes.titleweight": "bold", "font.size": 10,
})

METHODS = ["Source-only", "CORAL", "Self-training", "DANN", "Oracle"]


def run_methods(d, seed, angle_only=False):
    """Обучает все методы на одном наборе данных. Возвращает {имя: (модель, доп. данные)}."""
    Xs, ys, Xt = d["Xs"], d["ys"], d["Xt_train"]
    out = {}
    out["Source-only"] = (fit_classifier(Xs, ys, seed=seed), None)
    Xs_coral = coral_transform(Xs, Xt, lam=1.0)
    out["CORAL"] = (fit_classifier(Xs_coral, ys, seed=seed), Xs_coral)
    if not angle_only:
        out["Self-training"] = (fit_self_training(Xs, ys, Xt, seed=seed), None)
    out["DANN"] = (fit_dann(Xs, ys, Xt, seed=seed), None)
    if not angle_only:
        out["Oracle"] = (fit_classifier(Xt, d["yt_train_oracle"], seed=seed), None)
    return out


def accuracy(model, X, y):
    return float((predict(model, X) == y).mean())


# ---------------------------------------------------------------- графики
def scatter_domain(ax, X, y, marker, alpha=0.8, size=14, label_prefix=""):
    for cls, col in ((0, C0), (1, C1)):
        m = y == cls
        kw = dict(s=size, alpha=alpha, marker=marker, linewidths=1.0)
        if marker == "o":
            ax.scatter(X[m, 0], X[m, 1], c=col, edgecolors="white", linewidths=0.4,
                       s=size, alpha=alpha, marker="o", label=f"{label_prefix}класс {cls}")
        else:
            ax.scatter(X[m, 0], X[m, 1], c=col, label=f"{label_prefix}класс {cls}", **kw)


def plot_data(d, angle, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    scatter_domain(ax, d["Xs"], d["ys"], "o", label_prefix="source, ")
    scatter_domain(ax, d["Xt_test"], d["yt_test"], "x", alpha=0.6, label_prefix="target, ")
    ax.set_title(f"Source и target (поворот на {angle}°)", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_boundaries(d, models, accs, path):
    Xt, yt = d["Xt_test"], d["yt_test"]
    allX = np.vstack([d["Xs"], Xt])
    x0, x1 = allX[:, 0].min() - 0.5, allX[:, 0].max() + 0.5
    y0, y1 = allX[:, 1].min() - 0.5, allX[:, 1].max() + 0.5
    gx, gy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
    grid = np.c_[gx.ravel(), gy.ravel()].astype(np.float32)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2))
    axes = axes.ravel()

    ax = axes[0]
    scatter_domain(ax, d["Xs"], d["ys"], "o", label_prefix="source, ")
    scatter_domain(ax, Xt, yt, "x", alpha=0.5, label_prefix="target, ")
    ax.set_title("Данные: source (●) и target (×)", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    for ax, name in zip(axes[1:], METHODS):
        model, extra = models[name]
        p = predict_proba(model, grid)[:, 1].reshape(gx.shape)
        ax.contourf(gx, gy, p, levels=[0, 0.5, 1], colors=["#dbe8f8", "#fbe1d6"], alpha=0.9)
        ax.contour(gx, gy, p, levels=[0.5], colors=INK, linewidths=1.5)
        if name == "CORAL":
            ax.scatter(extra[:, 0], extra[:, 1], c=[C0 if v == 0 else C1 for v in d["ys"]],
                       s=8, alpha=0.35, marker="o", edgecolors="none")
        pred = predict(model, Xt)
        ok = pred == yt
        ax.scatter(Xt[ok, 0], Xt[ok, 1], c=[C0 if v == 0 else C1 for v in yt[ok]],
                   marker="x", s=14, linewidths=1.0, alpha=0.8)
        ax.scatter(Xt[~ok, 0], Xt[~ok, 1], facecolors="none", edgecolors=INK,
                   marker="o", s=40, linewidths=1.2, label="ошибка")
        ax.set_title(f"{name}: точность на target {accs[name]:.1%}", loc="left")
        if (~ok).any():
            ax.legend(frameon=False, fontsize=8, loc="lower left")

    for ax in axes:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Граница решения каждого метода на target (фон — предсказанный класс, "
                 "○ — ошибки)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_features(d, model_src, model_dann, path):
    """PCA признаков G(x): до адаптации (source-only) и после (DANN)."""
    Xs, ys, Xt, yt = d["Xs"], d["ys"], d["Xt_test"], d["yt_test"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model, title in ((axes[0], model_src, "Признаки без адаптации (source-only)"),
                             (axes[1], model_dann, "Признаки после DANN")):
        Fs, Ft = features(model, Xs), features(model, Xt)
        pca = PCA(n_components=2).fit(np.vstack([Fs, Ft]))
        Zs, Zt = pca.transform(Fs), pca.transform(Ft)
        scatter_domain(ax, Zs, ys, "o", alpha=0.6, label_prefix="source, ")
        scatter_domain(ax, Zt, yt, "x", alpha=0.6, label_prefix="target, ")
        ax.set_title(title, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Цвет — класс, форма — домен. После DANN target (×) ложится "
                 "на source (●) того же класса", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_sweep(angles, table, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for (name, vals), col in zip(table.items(), METHOD_COLORS):
        m = np.array([np.mean(v) for v in vals])
        s = np.array([np.std(v) for v in vals])
        ax.plot(angles, m, color=col, lw=2, marker="o", ms=5, label=name)
        ax.fill_between(angles, m - s, m + s, color=col, alpha=0.12, lw=0)
        ax.annotate(name, (angles[-1], m[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.set_xlabel("Угол поворота target, градусы")
    ax.set_ylabel("Точность на target")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", color=GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(angles[0] - 2, angles[-1] + 14)
    ax.set_title("Чем сильнее сдвиг, тем больше выигрыш от адаптации", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angle", type=float, default=35)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sweep", action="store_true", help="построить точность против угла")
    args = ap.parse_args()

    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    t0 = time.time()

    accs = {m: [] for m in METHODS}
    for seed in range(args.seeds):
        d = make_domains(angle=args.angle, seed=seed)
        models = run_methods(d, seed)
        for m in METHODS:
            accs[m].append(accuracy(models[m][0], d["Xt_test"], d["yt_test"]))
        print(f"seed {seed}: " + ", ".join(f"{m} {accs[m][-1]:.3f}" for m in METHODS))
        if seed == 0:
            first_d, first_models = d, models

    lines = [f"# Точность на target_test (поворот {args.angle:g}°, {args.seeds} seed)", "",
             "| Метод | Точность, среднее ± std |", "|---|---|"]
    for m in METHODS:
        lines.append(f"| {m} | {np.mean(accs[m]):.1%} ± {np.std(accs[m]):.1%} |")
    table = "\n".join(lines)
    print("\n" + table)
    (RES / "results.md").write_text(table + "\n", encoding="utf-8")

    plot_data(first_d, args.angle, FIG / "data.png")
    seed0 = {m: accs[m][0] for m in METHODS}
    plot_boundaries(first_d, first_models, seed0, FIG / "boundaries.png")
    plot_features(first_d, first_models["Source-only"][0], first_models["DANN"][0],
                  FIG / "dann_features.png")

    if args.sweep:
        angles = [0, 10, 20, 30, 40, 50, 60]
        names = ["Source-only", "CORAL", "DANN"]
        sweep = {n: [] for n in names}
        for a in angles:
            per = {n: [] for n in names}
            for seed in range(3):
                d = make_domains(angle=a, seed=seed)
                ms = run_methods(d, seed, angle_only=True)
                for n in names:
                    per[n].append(accuracy(ms[n][0], d["Xt_test"], d["yt_test"]))
            for n in names:
                sweep[n].append(per[n])
            print(f"угол {a:>2}°: " + ", ".join(f"{n} {np.mean(per[n]):.3f}" for n in names))
        plot_sweep(angles, sweep, FIG / "accuracy_vs_angle.png")

    print(f"\nГотово за {time.time() - t0:.0f} с. Графики: {FIG}, таблица: {RES / 'results.md'}")


if __name__ == "__main__":
    main()

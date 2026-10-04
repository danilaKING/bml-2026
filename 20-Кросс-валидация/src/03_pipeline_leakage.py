"""Смещение оценки при предобработке, выполненной вне процедуры.

Первый эксперимент проводится на некоррелированных данных, где истинное
значение accuracy равно 0.50; второй — на подвыборке breast_cancer,
дополненной шумовыми признаками. В обоих случаях сравниваются два порядка
действий: предобработка до разбиения и предобработка внутри конвейера.

Постановка первого эксперимента соответствует Hastie T., Tibshirani R.,
Friedman J. The Elements of Statistical Learning, § 7.10.2.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from common import RANDOM_STATE, ORANGE, RED, AQUA, MUTED, INK_2, save, section

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

N_OBJECTS = 100
N_FEATURES = 5000
N_SELECTED = 20

N_PATIENTS = 50
N_NOISE = 1000
N_SELECTED_REAL = 10
MEASUREMENT_NOISE = 2.0


def uncorrelated_data():
    """Эксперимент на данных, не содержащих зависимости между X и y."""
    section("1. Некоррелированные данные: n = 100, p = 5000")

    rng = np.random.RandomState(RANDOM_STATE)
    X = rng.normal(size=(N_OBJECTS, N_FEATURES))
    y = rng.randint(0, 2, size=N_OBJECTS)

    selector = SelectKBest(f_classif, k=N_SELECTED).fit(X, y)
    outside = cross_val_score(KNeighborsClassifier(n_neighbors=3),
                              selector.transform(X), y, cv=cv,
                              scoring="accuracy")

    pipe = Pipeline([("select", SelectKBest(f_classif, k=N_SELECTED)),
                     ("clf", KNeighborsClassifier(n_neighbors=3))])
    inside = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")

    print("  Отбор признаков вне процедуры:")
    print(f"    по блокам: {np.round(outside, 3)}")
    print(f"    среднее:   {outside.mean():.4f}")
    print("  Отбор признаков внутри конвейера:")
    print(f"    по блокам: {np.round(inside, 3)}")
    print(f"    среднее:   {inside.mean():.4f}")
    print(f"  Истинное значение: 0.5000. Смещение: "
          f"{100 * (outside.mean() - 0.5):+.1f} п.п.")
    print("  Среди 5000 независимых признаков существуют такие, что на 100 "
          "объектах\n  случайно коррелируют с y; отбор по всей выборке "
          "передаёт модели информацию\n  о контрольных объектах.")
    return outside.mean(), inside.mean()


def real_data(n_repeats=30):
    """Эксперимент на подвыборке breast_cancer с шумовыми признаками."""
    section("2. Реальные данные: n = 50, p = 1030")

    X_full, y_full = load_breast_cancer(return_X_y=True)
    print(f"  Объектов: {N_PATIENTS}, признаков: "
          f"{X_full.shape[1] + N_NOISE}, повторов: {n_repeats}.")

    outside, inside = [], []
    for seed in range(n_repeats):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_full), size=N_PATIENTS, replace=False)
        y = y_full[idx]
        X_signal = StandardScaler().fit_transform(X_full[idx])
        X_signal = X_signal + rng.normal(0, MEASUREMENT_NOISE, X_signal.shape)
        X = np.hstack([X_signal, rng.normal(size=(N_PATIENTS, N_NOISE))])

        cv_local = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        X_prepared = SelectKBest(f_classif, k=N_SELECTED_REAL).fit_transform(
            StandardScaler().fit_transform(X), y)
        outside.append(cross_val_score(
            LogisticRegression(max_iter=5000), X_prepared, y,
            cv=cv_local, scoring="accuracy").mean())

        pipe = Pipeline([("scale", StandardScaler()),
                         ("select", SelectKBest(f_classif, k=N_SELECTED_REAL)),
                         ("clf", LogisticRegression(max_iter=5000))])
        inside.append(cross_val_score(pipe, X, y, cv=cv_local,
                                      scoring="accuracy").mean())

    outside, inside = np.array(outside), np.array(inside)
    gap = outside - inside
    print(f"  Предобработка вне процедуры:    {outside.mean():.4f} "
          f"± {outside.std():.4f}")
    print(f"  Предобработка внутри конвейера: {inside.mean():.4f} "
          f"± {inside.std():.4f}")
    print(f"  Среднее смещение: {100 * gap.mean():+.1f} п.п.; "
          f"положительно в {100 * (gap > 0).mean():.0f} % повторов.")
    return outside.mean(), inside.mean(), gap


def plot_results(outside_noise, inside_noise, gap):
    """Рис. 5: смещение оценки в обоих экспериментах."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    vals = [outside_noise, inside_noise]
    bars = ax.bar(["отбор признаков\nвне процедуры CV",
                   "отбор признаков\nвнутри Pipeline"],
                  vals, color=[RED, AQUA], width=0.5, zorder=3)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1.6, zorder=1,
               label="истинное значение accuracy = 0.50")
    ax.legend(loc="upper right", fontsize=10)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.2f}",
                ha="center", fontsize=15, fontweight="bold", color=INK_2)
    ax.annotate("", xy=(0.33, outside_noise), xytext=(0.33, 0.5),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.8))
    ax.text(0.40, (outside_noise + 0.5) / 2,
            f"смещение\n+{100 * (outside_noise - 0.5):.0f} п.п.",
            color=RED, fontsize=11, va="center")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("accuracy (5-блочная CV)")
    ax.set_title("(а) Некоррелированные данные: n = 100, p = 5000",
                 fontsize=12, loc="left")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    ax.hist(100 * gap, bins=12, color=ORANGE, alpha=0.9,
            edgecolor="white", linewidth=0.8, zorder=3)
    top = ax.get_ylim()[1]
    ax.axvline(0, color=MUTED, lw=1.6, ls="--", zorder=4)
    ax.axvline(100 * gap.mean(), color=RED, lw=2.2, zorder=5)
    ax.text(100 * gap.mean(), top * 1.02,
            f"среднее смещение {100 * gap.mean():+.1f} п.п.",
            color=RED, fontsize=11, ha="center")
    ax.set_ylim(0, top * 1.22)
    ax.set_title(f"(б) Реальные данные: n = 50, p = 1030, "
                 f"{len(gap)} повторов", fontsize=12, loc="left")
    ax.set_xlabel("смещение оценки accuracy, п.п.")
    ax.set_ylabel("число повторов")
    ax.grid(axis="x", visible=False)

    fig.suptitle("Рис. 5. Смещение оценки при отборе признаков вне процедуры\n"
                 "кросс-валидации", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "fig05_data_leakage")


if __name__ == "__main__":
    outside_noise, inside_noise = uncorrelated_data()
    real_data_outside, real_data_inside, gap = real_data()
    plot_results(outside_noise, inside_noise, gap)

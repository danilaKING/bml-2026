"""Постановка задачи: эмпирическая ошибка и дисперсия оценки качества.

Модуль воспроизводит три численных результата, приведённых в разделе 1
презентации: зависимость ошибки от сложности модели, распределение оценки
accuracy по случайным разбиениям train/test и зависимость дисперсии оценки
от размера контрольной выборки.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
from common import (RANDOM_STATE, BLUE, ORANGE, AQUA, RED, MUTED, INK_2,
                    save, section)

SHARES = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
N_SPLITS_HOLDOUT = 300
N_REPEATS_SHARE = 120


def target(t):
    """Регрессионная зависимость, использованная для генерации данных."""
    return np.sin(2.2 * t) + 0.4 * t


def score_one_split(X, y, share, seed):
    """Accuracy модели на одном случайном разбиении train/test."""
    model = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=share, random_state=seed, stratify=y)
    model.fit(X_tr, y_tr)
    return accuracy_score(y_te, model.predict(X_te))


def empirical_error():
    """Ошибка на обучающей и контрольной выборках как функция сложности."""
    section("1. Эмпирическая ошибка и обобщающая способность")

    rng = np.random.RandomState(RANDOM_STATE)
    X_train = np.sort(rng.uniform(-1, 1, 30)).reshape(-1, 1)
    y_train = target(X_train.ravel()) + rng.normal(0, 0.22, 30)
    X_test = np.sort(rng.uniform(-1, 1, 400)).reshape(-1, 1)
    y_test = target(X_test.ravel()) + rng.normal(0, 0.22, 400)

    err_train, err_test = [], []
    for d in range(1, 16):
        model = make_pipeline(PolynomialFeatures(d), LinearRegression())
        model.fit(X_train, y_train)
        err_train.append(mean_squared_error(y_train, model.predict(X_train)))
        err_test.append(mean_squared_error(y_test, model.predict(X_test)))

    best = int(np.argmin(err_test)) + 1
    print(f"  Минимум ошибки на контрольной выборке: d = {best}, "
          f"MSE = {err_test[best - 1]:.4f}")
    print(f"  d = 15: MSE на обучающей {err_train[-1]:.4f}, "
          f"на контрольной {err_test[-1]:.4f}")
    print(f"  Отношение к минимуму: {err_test[-1] / err_test[best - 1]:.0f}")
    print("  Эмпирическая ошибка убывает монотонно и не может служить "
          "критерием выбора модели.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    grid = np.linspace(-1, 1, 400).reshape(-1, 1)
    ax.scatter(X_train.ravel(), y_train, s=34, color=MUTED, zorder=3,
               label="обучающая выборка (n = 30)")
    for d, color, style in [(1, AQUA, "--"), (best, BLUE, "-"), (15, RED, "-")]:
        model = make_pipeline(PolynomialFeatures(d), LinearRegression())
        model.fit(X_train, y_train)
        ax.plot(grid.ravel(), model.predict(grid), style, color=color,
                label=f"d = {d}")
    ax.set_ylim(-2.0, 2.0)
    ax.set_title("(а) Полиномиальные модели степеней 1, 5 и 15",
                 fontsize=12, loc="left")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left")

    ax = axes[1]
    degrees = list(range(1, 16))
    ax.plot(degrees, err_train, "-o", color=BLUE, label="обучающая выборка")
    ax.plot(degrees, err_test, "-o", color=ORANGE, label="контрольная выборка")
    ax.axvline(best, color=MUTED, lw=1, ls=":")
    ax.annotate(f"минимум при d = {best}", xy=(best, err_test[best - 1]),
                xytext=(7.0, 0.018), color=INK_2, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.set_yscale("log")
    ax.set_title("(б) Среднеквадратичная ошибка", fontsize=12, loc="left")
    ax.set_xlabel("степень полинома $d$")
    ax.set_ylabel("MSE (логарифмическая шкала)")
    ax.legend(loc="upper center")

    fig.suptitle("Рис. 1. Эмпирическая ошибка и ошибка на контрольной выборке\n"
                 "в зависимости от сложности модели",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "fig01_overfitting")


def holdout_variance():
    """Распределение оценки accuracy по случайным разбиениям train/test."""
    section("2. Дисперсия оценки при однократном разбиении")

    X, y = load_breast_cancer(return_X_y=True)
    model = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)

    scores = []
    for seed in range(N_SPLITS_HOLDOUT):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y)
        model.fit(X_tr, y_tr)
        scores.append(accuracy_score(y_te, model.predict(X_te)))
    scores = np.array(scores)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    print(f"  Разбиений: {N_SPLITS_HOLDOUT}, модель и выборка фиксированы.")
    print(f"  Минимум {scores.min():.4f}, максимум {scores.max():.4f}, "
          f"размах {100 * (scores.max() - scores.min()):.1f} п.п.")
    print(f"  Среднее {scores.mean():.4f}, "
          f"стандартное отклонение {scores.std():.4f}")
    print(f"  5-блочная кросс-валидация: {cv_scores.mean():.4f} "
          f"± {cv_scores.std():.4f}")

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.hist(scores, bins=28, color=BLUE, alpha=0.85, edgecolor="white",
            linewidth=0.8)
    ax.axvline(scores.mean(), color=ORANGE, lw=2,
               label=f"выборочное среднее = {scores.mean():.3f}")
    ax.axvline(scores.min(), color=MUTED, lw=1.4, ls="--")
    ax.axvline(scores.max(), color=MUTED, lw=1.4, ls="--")
    ax.annotate("", xy=(scores.min(), 30), xytext=(scores.max(), 30),
                arrowprops=dict(arrowstyle="<->", color=INK_2, lw=1.4))
    ax.text((scores.min() + scores.max()) / 2, 32,
            f"размах {100 * (scores.max() - scores.min()):.1f} п.п.",
            ha="center", color=INK_2, fontsize=11)
    ax.set_title("Рис. 2. Распределение оценки accuracy по 300 случайным\n"
                 "разбиениям train/test при фиксированной модели",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("accuracy на контрольной выборке")
    ax.set_ylabel("число разбиений")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig02_holdout_lottery")

    return scores, cv_scores


def control_sample_size():
    """Зависимость дисперсии оценки и качества модели от размера контроля."""
    section("3. Выбор размера контрольной выборки")

    X, y = load_breast_cancer(return_X_y=True)

    means, stds = [], []
    for share in SHARES:
        vals = []
        for seed in range(N_REPEATS_SHARE):
            vals.append(score_one_split(X, y, share, seed))
        stds.append(np.std(vals))
        means.append(np.mean(vals))

    print("  доля контроля | объём обучения | средняя accuracy | ст. откл.")
    for share, m, sd in zip(SHARES, means, stds):
        n_train = len(X) - int(len(X) * share)
        print(f"       {share:>4.0%}     |      {n_train:>4d}      "
              f"|      {m:.4f}      |   {sd:.4f}")
    print("  Рост контрольной выборки снижает дисперсию оценки и ухудшает "
          "саму модель.")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    xs = [100 * share for share in SHARES]

    ax = axes[0]
    ax.plot(xs, stds, "-o", color=ORANGE)
    ax.fill_between(xs, 0, stds, color=ORANGE, alpha=0.10)
    ax.set_xlabel("доля контрольной выборки, %")
    ax.set_ylabel("ст. отклонение оценки")
    ax.set_ylim(0, max(stds) * 1.25)
    ax.set_title("(а) Стандартное отклонение оценки", fontsize=12, loc="left")

    ax = axes[1]
    ax.plot(xs, means, "-s", color=BLUE)
    ax.set_xlabel("доля контрольной выборки, %")
    ax.set_ylabel("средняя accuracy")
    ax.set_title("(б) Среднее значение accuracy", fontsize=12, loc="left")

    fig.suptitle("Рис. 3. Зависимость стандартного отклонения оценки и среднего "
                 "качества\nмодели от доли контрольной выборки "
                 "(120 повторов на точку)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig03_size_dilemma")


if __name__ == "__main__":
    empirical_error()
    holdout_variance()
    control_sample_size()

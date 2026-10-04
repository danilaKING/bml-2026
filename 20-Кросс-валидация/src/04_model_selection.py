"""Подбор гиперпараметров и сравнение моделей средствами кросс-валидации.

Модуль строит валидационную кривую решающего дерева по параметру max_depth,
выполняет подбор конфигурации SVC по сетке и сравнивает пять моделей
на одних и тех же разбиениях.
"""

import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, validation_curve)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
from common import (RANDOM_STATE, BLUE, ORANGE, AQUA, RED, MUTED, INK_2,
                    save, section)

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

DEPTHS = np.arange(1, 16)
PARAM_GRID = {"clf__C": [0.01, 0.1, 1, 10, 100],
              "clf__gamma": [1e-4, 1e-3, 1e-2, 0.1, 1]}


def validation_curve_depth():
    """Валидационная кривая решающего дерева по параметру max_depth."""
    section("1. Валидационная кривая")

    train, test = validation_curve(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        X, y,
        param_name="max_depth",
        param_range=DEPTHS,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1)

    train_mean, train_std = train.mean(axis=1), train.std(axis=1)
    test_mean, test_std = test.mean(axis=1), test.std(axis=1)
    best = int(DEPTHS[test_mean.argmax()])

    print("  max_depth | обучающая | кросс-валидация ± ст. откл.")
    for d, a, b, s in zip(DEPTHS, train_mean, test_mean, test_std):
        mark = "  <- максимум" if d == best else ""
        print(f"     {d:>2}     |  {a:.4f}   | {b:.4f} ± {s:.4f}{mark}")
    print(f"\n  Максимум кросс-валидационной оценки: max_depth = {best}, "
          f"{test_mean.max():.4f}")
    print(f"  При max_depth = 15: {train_mean[-1]:.4f} на обучающей выборке, "
          f"{test_mean[-1]:.4f} на контрольной.")

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.plot(DEPTHS, train_mean, "-o", color=BLUE, label="обучающая выборка")
    ax.fill_between(DEPTHS, train_mean - train_std, train_mean + train_std,
                    color=BLUE, alpha=0.12)
    ax.plot(DEPTHS, test_mean, "-o", color=ORANGE, label="кросс-валидация")
    ax.fill_between(DEPTHS, test_mean - test_std, test_mean + test_std,
                    color=ORANGE, alpha=0.16, label="± стандартное отклонение")
    ax.axvline(best, color=MUTED, ls=":", lw=1.4)
    ax.plot([best], [test_mean.max()], "*", color=RED, markersize=18, zorder=5)
    ax.annotate(f"максимум при max_depth = {best}",
                xy=(best, test_mean.max()),
                xytext=(best + 1.2, test_mean.max() - 0.055),
                color=INK_2, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.set_xlim(0.5, 15.5)
    ax.set_title("Рис. 6. Валидационная кривая решающего дерева по параметру\n"
                 "max_depth (breast_cancer, 5-блочная стратифицированная CV)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("accuracy")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig06_validation_curve")
    return best


def grid_search():
    """Подбор конфигурации SVC по сетке гиперпараметров."""
    section("2. Подбор конфигурации по сетке")

    pipe = Pipeline([("scale", StandardScaler()),
                     ("clf", SVC())])

    started = time.perf_counter()
    gs = GridSearchCV(pipe, PARAM_GRID, cv=cv,
                      scoring="accuracy", n_jobs=-1)
    gs.fit(X, y)
    elapsed = time.perf_counter() - started

    n_configs = len(PARAM_GRID["clf__C"]) * len(PARAM_GRID["clf__gamma"])
    print(f"  Конфигураций: {n_configs}, обучений: "
          f"{n_configs * cv.get_n_splits()}, время: {elapsed:.2f} с")
    print(f"  Лучшая конфигурация: {gs.best_params_}")
    print(f"  Оценка лучшей конфигурации: {gs.best_score_:.4f}")
    print("  Величина best_score_ является максимумом по всем конфигурациям "
          "и смещена вверх;\n  несмещённую оценку даёт вложенная процедура "
          "(модуль 05_nested_cv.py).")

    scores = gs.cv_results_["mean_test_score"].reshape(
        len(PARAM_GRID["clf__C"]), len(PARAM_GRID["clf__gamma"]))

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    im = ax.imshow(scores, cmap="Blues", vmin=0.6, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(PARAM_GRID["clf__gamma"])))
    ax.set_xticklabels(PARAM_GRID["clf__gamma"])
    ax.set_yticks(range(len(PARAM_GRID["clf__C"])))
    ax.set_yticklabels(PARAM_GRID["clf__C"])
    ax.set_xlabel("gamma")
    ax.set_ylabel("C")
    ax.set_title("Рис. 7. Средняя accuracy по сетке\nгиперпараметров SVC",
                 fontsize=13, fontweight="bold")
    best_i, best_j = np.unravel_index(np.argmax(scores), scores.shape)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            ax.text(j, i, f"{scores[i, j]:.3f}", ha="center", va="center",
                    fontsize=10,
                    fontweight="bold" if (i, j) == (best_i, best_j) else "normal",
                    color="white" if scores[i, j] > 0.85 else INK_2)
    ax.add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1, fill=False,
                               edgecolor=RED, lw=3))
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="средняя accuracy по 5 блокам")
    fig.tight_layout()
    save(fig, "fig07_grid_search")
    return gs


def compare_models():
    """Сравнение пяти моделей на одних и тех же разбиениях."""
    section("3. Сравнение моделей на одних и тех же блоках")

    models = {
        "Логистическая регрессия": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000))]),
        "k ближайших соседей (k=5)": Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(5))]),
        "Решающее дерево": DecisionTreeClassifier(
            max_depth=4, random_state=RANDOM_STATE),
        "Случайный лес": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE),
        "Наивный Байес": GaussianNB(),
    }

    rows, per_fold = [], {}
    for name, estimator in models.items():
        started = time.perf_counter()
        scores = cross_val_score(estimator, X, y, cv=cv,
                                 scoring="accuracy", n_jobs=-1)
        rows.append({"модель": name,
                     "accuracy": scores.mean(),
                     "ст. откл.": scores.std(),
                     "минимум по блокам": scores.min(),
                     "время, с": time.perf_counter() - started})
        per_fold[name] = scores

    table = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best_name = table.iloc[0]["модель"]
    diff = table.iloc[0]["accuracy"] - table.iloc[1]["accuracy"]
    pooled = np.sqrt(table.iloc[0]["ст. откл."] ** 2
                     + table.iloc[1]["ст. откл."] ** 2)
    print(f"\n  Разность между первыми двумя моделями: {diff:.4f}; "
          f"объединённое ст. отклонение: {pooled:.4f}.")
    if diff < pooled:
        print("  Разность меньше разброса, модели статистически неразличимы.")

    fig, ax = plt.subplots(figsize=(10, 4.6))
    order = table["модель"].tolist()[::-1]
    ypos = np.arange(len(order))
    colors = [AQUA if name == best_name else BLUE for name in order]
    means = [per_fold[name].mean() for name in order]
    stds = [per_fold[name].std() for name in order]
    ax.barh(ypos, means, xerr=stds, color=colors, height=0.55, zorder=3,
            error_kw=dict(ecolor=INK_2, lw=1.4, capsize=4))
    for yp, name in zip(ypos, order):
        ax.scatter(per_fold[name], [yp] * len(per_fold[name]), color="white",
                   s=26, zorder=6, edgecolor=INK_2, linewidth=0.9)
        ax.text(0.8835, yp, f"{per_fold[name].mean():.4f}", va="center",
                ha="left", color="white", fontsize=11, fontweight="bold",
                zorder=7)
    ax.set_yticks(ypos)
    ax.set_yticklabels(order)
    ax.set_xlim(0.88, 1.0)
    ax.set_title("Рис. 8. Сравнение пяти моделей на одних и тех же блоках;\n"
                 "точками показаны значения по отдельным блокам",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("accuracy")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig08_model_comparison")
    return table


if __name__ == "__main__":
    validation_curve_depth()
    grid_search()
    compare_models()

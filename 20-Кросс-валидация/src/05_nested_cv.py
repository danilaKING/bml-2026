"""Вложенная кросс-валидация и смещение результата подбора гиперпараметров.

Величина GridSearchCV.best_score_ представляет собой максимум по всем
конфигурациям сетки, каждая из которых оценена с погрешностью. Математическое
ожидание максимума набора случайных величин превышает математическое ожидание
каждой из них, поэтому такая оценка систематически смещена вверх.

Модуль измеряет величину смещения: внутренний цикл выбирает конфигурацию,
внешний оценивает процедуру целиком.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import matplotlib.pyplot as plt
from common import RANDOM_STATE, BLUE, ORANGE, RED, AQUA, INK_2, save, section

PARAM_GRID = {"clf__C": [0.01, 0.1, 1, 10, 100, 1000],
              "clf__gamma": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]}


def run(n_trials=25, n_objects=120):
    """Сравнить best_score_ и оценку вложенной кросс-валидации."""
    section("Смещение результата подбора гиперпараметров")

    X_full, y_full = load_breast_cancer(return_X_y=True)
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_full), size=n_objects, replace=False)
    X, y = X_full[idx], y_full[idx]

    pipe = Pipeline([("scale", StandardScaler()),
                     ("clf", SVC())])
    n_configs = len(PARAM_GRID["clf__C"]) * len(PARAM_GRID["clf__gamma"])

    print(f"  Объектов: {n_objects}, конфигураций: {n_configs}, "
          f"независимых запусков: {n_trials}.")
    print(f"  Обучений на запуск: {4 * n_configs} без вложенности "
          f"и {4 * 4 * n_configs} с вложенностью.\n")

    best_scores, nested_scores = [], []
    for trial in range(n_trials):
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=trial)
        outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=trial)

        gs = GridSearchCV(pipe, PARAM_GRID, cv=inner,
                          scoring="accuracy", n_jobs=-1)
        gs.fit(X, y)
        best_scores.append(gs.best_score_)

        nested_scores.append(cross_val_score(gs, X, y, cv=outer,
                                             scoring="accuracy",
                                             n_jobs=-1).mean())

    best_scores = np.array(best_scores)
    nested_scores = np.array(nested_scores)
    diff = best_scores - nested_scores

    print("  запуск | best_score_ | вложенная CV | разность")
    for i in range(min(8, n_trials)):
        print(f"    {i + 1:>2}   |   {best_scores[i]:.4f}    "
              f"|    {nested_scores[i]:.4f}    |  {diff[i]:+.4f}")
    print(f"\n  best_score_ : {best_scores.mean():.4f} ± {best_scores.std():.4f}")
    print(f"  вложенная CV: {nested_scores.mean():.4f} "
          f"± {nested_scores.std():.4f}")
    print(f"  Среднее смещение: {100 * diff.mean():+.2f} п.п.; "
          f"положительно в {100 * (diff > 0).mean():.0f} % запусков.")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4),
                             gridspec_kw={"width_ratios": [1.6, 1]})

    ax = axes[0]
    t = np.arange(1, n_trials + 1)
    ax.plot(t, best_scores, "-o", color=ORANGE, label="gs.best_score_")
    ax.plot(t, nested_scores, "-o", color=BLUE,
            label="вложенная кросс-валидация")
    ax.fill_between(t, nested_scores, best_scores,
                    where=best_scores >= nested_scores,
                    color=RED, alpha=0.12, label="разность оценок")
    ax.set_title("(а) Результаты 25 независимых запусков",
                 fontsize=12, loc="left")
    ax.set_xlabel("номер запуска")
    ax.set_ylabel("accuracy")
    ax.legend(loc="lower center", ncol=3, fontsize=9, frameon=True,
              framealpha=0.92, facecolor="#fcfcfb", edgecolor="none")

    ax = axes[1]
    ax.bar(["gs.best_score_", "вложенная CV"],
           [best_scores.mean(), nested_scores.mean()],
           yerr=[best_scores.std(), nested_scores.std()],
           color=[ORANGE, AQUA], width=0.5, zorder=3,
           error_kw=dict(ecolor=INK_2, lw=1.4, capsize=5))
    for i, (v, e) in enumerate([(best_scores.mean(), best_scores.std()),
                                (nested_scores.mean(), nested_scores.std())]):
        ax.text(i, v + e + 0.004, f"{v:.4f}", ha="center", fontsize=12,
                fontweight="bold", color=INK_2)
    lo = min(best_scores.mean(), nested_scores.mean())
    hi = max(best_scores.mean(), nested_scores.mean())
    ax.set_ylim(lo - 0.05, hi + 0.035)
    ax.set_title(f"(б) Среднее смещение {100 * diff.mean():+.2f} п.п.",
                 fontsize=12, loc="left")
    ax.set_ylabel("средняя accuracy")
    ax.grid(axis="x", visible=False)

    fig.suptitle("Рис. 9. Оптимистическое смещение результата подбора "
                 "гиперпараметров\nотносительно вложенной кросс-валидации",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "fig09_nested_cv")
    return best_scores, nested_scores


if __name__ == "__main__":
    run()

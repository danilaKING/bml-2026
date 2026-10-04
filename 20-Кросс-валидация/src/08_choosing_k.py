"""Сравнение схем оценивания относительно известного истинного значения.

Генеральная совокупность задаётся явно, поэтому истинное качество модели
измеримо. Из неё многократно извлекается рабочая выборка; по ней вычисляются
оценки девятью схемами, а истинное значение определяется на оставшихся
объектах. Для каждой схемы измеряются смещение, стандартное отклонение,
RMSE и время вычисления.
"""

import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import (LeaveOneOut, RepeatedStratifiedKFold,
                                     ShuffleSplit, StratifiedKFold,
                                     cross_val_score)
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
from common import (RANDOM_STATE, BLUE, ORANGE, AQUA, RED, VIOLET, MAGENTA,
                    GREEN, MUTED, INK_2, save, section, subsection, raz)

N_SAMPLE = 200
N_REPEATS = 60
POP_SIZE = 20000
N_BOOTSTRAP = 25

SCHEMES = [
    ("Hold-out 25 %", lambda: ShuffleSplit(1, test_size=0.25, random_state=0)),
    ("K-Fold, K=2", lambda: StratifiedKFold(2, shuffle=True, random_state=0)),
    ("K-Fold, K=3", lambda: StratifiedKFold(3, shuffle=True, random_state=0)),
    ("K-Fold, K=5", lambda: StratifiedKFold(5, shuffle=True, random_state=0)),
    ("K-Fold, K=10", lambda: StratifiedKFold(10, shuffle=True, random_state=0)),
    ("K-Fold, K=20", lambda: StratifiedKFold(20, shuffle=True, random_state=0)),
    ("LOO (K=n)", lambda: LeaveOneOut()),
    ("ShuffleSplit ×10", lambda: ShuffleSplit(10, test_size=0.2,
                                              random_state=0)),
    ("Repeated 5×5", lambda: RepeatedStratifiedKFold(n_splits=5, n_repeats=5,
                                                     random_state=0)),
]
BOOTSTRAP_NAME = "Bootstrap OOB ×25"


def make_population():
    """Сгенерировать генеральную совокупность."""
    return make_classification(n_samples=POP_SIZE, n_features=20,
                               n_informative=6, n_redundant=4,
                               class_sep=0.9, flip_y=0.03,
                               random_state=RANDOM_STATE)


def make_model():
    """Оцениваемая модель."""
    return DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)


def bootstrap_oob(model, X, y, n_boot=N_BOOTSTRAP, rng=None):
    """Оценка out-of-bag: обучение на бутстрэп-выборке, контроль на остатке."""
    rng = rng or np.random.RandomState(0)
    n = len(y)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        oob = np.setdiff1d(np.arange(n), idx)
        if len(oob) == 0 or len(np.unique(y[idx])) < 2:
            continue
        scores.append(model.fit(X[idx], y[idx]).score(X[oob], y[oob]))
    return float(np.mean(scores))


def run_experiment():
    """Измерить свойства оценок всех схем относительно истинного значения."""
    section("Сравнение схем оценивания")

    X_pop, y_pop = make_population()
    print(f"  Генеральная совокупность: {POP_SIZE} объектов, 20 признаков.")
    print(f"  Рабочая выборка: {N_SAMPLE} объектов, повторов: {N_REPEATS}.")
    print("  Модель: решающее дерево глубины 6.\n")

    records = {name: {"est": [], "truth": [], "time": []}
               for name, _ in SCHEMES}
    records[BOOTSTRAP_NAME] = {"est": [], "truth": [], "time": []}

    rng = np.random.RandomState(RANDOM_STATE)
    for rep in range(N_REPEATS):
        idx = rng.choice(POP_SIZE, N_SAMPLE, replace=False)
        rest = np.setdiff1d(np.arange(POP_SIZE), idx)
        X, y = X_pop[idx], y_pop[idx]
        X_rest, y_rest = X_pop[rest], y_pop[rest]

        truth = make_model().fit(X, y).score(X_rest, y_rest)

        for name, make_cv in SCHEMES:
            started = time.perf_counter()
            scores = cross_val_score(make_model(), X, y, cv=make_cv(),
                                     scoring="accuracy")
            records[name]["time"].append(time.perf_counter() - started)
            records[name]["est"].append(scores.mean())
            records[name]["truth"].append(truth)

        started = time.perf_counter()
        estimate = bootstrap_oob(make_model(), X, y,
                                 rng=np.random.RandomState(rep))
        records[BOOTSTRAP_NAME]["est"].append(estimate)
        records[BOOTSTRAP_NAME]["truth"].append(truth)
        records[BOOTSTRAP_NAME]["time"].append(time.perf_counter() - started)

        if (rep + 1) % 20 == 0:
            print(f"    выполнено {rep + 1} / {N_REPEATS}")

    rows = []
    for name, data in records.items():
        est = np.array(data["est"])
        truth = np.array(data["truth"])
        rows.append({
            "схема": name,
            "оценка": est.mean(),
            "истина": truth.mean(),
            "смещение": (est - truth).mean(),
            "ст. откл.": est.std(),
            "RMSE": np.sqrt(((est - truth) ** 2).mean()),
            "время, с": np.mean(data["time"]),
        })
    table = pd.DataFrame(rows)

    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\n  Смещение — систематическая ошибка; отрицательное значение "
          "означает занижение\n  оценки вследствие обучения на урезанной "
          "выборке. RMSE объединяет смещение\n  и дисперсию и служит "
          "основным критерием сравнения.")

    value = lambda name, column: table[table["схема"] == name][column].iloc[0]
    best = table.loc[table["RMSE"].idxmin(), "схема"]

    subsection("Выводы")
    print(f"  1. Однократное разбиение: ст. отклонение "
          f"{value('Hold-out 25 %', 'ст. откл.'):.4f} против "
          f"{value('K-Fold, K=5', 'ст. откл.'):.4f} при K = 5, то есть в "
          f"{value('Hold-out 25 %', 'ст. откл.') / value('K-Fold, K=5', 'ст. откл.'):.1f} "
          "раза выше\n     при сопоставимом смещении.")

    print("\n  2. Смещение убывает с ростом K:")
    for name in ["K-Fold, K=2", "K-Fold, K=3", "K-Fold, K=5",
                 "K-Fold, K=10", "K-Fold, K=20", "LOO (K=n)"]:
        print(f"       {name:<14} {value(name, 'смещение'):+.4f}")
    print("     Причина: при малом K обучающая выборка существенно меньше "
          "полной.")

    ratio = N_SAMPLE / 5
    time_ratio = value("LOO (K=n)", "время, с") / value("K-Fold, K=5",
                                                        "время, с")
    print(f"\n  3. Контроль по одному объекту: смещение "
          f"{value('LOO (K=n)', 'смещение'):+.4f} против "
          f"{value('K-Fold, K=5', 'смещение'):+.4f} при K = 5,")
    print(f"     ст. отклонение {value('LOO (K=n)', 'ст. откл.'):.4f} против "
          f"{value('K-Fold, K=5', 'ст. откл.'):.4f}, "
          f"обучений {N_SAMPLE} вместо 5 —")
    print(f"     в {ratio:.0f} {raz(ratio)} больше (по времени "
          f"в {time_ratio:.0f} {raz(time_ratio)}). RMSE при этом сопоставима.")

    print(f"\n  4. Минимальная RMSE достигается схемой «{best}»: "
          f"{value(best, 'RMSE'):.4f} против "
          f"{value('K-Fold, K=5', 'RMSE'):.4f} при K = 5.")
    print("     Схемы с повторами усредняют по 25-50 разбиениям, что снижает "
          "дисперсию оценки.")
    return table, records


def plot_dependence_on_k(table):
    """Рис. 16: свойства оценки и стоимость как функции числа блоков."""
    names = ["K-Fold, K=2", "K-Fold, K=3", "K-Fold, K=5", "K-Fold, K=10",
             "K-Fold, K=20", "LOO (K=n)"]
    value = lambda name, column: table[table["схема"] == name][column].iloc[0]
    bias = [value(n, "смещение") for n in names]
    std = [value(n, "ст. откл.") for n in names]
    rmse = [value(n, "RMSE") for n in names]
    elapsed = [value(n, "время, с") for n in names]
    ticks = ["2", "3", "5", "10", "20", f"LOO\n({N_SAMPLE})"]
    xs = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    ax.plot(xs, np.abs(bias), "-o", color=ORANGE, label="|смещение|")
    ax.plot(xs, std, "-o", color=BLUE, label="стандартное отклонение")
    ax.plot(xs, rmse, "-o", color=RED, lw=2.6, label="RMSE")
    best_i = int(np.argmin(rmse))
    ax.plot([xs[best_i]], [rmse[best_i]], "*", color=RED, markersize=20,
            zorder=6)
    ax.annotate(f"минимум RMSE при K = {ticks[best_i]}",
                xy=(xs[best_i], rmse[best_i]),
                xytext=(xs[best_i] + 0.30, rmse[best_i] + 0.014),
                color=INK_2, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.set_xticks(xs)
    ax.set_xticklabels(ticks)
    ax.set_xlabel("число блоков $K$")
    ax.set_ylabel("ошибка оценивания")
    ax.set_ylim(-0.004, 0.079)
    ax.set_title("(а) Смещение, стандартное отклонение и RMSE",
                 fontsize=12, loc="left")
    ax.legend(loc="upper center", ncol=3, fontsize=9.5, frameon=False)

    ax = axes[1]
    ax.plot(xs, elapsed, "-o", color=VIOLET)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(ticks)
    ax.set_xlabel("число блоков $K$")
    ax.set_ylabel("время одной оценки, с (логарифмическая шкала)")
    ax.set_title("(б) Вычислительная стоимость", fontsize=12, loc="left")
    for x, v in zip(xs, elapsed):
        ax.text(x, v * 1.55, f"{v * 1000:.0f} мс", ha="center", color=INK_2,
                fontsize=9)
    ax.set_ylim(min(elapsed) * 0.5, max(elapsed) * 4)

    fig.suptitle("Рис. 16. Зависимость смещения, стандартного отклонения, "
                 "RMSE\nи времени вычисления оценки от числа блоков K",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    save(fig, "fig16_choosing_k")


def plot_error_distribution(records):
    """Рис. 17: распределение ошибки оценивания по схемам."""
    names = ["Hold-out 25 %", "K-Fold, K=2", "K-Fold, K=5", "K-Fold, K=10",
             "LOO (K=n)", "ShuffleSplit ×10", "Repeated 5×5", BOOTSTRAP_NAME]
    colors = [MUTED, ORANGE, AQUA, BLUE, VIOLET, MAGENTA, GREEN, RED]

    fig, ax = plt.subplots(figsize=(11, 5.0))
    for i, (name, color) in enumerate(zip(names, colors)):
        errors = (np.array(records[name]["est"])
                  - np.array(records[name]["truth"]))
        jitter = np.random.RandomState(i).normal(0, 0.07, len(errors))
        ax.scatter(errors, np.full(len(errors), i) + jitter,
                   color=color, alpha=0.55, s=26, lw=0)
        ax.scatter([errors.mean()], [i], color=color, s=170, marker="D",
                   edgecolor="white", linewidth=1.6, zorder=5)
        ax.text(0.20, i, f"смещение {errors.mean():+.3f}    "
                         f"std {errors.std():.3f}",
                va="center", color=INK_2, fontsize=10)
    ax.axvline(0, color=RED, lw=2, ls="--", zorder=1)
    ax.text(0.004, len(names) - 0.35, "истинное значение", color=RED,
            fontsize=10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlim(-0.20, 0.36)
    ax.invert_yaxis()
    ax.set_xlabel("ошибка оценивания (оценка − истинное значение)")
    ax.set_title("Рис. 17. Распределение ошибки оценивания для девяти схем\n"
                 "(60 повторов; точка — один повтор, ромб — среднее)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig17_scheme_targets")


if __name__ == "__main__":
    table, records = run_experiment()
    plot_dependence_on_k(table)
    plot_error_distribution(records)

"""Схемы разбиения выборки и области их применения.

Модуль строит структуру разбиений шести схем на одной выборке и оценивает
последствия выбора неподходящей схемы в трёх постановках: дисбаланс классов,
наличие групп связанных объектов и наличие временного порядка.
"""

import warnings

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GroupKFold, KFold, ShuffleSplit,
                                     StratifiedKFold, TimeSeriesSplit,
                                     cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from common import (RANDOM_STATE, BLUE, ORANGE, AQUA, RED, VIOLET, MAGENTA,
                    YELLOW, GREEN, MUTED, INK_2, GRID, save, section)

GROUP_COLORS = [VIOLET, AQUA, YELLOW, RED, GREEN, BLUE, ORANGE, MAGENTA]


def splitting_schemes():
    """Рис. 12: структура разбиений шести схем на одной выборке."""
    section("1. Структура разбиений")

    n = 40
    rng = np.random.RandomState(RANDOM_STATE)
    y = np.array([1] * 8 + [0] * 32)
    groups = np.repeat(np.arange(8), 5)
    X = rng.normal(size=(n, 3))

    schemes = [
        ("KFold(5), shuffle=False", KFold(5), None),
        ("KFold(5), shuffle=True",
         KFold(5, shuffle=True, random_state=RANDOM_STATE), None),
        ("StratifiedKFold(5)",
         StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), None),
        ("GroupKFold(5)", GroupKFold(5), groups),
        ("ShuffleSplit(5, train_size=0.6, test_size=0.2)",
         ShuffleSplit(5, train_size=0.6, test_size=0.2,
                      random_state=RANDOM_STATE), None),
        ("TimeSeriesSplit(5)", TimeSeriesSplit(5), None),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 8.2))
    for ax, (title, splitter, g) in zip(axes.ravel(), schemes):
        for i, (train, test) in enumerate(splitter.split(X, y, g)):
            row = np.full(n, np.nan)
            row[train] = 0
            row[test] = 1
            ax.scatter(range(n), [i] * n,
                       c=[BLUE if v == 0 else ORANGE if v == 1 else "#ffffff"
                          for v in row], marker="s", s=46, lw=0)
        ax.scatter(range(n), [-1.3] * n,
                   c=[MAGENTA if c == 1 else GRID for c in y],
                   marker="s", s=46, lw=0)
        ax.scatter(range(n), [-2.2] * n,
                   c=[GROUP_COLORS[g_] for g_ in groups],
                   marker="s", s=46, lw=0)
        n_splits = splitter.get_n_splits(X, y, g)
        ax.set_yticks(list(range(n_splits)) + [-1.3, -2.2])
        ax.set_yticklabels([f"блок {i + 1}" for i in range(n_splits)]
                           + ["класс", "группа"], fontsize=9)
        ax.set_ylim(-2.9, n_splits - 0.3)
        ax.invert_yaxis()
        ax.set_xlim(-1, n)
        ax.set_title(title, fontsize=11.5, loc="left")
        ax.grid(False)
        ax.set_xticks([0, 10, 20, 30, 39])
        ax.set_xlabel("номер объекта", fontsize=9)

    handles = [Patch(facecolor=BLUE, label="обучение"),
               Patch(facecolor=ORANGE, label="тест"),
               Patch(facecolor="#ffffff", edgecolor=GRID,
                     label="не используется")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Рис. 12. Структура разбиений для шести схем "
                 "кросс-валидации (n = 40)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    save(fig, "fig12_cv_strategies_map")

    print("  Построены шесть схем разбиения одной и той же выборки.")
    print("  KFold без перемешивания следует исходному порядку объектов;")
    print("  StratifiedKFold сохраняет доли классов; GroupKFold не разрывает "
          "группу;")
    print("  ShuffleSplit формирует независимые разбиения; TimeSeriesSplit "
          "использует только предшествующие наблюдения.")


def stratification(n_repeats=20):
    """Рис. 13: состав блоков и разброс оценки при дисбалансе классов."""
    section("2. Стратифицированное разбиение")

    X, y = make_classification(n_samples=300, n_features=12, n_informative=5,
                               weights=[0.94, 0.06], flip_y=0.01,
                               random_state=RANDOM_STATE)
    print(f"  Распределение классов: {np.bincount(y)}, "
          f"доля редкого класса {100 * y.mean():.1f} %.")

    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=5000))

    plain, strat = [], []
    plain_counts, strat_counts = [], []
    undefined = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        for seed in range(n_repeats):
            kf = KFold(5, shuffle=True, random_state=seed)
            sk = StratifiedKFold(5, shuffle=True, random_state=seed)
            scores_plain = cross_val_score(model, X, y, cv=kf,
                                           scoring="roc_auc")
            scores_strat = cross_val_score(model, X, y, cv=sk,
                                           scoring="roc_auc")
            undefined += int(np.isnan(scores_plain).sum())
            plain.append(np.nanmean(scores_plain))
            strat.append(scores_strat.mean())
            plain_counts += [int(y[test].sum()) for _, test in kf.split(X, y)]
            strat_counts += [int(y[test].sum()) for _, test in sk.split(X, y)]

    plain, strat = np.array(plain), np.array(strat)
    print(f"\n  Объектов редкого класса в контрольном блоке "
          f"({n_repeats * 5} блоков):")
    print(f"    KFold          : от {min(plain_counts)} до {max(plain_counts)}")
    print(f"    StratifiedKFold: от {min(strat_counts)} до {max(strat_counts)}")
    print(f"  Блоков без редкого класса (ROC-AUC не определён): "
          f"{undefined} у KFold, "
          f"{sum(1 for c in strat_counts if c == 0)} у StratifiedKFold.")
    print(f"\n  ROC-AUC по {n_repeats} повторам:")
    print(f"    KFold          : {plain.mean():.4f} ± {plain.std():.4f}")
    print(f"    StratifiedKFold: {strat.mean():.4f} ± {strat.std():.4f}")
    print(f"  Отношение стандартных отклонений: "
          f"{plain.std() / max(strat.std(), 1e-12):.1f}")
    print("  При передаче классификатору параметра cv=5 стратификация "
          "применяется по умолчанию.")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    bins = np.arange(min(plain_counts) - 0.5, max(plain_counts) + 1.5)
    ax.hist(plain_counts, bins=bins, color=ORANGE, alpha=0.85,
            edgecolor="white", label="KFold")
    ax.hist(strat_counts, bins=bins, color=AQUA, alpha=0.9,
            edgecolor="white", label="StratifiedKFold")
    ax.set_title("(а) Состав тестовых блоков", fontsize=12, loc="left")
    ax.set_xlabel("число объектов редкого класса в блоке")
    ax.set_ylabel("число блоков")
    ax.legend()

    ax = axes[1]
    box = ax.boxplot([plain, strat], tick_labels=["KFold", "StratifiedKFold"],
                     widths=0.45, patch_artist=True,
                     medianprops=dict(color=INK_2, lw=2),
                     whiskerprops=dict(color=MUTED),
                     capprops=dict(color=MUTED),
                     flierprops=dict(markeredgecolor=MUTED, markersize=5))
    for patch, color in zip(box["boxes"], [ORANGE, AQUA]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor(INK_2)
    ax.set_title("(б) Разброс оценки по 20 повторам", fontsize=12, loc="left")
    ax.set_ylabel("ROC-AUC")
    ax.set_xlabel("схема разбиения")
    ax.grid(axis="x", visible=False)

    fig.suptitle("Рис. 13. Состав тестовых блоков и разброс оценки ROC-AUC\n"
                 "при KFold и StratifiedKFold (доля редкого класса 6 %)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    save(fig, "fig13_stratified")


def grouped_objects():
    """Рис. 14: оценка при разбиении по объектам и по группам."""
    section("3. Разбиение при наличии групп связанных объектов")

    rng = np.random.RandomState(RANDOM_STATE)
    n_patients, per_patient = 40, 10
    patient_effect = rng.normal(0, 3.0, size=(n_patients, 6))
    patient_label = rng.randint(0, 2, size=n_patients)

    X, y, groups = [], [], []
    for p in range(n_patients):
        signal = 1.2 * patient_label[p]
        X.append(patient_effect[p] + signal
                 + rng.normal(0, 1.0, size=(per_patient, 6)))
        y += [patient_label[p]] * per_patient
        groups += [p] * per_patient
    X = np.vstack(X)
    y = np.array(y)
    groups = np.array(groups)

    print(f"  {n_patients} пациентов по {per_patient} снимков, "
          f"всего {len(y)} объектов.")
    print("  Целевая переменная определена на уровне пациента; снимки одного "
          "пациента статистически зависимы.\n")

    model = RandomForestClassifier(n_estimators=200,
                                   random_state=RANDOM_STATE)
    by_object = cross_val_score(
        model, X, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy", n_jobs=-1)
    by_group = cross_val_score(model, X, y, groups=groups, cv=GroupKFold(5),
                               scoring="accuracy", n_jobs=-1)

    print(f"  Разбиение по снимкам:   {by_object.mean():.4f} "
          f"± {by_object.std():.4f}")
    print(f"  Разбиение по пациентам: {by_group.mean():.4f} "
          f"± {by_group.std():.4f}")
    print(f"  Смещение: {100 * (by_object.mean() - by_group.mean()):+.1f} п.п.")
    print("  При разбиении по снимкам модель различает пациентов, а не классы.")

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    bars = ax.bar(["StratifiedKFold\n(разбиение по снимкам)",
                   "GroupKFold\n(разбиение по пациентам)"],
                  [by_object.mean(), by_group.mean()],
                  yerr=[by_object.std(), by_group.std()],
                  color=[RED, AQUA], width=0.5, zorder=3,
                  error_kw=dict(ecolor=INK_2, lw=1.4, capsize=5))
    for bar, v, e in zip(bars, [by_object.mean(), by_group.mean()],
                         [by_object.std(), by_group.std()]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + e + 0.025, f"{v:.3f}",
                ha="center", fontsize=13, fontweight="bold", color=INK_2)
    ax.annotate("", xy=(0.35, by_object.mean()),
                xytext=(0.35, by_group.mean()),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.8))
    ax.text(0.42, (by_object.mean() + by_group.mean()) / 2,
            f"смещение\n{100 * (by_object.mean() - by_group.mean()):.0f} п.п.",
            color=RED, fontsize=11, va="center")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("accuracy")
    ax.set_title("Рис. 14. Оценка accuracy при разбиении по объектам\n"
                 "и по группам (40 пациентов × 10 снимков)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig14_group_kfold")


def temporal_order():
    """Рис. 15: сравнение перемешанного и хронологического разбиения."""
    section("4. Хронологическое разбиение")

    rng = np.random.RandomState(RANDOM_STATE)
    n = 364
    t = np.arange(n)
    series = (100
              + 0.25 * t
              + 18 * np.sin(2 * np.pi * t / 7)
              + np.cumsum(rng.normal(0, 1.0, n))
              + rng.normal(0, 3, n))

    X = np.column_stack([t % 7,
                         t // 7,
                         np.sin(2 * np.pi * t / 7),
                         np.cos(2 * np.pi * t / 7)])
    y = series

    model = RandomForestRegressor(n_estimators=200,
                                  random_state=RANDOM_STATE)
    shuffled = -cross_val_score(
        model, X, y, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_mean_absolute_error", n_jobs=-1)
    ordered = -cross_val_score(model, X, y, cv=TimeSeriesSplit(5),
                               scoring="neg_mean_absolute_error", n_jobs=-1)

    print(f"  Ряд: {n} ежедневных наблюдений; тренд, недельная сезонность, "
          "случайное блуждание.")
    print("  Признаки: день недели, номер недели, гармоники недельного "
          "периода.\n")
    print(f"  KFold, shuffle=True: MAE = {shuffled.mean():.2f}")
    print(f"  TimeSeriesSplit    : MAE = {ordered.mean():.2f}")
    print(f"  Отношение оценок: {ordered.mean() / shuffled.mean():.1f}")
    print("  При перемешивании решается задача интерполяции пропусков "
          "в известном прошлом,")
    print("  а не задача прогнозирования. Параметр gap задаёт зазор между "
          "обучающим\n  и контрольным интервалами.")

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.0),
                             gridspec_kw={"height_ratios": [1.4, 1]})

    ax = axes[0]
    ax.plot(t, series, color=BLUE, lw=1.4)
    for i, (_, test) in enumerate(TimeSeriesSplit(5).split(X)):
        ax.axvspan(test[0], test[-1], color=ORANGE, alpha=0.10 + 0.035 * i)
        ax.text((test[0] + test[-1]) / 2, series.max() * 1.03, f"тест {i + 1}",
                ha="center", color=INK_2, fontsize=9)
    ax.set_ylim(series.min() * 0.9, series.max() * 1.12)
    ax.set_title("(а) Ряд и границы тестовых интервалов TimeSeriesSplit",
                 fontsize=12, loc="left")
    ax.set_xlabel("день")
    ax.set_ylabel("продажи, шт.")

    ax = axes[1]
    bars = ax.bar(["KFold, shuffle=True", "TimeSeriesSplit"],
                  [shuffled.mean(), ordered.mean()],
                  color=[RED, AQUA], width=0.45, zorder=3)
    for bar, v in zip(bars, [shuffled.mean(), ordered.mean()]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4, f"MAE = {v:.2f}",
                ha="center", fontsize=12, fontweight="bold", color=INK_2)
    ax.set_ylim(0, max(shuffled.mean(), ordered.mean()) * 1.30)
    ax.set_ylabel("MAE")
    ax.set_title(f"(б) Отношение оценок MAE: "
                 f"{ordered.mean() / shuffled.mean():.1f}",
                 fontsize=12, loc="left")
    ax.grid(axis="x", visible=False)

    fig.suptitle("Рис. 15. Схема TimeSeriesSplit и сравнение MAE "
                 "при перемешанном\nи хронологическом разбиении",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save(fig, "fig15_timeseries")


if __name__ == "__main__":
    splitting_schemes()
    stratification()
    grouped_objects()
    temporal_order()

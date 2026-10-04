"""Программная реализация K-блочной кросс-валидации.

Модуль содержит прямую реализацию процедуры, не использующую сплиттеры
scikit-learn, и сверяет её результат со штатными функциями библиотеки:
cross_val_score, cross_validate и cross_val_predict.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict,
                                     cross_val_score, cross_validate)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from common import (RANDOM_STATE, BLUE, ORANGE, AQUA, INK_2,
                    save, section, subsection)

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

X, y = load_breast_cancer(return_X_y=True)
model = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=5000))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def kfold_cross_val(estimator, X, y, K=5, random_state=0):
    """K-блочная кросс-валидация без использования сплиттеров scikit-learn.

    Индексы объектов перемешиваются и разделяются на K блоков; каждый блок
    поочерёдно объявляется контрольным, модель обучается на объединении
    остальных. Возвращает вектор из K оценок accuracy.
    """
    idx = np.arange(len(y))
    np.random.RandomState(random_state).shuffle(idx)
    folds = np.array_split(idx, K)

    scores = []
    for k in range(K):
        test = folds[k]
        train = np.concatenate([folds[j] for j in range(K)
                                if j != k])
        estimator.fit(X[train], y[train])
        pred = estimator.predict(X[test])
        scores.append(accuracy_score(y[test], pred))
    return np.array(scores)


def basic_interface():
    """Базовый вызов cross_val_score."""
    section("1. Базовый интерфейс scikit-learn")

    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    print(f"  Выборка: {X.shape[0]} объектов, {X.shape[1]} признаков, "
          f"классы {np.bincount(y)}")
    print(f"  Оценки по блокам: {np.round(scores, 4)}")
    print(f"  accuracy = {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Размах между блоками: {scores.max() - scores.min():.4f}")
    return scores


def extended_interface():
    """Несколько метрик за один проход и оценки на обучающих блоках."""
    section("2. Функция cross_validate")

    res = cross_validate(model, X, y, cv=cv, scoring=METRICS,
                         return_train_score=True, n_jobs=-1)

    table = pd.DataFrame([{
        "метрика": m,
        "контрольная": res[f"test_{m}"].mean(),
        "ст. откл.": res[f"test_{m}"].std(),
        "обучающая": res[f"train_{m}"].mean(),
        "разность": res[f"train_{m}"].mean() - res[f"test_{m}"].mean(),
    } for m in METRICS])
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  Среднее время обучения на блоке: "
          f"{res['fit_time'].mean() * 1000:.1f} мс")
    print("  Разность оценок на обучающей и контрольной выборках "
          "характеризует переобучение.")
    return res


def verify_implementation(reference):
    """Сверка собственной реализации со штатными функциями библиотеки."""
    section("3. Сверка собственной реализации")

    subsection("3.1. Цикл по разбиениям, полученным от StratifiedKFold")
    manual = []
    print("  блок | объём обучающей | объём контрольной | accuracy")
    for k, (train, test) in enumerate(cv.split(X, y), start=1):
        model.fit(X[train], y[train])
        score = accuracy_score(y[test], model.predict(X[test]))
        manual.append(score)
        print(f"   {k}   |      {len(train):>4d}       "
              f"|       {len(test):>4d}       |  {score:.4f}")
    manual = np.array(manual)
    print(f"\n  Среднее по собственному циклу: {manual.mean():.10f}")
    print(f"  Среднее cross_val_score      : {reference.mean():.10f}")
    print(f"  Значения совпадают: {np.allclose(manual, reference)}")

    subsection("3.2. Реализация kfold_cross_val")
    own = kfold_cross_val(model, X, y, K=5, random_state=0)
    ref = cross_val_score(model, X, y, scoring="accuracy",
                          cv=KFold(n_splits=5, shuffle=True, random_state=0))
    print(f"  kfold_cross_val              : {np.round(own, 6)}")
    print(f"  cross_val_score(cv=KFold(...)): {np.round(ref, 6)}")
    print(f"  Максимальное расхождение     : {np.abs(own - ref).max():.2e}")
    return manual


def out_of_fold_predictions():
    """Предсказания, полученные вне обучающих блоков."""
    section("4. Функция cross_val_predict")

    oof = cross_val_predict(model, X, y, cv=cv)
    print(f"  Получено {len(oof)} предсказаний, каждое сделано моделью, "
          "не наблюдавшей данный объект.")
    print(f"  accuracy по этим предсказаниям: {accuracy_score(y, oof):.4f}")
    print("  Величина не совпадает со средним cross_val_score, поскольку "
          "блоки имеют разный размер.")
    print("  Назначение: анализ ошибок, построение ROC-кривой, стекинг.")
    return oof


def plot_results(scores, res):
    """Рис. 4: оценки по блокам и сводные значения метрик."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ks = np.arange(1, len(scores) + 1)
    ax.bar(ks, scores, color=BLUE, width=0.55, zorder=3)
    ax.axhline(scores.mean(), color=ORANGE, lw=2, zorder=4,
               label=f"среднее = {scores.mean():.4f}")
    ax.fill_between([0.4, len(scores) + 0.6],
                    scores.mean() - scores.std(), scores.mean() + scores.std(),
                    color=ORANGE, alpha=0.14, zorder=2,
                    label=f"± std = {scores.std():.4f}")
    for k, s in zip(ks, scores):
        ax.text(k, s + 0.004, f"{s:.3f}", ha="center", color=INK_2, fontsize=10)
    ax.set_xlim(0.4, len(scores) + 0.6)
    ax.set_ylim(0.9, 1.02)
    ax.set_xticks(ks)
    ax.set_title("(а) Оценки по блокам", fontsize=12, loc="left")
    ax.set_xlabel("номер блока $k$")
    ax.set_ylabel("accuracy")
    ax.legend(loc="lower right")

    ax = axes[1]
    means = [res[f"test_{m}"].mean() for m in METRICS]
    errs = [res[f"test_{m}"].std() for m in METRICS]
    ypos = np.arange(len(METRICS))[::-1]
    ax.barh(ypos, means, xerr=errs, color=AQUA, height=0.55, zorder=3,
            error_kw=dict(ecolor=INK_2, lw=1.4, capsize=4))
    for yp, m in zip(ypos, means):
        ax.text(0.903, yp, f"{m:.3f}", va="center", ha="left",
                color="white", fontsize=11, fontweight="bold", zorder=5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(METRICS)
    ax.set_xlim(0.9, 1.005)
    ax.set_title("(б) Значения метрик", fontsize=12, loc="left")
    ax.set_xlabel("значение метрики, среднее ± стандартное отклонение")
    ax.grid(axis="y", visible=False)

    fig.suptitle("Рис. 4. Результаты 5-блочной стратифицированной "
                 "кросс-валидации\n(breast_cancer, логистическая регрессия)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "fig04_cross_val_basics")


if __name__ == "__main__":
    scores = basic_interface()
    res = extended_interface()
    verify_implementation(scores)
    out_of_fold_predictions()
    plot_results(scores, res)

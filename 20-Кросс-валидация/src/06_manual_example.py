"""Численный пример: кросс-валидация на выборке из шести наблюдений.

Выборка y = (2, 4, 6, 8, 10, 12); модель — константный прогноз, равный
среднему обучающей выборки; функция потерь — MAE. Рассматриваются разбиение
на три блока при shuffle=False и shuffle=True, контроль по одному объекту
и три варианта однократного разбиения. Результаты сверяются с scikit-learn.
"""

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import (BLUE, ORANGE, AQUA, RED, MUTED, INK_2,
                    save, section, subsection)

Y = np.array([2, 4, 6, 8, 10, 12], dtype=float)

FOLDS_SORTED = [(Y[[0, 1]], Y[[2, 3, 4, 5]]),
                (Y[[2, 3]], Y[[0, 1, 4, 5]]),
                (Y[[4, 5]], Y[[0, 1, 2, 3]])]

FOLDS_SHUFFLED = [(np.array([2., 10.]), np.array([4., 12., 6., 8.])),
                  (np.array([4., 12.]), np.array([2., 10., 6., 8.])),
                  (np.array([6., 8.]), np.array([2., 10., 4., 12.]))]

HOLDOUTS = [("{10, 12}", np.array([10., 12.]), np.array([2., 4., 6., 8.])),
            ("{4, 10}", np.array([4., 10.]), np.array([2., 6., 8., 12.])),
            ("{2, 12}", np.array([2., 12.]), np.array([4., 6., 8., 10.]))]


def fold_loss(test_values, train_values, title):
    """Вычислить и вывести значение MAE на одном блоке."""
    prediction = train_values.mean()
    errors = np.abs(test_values - prediction)
    train_str = " + ".join(f"{v:g}" for v in train_values)
    print(f"    {title}")
    print(f"      обучение {{{', '.join(f'{v:g}' for v in train_values)}}}: "
          f"прогноз = ({train_str}) / {len(train_values)} = {prediction:g}")
    print(f"      контроль {{{', '.join(f'{v:g}' for v in test_values)}}}: "
          + ", ".join(f"|{t:g} - {prediction:g}| = {e:g}"
                      for t, e in zip(test_values, errors)))
    print(f"      MAE = ({' + '.join(f'{e:g}' for e in errors)}) / "
          f"{len(errors)} = {errors.mean():g}")
    return errors.mean()


def kfold_sorted():
    """Разбиение на три блока в исходном порядке наблюдений."""
    section("1. Разбиение на K = 3 блока, shuffle=False")

    print("  Блоки: D1 = {2, 4}, D2 = {6, 8}, D3 = {10, 12}\n")
    losses = [fold_loss(test, train, f"k = {k}")
              for k, (test, train) in enumerate(FOLDS_SORTED, start=1)]

    estimate = float(np.mean(losses))
    print(f"\n  CV(3) = ({' + '.join(f'{v:g}' for v in losses)}) / 3 = "
          f"{sum(losses):g} / 3 = {estimate:.2f}")
    print(f"  Значения по блокам: от {min(losses):g} до {max(losses):g}; "
          "блоки неоднородны.")
    return losses, estimate


def kfold_shuffled():
    """Тот же расчёт после перестановки наблюдений."""
    section("2. Разбиение на K = 3 блока, shuffle=True")

    print("  Порядок: 2, 10, 4, 12, 6, 8")
    print("  Блоки: D1 = {2, 10}, D2 = {4, 12}, D3 = {6, 8}\n")
    losses = [fold_loss(test, train, f"k = {k}")
              for k, (test, train) in enumerate(FOLDS_SHUFFLED, start=1)]

    estimate = float(np.mean(losses))
    print(f"\n  CV(3) = ({' + '.join(f'{v:g}' for v in losses)}) / 3 = "
          f"{estimate:.2f}")
    print("  Исходные данные упорядочены по значению целевой переменной, "
          "поэтому при\n  shuffle=False блоки однородны, а обучающая выборка "
          "систематически смещена.")
    print("  Значение по умолчанию в scikit-learn: shuffle=False.")
    return losses, estimate


def leave_one_out():
    """Контроль по одному объекту."""
    section("3. Контроль по одному объекту (K = n)")

    print("  объект | обучающая выборка        | прогноз | ошибка")
    errors = []
    for i in range(len(Y)):
        train = np.delete(Y, i)
        prediction = train.mean()
        error = abs(Y[i] - prediction)
        errors.append(error)
        print(f"   {Y[i]:>4.0f}  | {', '.join(f'{v:g}' for v in train):<24}|"
              f"  {prediction:>5.1f}  |  {error:.1f}")

    estimate = float(np.mean(errors))
    print(f"\n  CV(6) = ({' + '.join(f'{e:g}' for e in errors)}) / 6 = "
          f"{sum(errors):g} / 6 = {estimate:.2f}")
    print("  Число обучений вдвое больше, чем при K = 3.")
    return errors, estimate


def holdout_estimates():
    """Три варианта однократного разбиения."""
    section("4. Однократное разбиение")

    results = []
    for title, test, train in HOLDOUTS:
        results.append((title, fold_loss(test, train,
                                         f"контрольная пара {title}")))
        print()
    values = [v for _, v in results]
    print(f"  Оценки: от {min(values):g} до {max(values):g} "
          "при одной и той же модели и выборке.")
    return results


def verify(cv_sorted, cv_shuffled, cv_loo, holdouts):
    """Истинное значение MAE и сверка расчёта со scikit-learn."""
    section("5. Истинное значение и сверка со scikit-learn")

    prediction = Y.mean()
    truth = np.abs(Y - prediction).mean()
    print(f"  Модель на всей выборке даёт прогноз {prediction:g}.")
    print(f"  MAE = ({' + '.join(f'{abs(v - prediction):g}' for v in Y)}) / 6 "
          f"= {truth:.2f}\n")

    print("  схема                          оценка   отклонение")
    print(f"    K = 3, shuffle=False         {cv_sorted:>6.2f}     "
          f"{cv_sorted - truth:+.2f}")
    print(f"    K = 3, shuffle=True          {cv_shuffled:>6.2f}     "
          f"{cv_shuffled - truth:+.2f}")
    print(f"    K = 6 (LOO)                  {cv_loo:>6.2f}     "
          f"{cv_loo - truth:+.2f}")
    for title, value in holdouts:
        print(f"    однократное {title:<16} {value:>6.2f}     "
              f"{value - truth:+.2f}")
    print("\n  Оценки смещены вверх: модель обучается на 4-5 наблюдениях "
          "вместо 6.")

    subsection("Сверка со scikit-learn")
    X = Y.reshape(-1, 1)
    model = DummyRegressor(strategy="mean")

    by_fold = -cross_val_score(model, X, Y, cv=KFold(3, shuffle=False),
                               scoring="neg_mean_absolute_error")
    loo = -cross_val_score(model, X, Y, cv=LeaveOneOut(),
                           scoring="neg_mean_absolute_error")

    print(f"  KFold(3, shuffle=False): по блокам {by_fold}, "
          f"среднее {by_fold.mean():.4f}")
    print(f"    ручной расчёт {cv_sorted:.4f}, совпадает: "
          f"{np.isclose(by_fold.mean(), cv_sorted)}")
    print(f"  LeaveOneOut: среднее {loo.mean():.4f}")
    print(f"    ручной расчёт {cv_loo:.4f}, совпадает: "
          f"{np.isclose(loo.mean(), cv_loo)}")
    return truth


def plot_folds(losses_sorted, cv_sorted, losses_shuffled, cv_shuffled):
    """Рис. 10: структура блоков и значения MAE при обоих порядках."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2))

    layouts = [
        (axes[0], [2, 4, 6, 8, 10, 12], [[0, 1], [2, 3], [4, 5]],
         losses_sorted, cv_sorted,
         "(а) shuffle=False: исходный (упорядоченный) порядок"),
        (axes[1], [2, 10, 4, 12, 6, 8], [[0, 1], [2, 3], [4, 5]],
         losses_shuffled, cv_shuffled,
         "(б) shuffle=True: порядок 2, 10, 4, 12, 6, 8"),
    ]

    for ax, values, folds, losses, estimate, title in layouts:
        for f, (test_pos, loss) in enumerate(zip(folds, losses)):
            row = 2 - f
            for j, v in enumerate(values):
                is_test = j in test_pos
                ax.add_patch(Rectangle((j, row - 0.34), 0.9, 0.68,
                                       facecolor=ORANGE if is_test else BLUE,
                                       alpha=1.0 if is_test else 0.35,
                                       edgecolor="white", linewidth=2))
                ax.text(j + 0.45, row, f"{v:g}", ha="center", va="center",
                        color="white", fontsize=13, fontweight="bold")
            train_values = [v for j, v in enumerate(values)
                            if j not in test_pos]
            ax.text(6.35, row, f"прогноз = {np.mean(train_values):g}",
                    va="center", color=INK_2, fontsize=11)
            ax.text(8.55, row, f"MAE = {loss:g}", va="center", color=INK_2,
                    fontsize=12, fontweight="bold")
            ax.text(-0.35, row, f"Блок {f + 1}", va="center", ha="right",
                    color=MUTED, fontsize=11)

        ax.plot([8.45, 9.95], [-0.30, -0.30], color=MUTED, lw=1.2)
        ax.text(8.55, -0.62, f"CV = {estimate:.2f}", color=RED, fontsize=14,
                fontweight="bold")
        ax.set_xlim(-1.5, 10.4)
        ax.set_ylim(-0.95, 2.95)
        ax.set_title(title, loc="left", fontsize=12)
        ax.axis("off")

    axes[0].add_patch(Rectangle((0, 2.62), 0.38, 0.24, facecolor=BLUE,
                                alpha=0.35))
    axes[0].text(0.52, 2.74, "обучение", va="center", color=INK_2, fontsize=10)
    axes[0].add_patch(Rectangle((2.0, 2.62), 0.38, 0.24, facecolor=ORANGE))
    axes[0].text(2.52, 2.74, "тест", va="center", color=INK_2, fontsize=10)

    fig.suptitle("Рис. 10. Разбиение на три блока и значения MAE "
                 "при shuffle=False\nи shuffle=True",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "fig10_manual_folds")


def plot_estimates(cv_sorted, cv_shuffled, cv_loo, holdouts, truth):
    """Рис. 11: оценки, полученные шестью схемами."""
    fig, ax = plt.subplots(figsize=(10, 4.4))

    labels = ["hold-out\n{10, 12}", "hold-out\n{4, 10}", "hold-out\n{2, 12}",
              "K = 3\nshuffle=False", "K = 3\nshuffle=True", "LOO\nK = 6"]
    values = [holdouts[0][1], holdouts[1][1], holdouts[2][1],
              cv_sorted, cv_shuffled, cv_loo]
    colors = [MUTED, MUTED, MUTED, ORANGE, AQUA, BLUE]

    bars = ax.bar(labels, values, color=colors, width=0.6, zorder=3)
    ax.axhline(truth, color=RED, lw=2, ls="--", zorder=4,
               label=f"истинное значение MAE = {truth:.2f}")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1, f"{v:.2f}",
                ha="center", fontsize=12, fontweight="bold", color=INK_2)
    ax.set_ylim(0, max(values) * 1.28)
    ax.set_ylabel("оценка MAE, млн руб.")
    ax.set_title("Рис. 11. Оценки MAE, полученные шестью схемами\n"
                 "на одной и той же выборке из шести наблюдений",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    save(fig, "fig11_manual_estimates")


if __name__ == "__main__":
    losses_sorted, cv_sorted = kfold_sorted()
    losses_shuffled, cv_shuffled = kfold_shuffled()
    _, cv_loo = leave_one_out()
    holdouts = holdout_estimates()
    truth = verify(cv_sorted, cv_shuffled, cv_loo, holdouts)

    plot_folds(losses_sorted, cv_sorted, losses_shuffled, cv_shuffled)
    plot_estimates(cv_sorted, cv_shuffled, cv_loo, holdouts, truth)

import numpy as np
from coral import coral_transform

Xs = np.array([[0, 0], [2, 0],    # норма
               [0, 4], [2, 4]],   # дефект
              dtype=float)
ys = np.array([0, 0, 1, 1])

Xt = np.array([[0, 0], [4, 0], [0, 2], [4, 2]], dtype=float)

print("=== Ручной пример CORAL (деление на n, без регуляризации) ===\n")
Xs_new = coral_transform(Xs, Xt, lam=0.0, ddof=0, verbose=True)


thr_before = (Xs[ys == 0, 1].max() + Xs[ys == 1, 1].min()) / 2
thr_after = (Xs_new[ys == 0, 1].max() + Xs_new[ys == 1, 1].min()) / 2
print(f"\nПорог по y до CORAL:    y > {thr_before:g}")
print(f"Порог по y после CORAL: y > {thr_after:g}")
print("Target с этим порогом:", ["дефект" if p[1] > thr_after else "норма" for p in Xt])


same = np.allclose(Xs_new, coral_transform(Xs, Xt, lam=0.0, ddof=1))
print("\nС ddof=1 результат тот же:", same)

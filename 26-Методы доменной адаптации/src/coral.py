import numpy as np


def sqrtm_sym(C, power):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-12, None)
    return (V * w ** power) @ V.T


def coral_transform(Xs, Xt, lam=1.0, ddof=1, verbose=False):
    d = Xs.shape[1]
    mu_s, mu_t = Xs.mean(axis=0), Xt.mean(axis=0)
    Cs = np.cov(Xs, rowvar=False, ddof=ddof) + lam * np.eye(d)
    Ct = np.cov(Xt, rowvar=False, ddof=ddof) + lam * np.eye(d)
    A = sqrtm_sym(Cs, -0.5) @ sqrtm_sym(Ct, 0.5)
    Xs_new = (Xs - mu_s) @ A + mu_t

    if verbose:
        np.set_printoptions(precision=3, suppress=True)
        print("1) Средние:        mu_s =", mu_s, "  mu_t =", mu_t)
        print("2) Ковариации:\n   Cs =\n", Cs, "\n   Ct =\n", Ct)
        print("3) A = Cs^(-1/2) @ Ct^(1/2) =\n", A)
        print("4) Центрированный source:\n", Xs - mu_s)
        print("5) После умножения на A:\n", (Xs - mu_s) @ A)
        print("6) После сдвига к mu_t (итог):\n", Xs_new)
        print("   Проверка: ковариация итога =\n",
              np.cov(Xs_new, rowvar=False, ddof=ddof) + lam * np.eye(d))
    return Xs_new.astype(np.float32)

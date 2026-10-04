import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

def k_rbf(X1, X2, l, sigma_f, period):
    sqdist = (X1[:, None] - X2[None, :]) ** 2
    return sigma_f ** 2 * np.exp(-0.5 * sqdist / l ** 2)


def k_linear(X1, X2, l, sigma_f, period):
    return sigma_f ** 2 * (X1[:, None] * X2[None, :] + 1.0)


def k_periodic(X1, X2, l, sigma_f, period):
    d = np.abs(X1[:, None] - X2[None, :])
    return sigma_f ** 2 * np.exp(-2 * np.sin(np.pi * d / period) ** 2 / l ** 2)


KERNELS = {"RBF": k_rbf, "Linear": k_linear, "Periodic": k_periodic}


def gp_posterior(kernel_fn, X_train, y_train, X_test, l, sigma_f, sigma_n, period):
    K = kernel_fn(X_train, X_train, l, sigma_f, period) + sigma_n ** 2 * np.eye(len(X_train))
    K_s = kernel_fn(X_test, X_train, l, sigma_f, period)
    K_ss = kernel_fn(X_test, X_test, l, sigma_f, period)

    alpha = np.linalg.solve(K, y_train)
    mu = K_s @ alpha
    cov = K_ss - K_s @ np.linalg.solve(K, K_s.T)
    return mu, cov, alpha, K


def log_marginal_likelihood(K, alpha, y_train):
    n = len(y_train)
    _, logdet = np.linalg.slogdet(K)
    return -0.5 * y_train @ alpha - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)

X_train = np.array([0.0, 1.0, 2.5, 4.0])
y_train = np.array([1.0, 2.0, 0.5, 1.5])
X_test = np.linspace(-1, 5, 200)
X_MARK = 0.5


def main():
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_axes([0.08, 0.42, 0.58, 0.52])
    ax_radio = fig.add_axes([0.72, 0.66, 0.24, 0.28])
    ax_l = fig.add_axes([0.12, 0.30, 0.48, 0.03])
    ax_sf = fig.add_axes([0.12, 0.24, 0.48, 0.03])
    ax_sn = fig.add_axes([0.12, 0.18, 0.48, 0.03])
    ax_period = fig.add_axes([0.12, 0.12, 0.48, 0.03])

    ax_radio.set_title("ядро k(x,x')", fontsize=10)
    radio = RadioButtons(ax_radio, list(KERNELS.keys()))

    s_l = Slider(ax_l, "l (длина корр.)", 0.1, 3.0, valinit=1.0)
    s_sf = Slider(ax_sf, "σ_f (масштаб)", 0.1, 3.0, valinit=1.0)
    s_sn = Slider(ax_sn, "σ_n (шум)", 0.001, 1.0, valinit=0.10)
    s_period = Slider(ax_period, "period (для Periodic)", 0.2, 4.0, valinit=1.0)

    info_text = fig.text(0.72, 0.58, "", family="monospace", fontsize=8, va="top")

    mean_line, = ax.plot([], [], color="C0", label="апостериорное среднее μ*(x)")
    band = [None]  # ссылка на PolyCollection доверительной полосы (удаляем/пересоздаём)
    ax.scatter(X_train, y_train, color="red", zorder=5, label="обучающие точки")
    ax.axvline(X_MARK, color="gray", linestyle=":", linewidth=1)
    ax.set_xlim(X_test.min(), X_test.max())
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(loc="upper left", fontsize=8)

    def update(_event=None):
        kernel_fn = KERNELS[radio.value_selected]
        l, sf, sn, period = s_l.val, s_sf.val, s_sn.val, s_period.val

        mu, cov, alpha, K = gp_posterior(kernel_fn, X_train, y_train, X_test,
                                          l, sf, sn, period)
        std = np.sqrt(np.clip(np.diag(cov), 0, None))
        lml = log_marginal_likelihood(K, alpha, y_train)

        mu_mark, cov_mark, _, _ = gp_posterior(kernel_fn, X_train, y_train,
                                                np.array([X_MARK]), l, sf, sn, period)
        std_mark = float(np.sqrt(max(cov_mark[0, 0], 0)))

        mean_line.set_data(X_test, mu)
        if band[0] is not None:
            band[0].remove()
        band[0] = ax.fill_between(X_test, mu - 1.96 * std, mu + 1.96 * std,
                                   color="C0", alpha=0.2)

        lo = min((mu - 1.96 * std).min(), y_train.min()) - 0.5
        hi = max((mu + 1.96 * std).max(), y_train.max()) + 0.5
        ax.set_ylim(lo, hi)

        info_text.set_text(
            f"ядро: {radio.value_selected}\n"
            f"l={l:.2f}  σ_f={sf:.2f}  σ_n={sn:.3f}  period={period:.2f}\n\n"
            f"log p(y|θ) = {lml:.3f}\n\n"
            f"alpha = (K+σ_n^2 I)^-1 y =\n{np.array2string(alpha, precision=3)}\n\n"
            f"K =\n{np.array2string(K, precision=2, suppress_small=True)}\n\n"
            f"в x*={X_MARK}:\n  mu* = {mu_mark[0]:.3f}\n  sigma* = {std_mark:.3f}"
        )
        fig.canvas.draw_idle()

    s_l.on_changed(update)
    s_sf.on_changed(update)
    s_sn.on_changed(update)
    s_period.on_changed(update)
    radio.on_clicked(update)

    update()

    plt.show()


if __name__ == "__main__":
    main()
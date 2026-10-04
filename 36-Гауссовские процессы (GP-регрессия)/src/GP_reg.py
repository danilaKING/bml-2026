import numpy as np
import matplotlib.pyplot as plt

KERNEL_NAME = "Periodic"          # "RBF" | "Linear" | "Periodic"
THETA0 = [1.0, 1.0, 1.0, 1.0]
BOUNDS = [(0.05, 1.0), (0.3, 2.0), (1e-3, 1.0), (1e-3, 10.0)]
N_RESTARTS = 10
N_TRAIN = 50
X_train = np.linspace(0.0, 4.8, N_TRAIN)
rng = np.random.default_rng(1408)
y_train = np.sin(2 * np.pi * X_train) + 0.08 * rng.standard_normal(N_TRAIN)

N_TEST = 1000
X_test = np.linspace(-1.0, 6.0, N_TEST)
X_MARK = 2.3

def kernel_rbf(X1, X2, l, sigma_f, period):
    sqdist = (X1[:, None] - X2[None, :]) ** 2
    exp_term = np.exp(-0.5 * sqdist / l ** 2)
    K = sigma_f ** 2 * exp_term
    dK_dl = K * sqdist / l ** 3
    dK_dsf = 2.0 * sigma_f * exp_term
    dK_dp = np.zeros_like(K)
    return K, [dK_dl, dK_dsf, dK_dp]


def kernel_linear(X1, X2, l, sigma_f, period):
    base = X1[:, None] * X2[None, :] + 1.0
    K = sigma_f ** 2 * base
    dK_dl = np.zeros_like(K)
    dK_dsf = 2.0 * sigma_f * base
    dK_dp = np.zeros_like(K)
    return K, [dK_dl, dK_dsf, dK_dp]


def kernel_periodic(X1, X2, l, sigma_f, period):
    d = np.abs(X1[:, None] - X2[None, :])
    arg = np.pi * d / period
    s2 = np.sin(arg) ** 2
    exp_term = np.exp(-2.0 * s2 / l ** 2)
    K = sigma_f ** 2 * exp_term
    dK_dl = K * 4.0 * s2 / l ** 3
    dK_dsf = 2.0 * sigma_f * exp_term
    dK_dp = K * (4.0 * np.pi * d / (l ** 2 * period ** 2)) * np.sin(arg) * np.cos(arg)
    return K, [dK_dl, dK_dsf, dK_dp]


KERNELS = {"RBF": kernel_rbf, "Linear": kernel_linear, "Periodic": kernel_periodic}

def log_marginal_likelihood_and_grad(kernel_fn, X_train, y_train,
                                      l, sigma_f, sigma_n, period):
    n = len(X_train)
    K, dK_list = kernel_fn(X_train, X_train, l, sigma_f, period)
    K_sigma = K + sigma_n ** 2 * np.eye(n)

    try:
        L = np.linalg.cholesky(K_sigma)
    except np.linalg.LinAlgError:
        return -np.inf, np.zeros(4)

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    logdet = 2.0 * np.sum(np.log(np.diag(L)))

    logp = -0.5 * y_train @ alpha - 0.5 * logdet - 0.5 * n * np.log(2.0 * np.pi)

    K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))
    common = 0.5 * (np.outer(alpha, alpha) - K_inv)

    grads = np.zeros(4)
    for i, dK in enumerate(dK_list):
        grads[i] = np.sum(common * dK)
    grads[2] = sigma_n * (alpha @ alpha - np.trace(K_inv))

    return logp, grads

def gp_posterior(kernel_fn, X_train, y_train, X_test,
                  l, sigma_f, sigma_n, period):
    n = len(X_train)
    K, _ = kernel_fn(X_train, X_train, l, sigma_f, period)
    K_sigma = K + sigma_n ** 2 * np.eye(n)
    K_s, _ = kernel_fn(X_test, X_train, l, sigma_f, period)
    K_ss, _ = kernel_fn(X_test, X_test, l, sigma_f, period)

    L = np.linalg.cholesky(K_sigma)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mu = K_s @ alpha
    v = np.linalg.solve(L, K_s.T)
    cov = K_ss - v.T @ v
    return mu, cov

def optimize_hyperparams(kernel_fn, X_train, y_train, theta0, bounds,
                          n_iter=500, tol=1e-8):
    theta = np.array(theta0, dtype=float)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    logp, grad = log_marginal_likelihood_and_grad(
        kernel_fn, X_train, y_train, *theta)

    step = 1.0
    for _ in range(n_iter):
        if not np.isfinite(logp) or not np.all(np.isfinite(grad)):
            break
        if np.linalg.norm(grad) < tol:
            break

        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            direction = grad / grad_norm
        else:
            break

        improved = False
        trial_step = step
        for _ in range(30):
            trial = np.clip(theta + trial_step * direction, lo, hi)
            trial_logp, trial_grad = log_marginal_likelihood_and_grad(
                kernel_fn, X_train, y_train, *trial)
            if np.isfinite(trial_logp) and trial_logp > logp:
                old_logp = logp
                theta, logp, grad = trial, trial_logp, trial_grad
                step = trial_step * 1.5
                improved = True
                break
            trial_step *= 0.5

        if not improved:
            break
        if abs(logp - old_logp) < tol * max(1.0, abs(logp)):
            break

    return theta


def optimize_with_restarts(kernel_fn, X_train, y_train, theta0, bounds,
                            n_restarts=20, seed=0):
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    log_lo, log_hi = np.log(lo), np.log(hi)

    rng_r = np.random.default_rng(seed)

    best_theta = None
    best_logp = -np.inf
    results = []

    inits = [np.array(theta0, dtype=float)]
    for _ in range(n_restarts - 1):
        inits.append(np.exp(rng_r.uniform(log_lo, log_hi)))

    for k, init in enumerate(inits):
        theta = optimize_hyperparams(kernel_fn, X_train, y_train, init, bounds)
        logp, _ = log_marginal_likelihood_and_grad(
            kernel_fn, X_train, y_train, *theta)
        results.append((k, theta.copy(), logp))
        if logp > best_logp:
            best_logp = logp
            best_theta = theta

    return best_theta, best_logp


def draw_gp(kernel_fn, kernel_name, theta,
             y_mean, y_std, title_suffix=""):
    l, sf, sn, period = theta

    y_norm = (y_train - y_mean) / y_std

    logp, _ = log_marginal_likelihood_and_grad(
        kernel_fn, X_train, y_norm, l, sf, sn, period)

    mu_n, cov_n = gp_posterior(kernel_fn, X_train, y_norm, X_test,
                                l, sf, sn, period)
    std_n = np.sqrt(np.clip(np.diag(cov_n), 0, None))

    mu = mu_n * y_std + y_mean
    std = std_n * y_std

    mu_m_n, cov_m_n = gp_posterior(kernel_fn, X_train, y_norm,
                                    np.array([X_MARK]), l, sf, sn, period)
    mu_m = mu_m_n[0] * y_std + y_mean
    s_m = float(np.sqrt(max(cov_m_n[0, 0], 0.0))) * y_std

    print(f"\n--- {title_suffix} ---")
    print(f"θ = (l, σ_f, σ_n, period) = "
          f"({l:.4f}, {sf:.4f}, {sn:.4f}, {period:.4f})")
    print(f"log p(y|θ) = {logp:.4f}")
    print(f"в x* = {X_MARK}:  mu* = {mu_m:.4f},  sigma* = {s_m:.4f}")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(X_test,
                     mu - 1.96 * std,
                     mu + 1.96 * std,
                     color="C0", alpha=0.2, label="95% интервал")
    ax.plot(X_test, mu, color="C0", lw=1.8, label="μ*(x)")
    ax.scatter(X_train, y_train, color="red", s=35, zorder=5,
               label="обучающие точки")

    ax.axvline(X_MARK, color="gray", linestyle=":", lw=0.9, alpha=0.8)
    ax.errorbar([X_MARK], [mu_m], yerr=[1.96 * s_m],
                 fmt="o", color="black", capsize=4, zorder=6)

    lo = min((mu - 1.96 * std).min(), y_train.min())
    hi = max((mu + 1.96 * std).max(), y_train.max())
    span = hi - lo
    ax.set_ylim(lo - 0.05 * span, hi + 0.25 * span)

    ax.annotate(f"μ*={mu_m:.2f}\nσ*={s_m:.2f}",
                 xy=(X_MARK, mu_m),
                 xytext=(0.78, 0.93),
                 textcoords="axes fraction",
                 fontsize=10, ha="center", va="top",
                 bbox=dict(boxstyle="round,pad=0.3",
                           fc="white", ec="gray", alpha=0.9),
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.9,
                                  shrinkA=2, shrinkB=4))

    ax.set_xlim(X_test.min(), X_test.max())
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"GP-регрессия, ядро {kernel_name}, n = {len(X_train)}{title_suffix}\n"
                  f"θ = (l={l:.3f}, σ_f={sf:.3f}, σ_n={sn:.4f}, p={period:.3f}),  "
                  f"log p(y|θ) = {logp:.2f}")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    kernel_fn = KERNELS[KERNEL_NAME]

    y_mean = y_train.mean()
    y_std = y_train.std()
    print(f"Ядро: {KERNEL_NAME}")
    print(f"n_train = {len(X_train)}")
    print(f"y: mean = {y_mean:.4f}, std = {y_std:.4f}")

    theta0 = np.array(THETA0, dtype=float)

    draw_gp(kernel_fn, KERNEL_NAME, theta0, y_mean, y_std,
            title_suffix="   [начальное θ]")

    theta_opt, _ = optimize_with_restarts(
        kernel_fn, X_train, (y_train - y_mean) / y_std,
        theta0, BOUNDS, n_restarts=N_RESTARTS)

    draw_gp(kernel_fn, KERNEL_NAME, theta_opt, y_mean, y_std,
            title_suffix="   [обученное θ]")

    plt.show()
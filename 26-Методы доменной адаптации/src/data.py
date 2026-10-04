import numpy as np
from sklearn.datasets import make_moons

MOONS_CENTER = np.array([0.5, 0.25])


def rotate(X, angle_deg, center=MOONS_CENTER):
    """Поворот точек X на angle_deg градусов против часовой стрелки вокруг center."""
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)],
                  [np.sin(a),  np.cos(a)]])
    return (X - center) @ R.T + center


def make_domains(angle=35, n_source=600, n_target=1200, noise=0.1, seed=0):
    Xs, ys = make_moons(n_samples=n_source, noise=noise, random_state=seed)
    Xt, yt = make_moons(n_samples=n_target, noise=noise, random_state=seed + 1000)
    Xt = rotate(Xt, angle)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xt))
    half = len(Xt) // 2
    tr, te = idx[:half], idx[half:]

    return {
        "Xs": Xs.astype(np.float32), "ys": ys,
        "Xt_train": Xt[tr].astype(np.float32),           # без меток!
        "Xt_test": Xt[te].astype(np.float32), "yt_test": yt[te],
        "yt_train_oracle": yt[tr],                      # только для oracle
    }

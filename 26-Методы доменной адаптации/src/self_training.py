import numpy as np

from models import fit_classifier, predict_proba


def fit_self_training(Xs, ys, Xt, seed=0, rounds=5, threshold=0.9, n_iter=2000, verbose=False):
    model = fit_classifier(Xs, ys, seed=seed, n_iter=n_iter)
    for r in range(rounds):
        proba = predict_proba(model, Xt)
        conf, pseudo = proba.max(axis=1), proba.argmax(axis=1)
        mask = conf > threshold
        if verbose:
            print(f"раунд {r + 1}: уверенных target-точек {mask.sum()}/{len(Xt)}, "
                  f"доли классов {np.bincount(pseudo[mask], minlength=2)}")
        X_aug = np.concatenate([Xs, Xt[mask]])
        y_aug = np.concatenate([ys, pseudo[mask]])
        model = fit_classifier(X_aug, y_aug, seed=seed, n_iter=n_iter)
    return model

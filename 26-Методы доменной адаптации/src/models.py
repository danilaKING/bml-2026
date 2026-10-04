import numpy as np
import torch
import torch.nn as nn

HIDDEN = 32


def make_feature_extractor():
    return nn.Sequential(
        nn.Linear(2, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
    )


def make_label_classifier():
    return nn.Linear(HIDDEN, 2)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.G = make_feature_extractor()
        self.C = make_label_classifier()

    def forward(self, x):
        return self.C(self.G(x))


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_classifier(X, y, seed=0, n_iter=2000, batch=64, lr=1e-3):
    set_seed(seed)
    model = Classifier()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(n_iter):
        idx = torch.randint(0, len(X), (batch,))
        loss = loss_fn(model(X[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def predict_proba(model, X):
    X = torch.as_tensor(X, dtype=torch.float32)
    return torch.softmax(model(X), dim=1).numpy()


def predict(model, X):
    return predict_proba(model, X).argmax(axis=1)


@torch.no_grad()
def features(model, X):
    return model.G(torch.as_tensor(X, dtype=torch.float32)).numpy()

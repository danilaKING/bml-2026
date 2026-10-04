import numpy as np
import torch
import torch.nn as nn

from models import Classifier, HIDDEN, set_seed


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Меняем знак градиента; для lam градиент не нужен -> None
        return -ctx.lam * grad_output, None


def grad_reverse(x, lam):
    return GradReverse.apply(x, lam)


class DANN(Classifier):

    def __init__(self):
        super().__init__()
        self.D = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def domain_logits(self, x, lam):
        return self.D(grad_reverse(self.G(x), lam)).squeeze(1)


def lambda_schedule(p, gamma=10.0):
    return 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0


def fit_dann(Xs, ys, Xt, seed=0, n_iter=3000, batch=64, lr=1e-3, lam_max=1.0, log_every=0):
    set_seed(seed)
    model = DANN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()

    Xs = torch.as_tensor(Xs, dtype=torch.float32)
    ys = torch.as_tensor(ys, dtype=torch.long)
    Xt = torch.as_tensor(Xt, dtype=torch.float32)
    dom = torch.cat([torch.zeros(batch), torch.ones(batch)])   # 0 — source, 1 — target

    for it in range(n_iter):
        lam = lam_max * lambda_schedule(it / n_iter)
        i_s = torch.randint(0, len(Xs), (batch,))
        xs, yb = Xs[i_s], ys[i_s]
        xt = Xt[torch.randint(0, len(Xt), (batch,))]

        loss_cls = ce(model(xs), yb)
        loss_dom = bce(model.domain_logits(torch.cat([xs, xt]), lam), dom)

        loss = loss_cls + loss_dom
        opt.zero_grad()
        loss.backward()
        opt.step()

        if log_every and it % log_every == 0:
            with torch.no_grad():
                d_acc = ((model.domain_logits(torch.cat([xs, xt]), 0.0) > 0).float() == dom).float().mean()
            print(f"it={it:5d}  lambda={lam:.3f}  L_cls={loss_cls.item():.3f}  "
                  f"L_dom={loss_dom.item():.3f}  точность D={d_acc.item():.2f}")
    return model

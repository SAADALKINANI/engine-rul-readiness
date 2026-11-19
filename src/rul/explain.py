"""Integrated Gradients attribution for sequence models.

Answers the maintainer's question "*why* does the model think engine 31 is close to failure?"
by attributing the prediction to (time step, sensor) cells. Implemented from scratch
(Sundararajan et al., 2017) so it works for any differentiable model without extra deps.

Baseline: the engine's own *first* window would be ideal but is not always available,
so we use the training-set "healthy" mean (windows with RUL == cap), i.e. attributions
explain the deviation from a healthy engine.
"""
from __future__ import annotations

import numpy as np
import torch


def integrated_gradients(model: torch.nn.Module, x: torch.Tensor, baseline: torch.Tensor,
                         steps: int = 64) -> torch.Tensor:
    model.eval()
    alphas = torch.linspace(0, 1, steps).view(-1, 1, 1, 1)
    path = baseline.unsqueeze(0) + alphas * (x - baseline).unsqueeze(0)   # (S, B, T, F)
    s, b, t, f = path.shape
    path = path.reshape(s * b, t, f).requires_grad_(True)
    out = model(path).sum()
    grads, = torch.autograd.grad(out, path)
    avg = grads.reshape(s, b, t, f).mean(0)
    return (x - baseline) * avg           # (B, T, F)


def sensor_importance(attr: torch.Tensor | np.ndarray, features: list[str]) -> dict[str, float]:
    a = np.abs(np.asarray(attr)).sum(axis=(0, 1))
    a = a / a.sum()
    return dict(sorted(zip(features, a.round(4).tolist()), key=lambda kv: -kv[1]))

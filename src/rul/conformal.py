"""Distribution-free prediction intervals via split conformal prediction.

Why conformal? A maintenance planner needs a *guarantee*, not a hopeful error bar:
"this engine has at least L cycles left with 90% confidence". Split conformal prediction
gives finite-sample, model-agnostic marginal coverage >= 1 - alpha, provided calibration
and test points are exchangeable.

Two variants are implemented:

* ``SplitConformal``  - symmetric absolute-residual intervals.
* ``MondrianConformal`` - conditional (group-wise) calibration by predicted-RUL band.
  Errors are much larger for healthy engines (high RUL) than for engines near failure;
  a single global quantile would over-cover near failure and under-cover far from it.
  Mondrian calibration gives coverage per band, which is what matters operationally:
  intervals near failure become tight.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q, method="higher"))


@dataclass
class SplitConformal:
    alpha: float = 0.1
    q_: float | None = None

    def fit(self, y_cal: np.ndarray, pred_cal: np.ndarray) -> "SplitConformal":
        self.q_ = conformal_quantile(np.abs(y_cal - pred_cal), self.alpha)
        return self

    def predict(self, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.clip(pred - self.q_, 0, None), pred + self.q_


@dataclass
class MondrianConformal:
    alpha: float = 0.1
    bands: tuple[float, ...] = (0, 40, 80, np.inf)
    q_: dict[int, float] = field(default_factory=dict)

    def _band(self, pred: np.ndarray) -> np.ndarray:
        return np.digitize(pred, self.bands[1:-1])

    def fit(self, y_cal: np.ndarray, pred_cal: np.ndarray) -> "MondrianConformal":
        b = self._band(pred_cal)
        fallback = conformal_quantile(np.abs(y_cal - pred_cal), self.alpha)
        for k in range(len(self.bands) - 1):
            s = np.abs(y_cal[b == k] - pred_cal[b == k])
            self.q_[k] = conformal_quantile(s, self.alpha) if len(s) >= 20 else fallback
        return self

    def predict(self, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.array([self.q_[k] for k in self._band(pred)])
        return np.clip(pred - q, 0, None), pred + q


def coverage_report(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    inside = (y >= lo) & (y <= hi)
    return {"coverage": float(inside.mean()), "mean_width": float((hi - lo).mean()),
            "lower_bound_violations": float((y < lo).mean())}

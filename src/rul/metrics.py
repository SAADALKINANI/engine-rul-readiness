from __future__ import annotations

import numpy as np


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def nasa_score(y, p) -> float:
    """PHM08 asymmetric scoring function (Saxena et al. 2008).

    Late predictions (predicting more life than remains) are penalised more heavily than
    early ones, because a late prediction means an engine can fail in service.
    """
    d = np.asarray(p) - np.asarray(y)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))

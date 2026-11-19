"""Risk-aware, capacity-constrained maintenance scheduling.

Turns RUL predictions into a decision: *which engine goes into which hangar slot, on which day*.

For engine e maintained on day d the expected cost is

    C(e, d) = C_planned
            + C_failure * P(RUL_e < d * cycles_per_day)          # risk of in-service failure first
            + C_waste   * max(0, E[RUL_e] - d * cycles_per_day)  # useful life thrown away

P(RUL_e < t) is computed from the *empirical calibration residuals* (the same held-out data
used for conformal prediction), so the failure probabilities are calibrated rather than
assumed Gaussian. Engines can also be deferred past the planning horizon at a cost equal to
their failure risk by the horizon end.

The assignment is solved exactly as a mixed-integer linear program (scipy/HiGHS):
    min  sum_{e,d} C(e,d) x_{e,d}
    s.t. sum_d x_{e,d} = 1                 (each engine scheduled once, or deferred)
         sum_e x_{e,d} <= hangar_slots    (daily capacity)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass
class CostModel:
    planned: float = 250_000          # planned shop visit, USD
    failure: float = 2_500_000        # unplanned removal / AOG / mission loss, USD
    waste_per_cycle: float = 1_500    # value of useful life discarded, USD per cycle
    cycles_per_day: float = 2.0
    hangar_slots: int = 3
    horizon_days: int = 30


def failure_prob(pred: np.ndarray, t_cycles: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """P(true RUL < t) where true = pred + r and r ~ empirical calibration residuals."""
    r = np.sort(residuals)
    need = t_cycles[None, :] - pred[:, None]          # true < t  <=>  r < t - pred
    return np.searchsorted(r, need, side="left") / len(r)


def schedule(fleet: pd.DataFrame, residuals: np.ndarray, cm: CostModel = CostModel()) -> pd.DataFrame:
    """fleet: columns [unit, pred_rul]. Returns one row per engine with its day (or 'defer')."""
    pred = fleet["pred_rul"].to_numpy(float)
    n, H = len(pred), cm.horizon_days
    days = np.arange(H)
    t = days * cm.cycles_per_day
    p_fail = failure_prob(pred, t, residuals)                                        # (n, H)
    cost = cm.planned + cm.failure * p_fail + cm.waste_per_cycle * np.maximum(0, pred[:, None] - t)
    p_defer = failure_prob(pred, np.array([H * cm.cycles_per_day]), residuals)[:, 0]
    defer_cost = cm.failure * p_defer                                                  # (n,)

    c = np.concatenate([cost.ravel(), defer_cost])                                   # n*H + n vars
    nv = n * H + n
    A_one = np.zeros((n, nv)); A_cap = np.zeros((H, nv))
    for e in range(n):
        A_one[e, e * H:(e + 1) * H] = 1
        A_one[e, n * H + e] = 1
    for d in range(H):
        A_cap[d, d:n * H:H] = 1
    res = milp(c, integrality=np.ones(nv), bounds=Bounds(0, 1),
               constraints=[LinearConstraint(A_one, 1, 1), LinearConstraint(A_cap, 0, cm.hangar_slots)],
               options={"time_limit": 60})
    if res.x is None:
        raise RuntimeError(f"MILP failed: {res.message}")
    x = res.x.round()
    out = fleet.copy()
    day = np.full(n, -1)
    for e in range(n):
        hit = np.where(x[e * H:(e + 1) * H] > 0.5)[0]
        if len(hit):
            day[e] = hit[0]
    out["maint_day"] = np.where(day >= 0, day, np.nan)
    out["p_fail_before_maint"] = [p_fail[e, day[e]] if day[e] >= 0 else p_defer[e] for e in range(n)]
    out["expected_cost"] = [cost[e, day[e]] if day[e] >= 0 else defer_cost[e] for e in range(n)]
    out["action"] = np.where(day >= 0, "schedule", "defer (re-assess next cycle)")
    return out.sort_values(["maint_day", "pred_rul"], na_position="last").reset_index(drop=True)


def threshold_policy(fleet: pd.DataFrame, cm: CostModel, threshold: float = 30) -> pd.DataFrame:
    """Baseline: maintain any engine whose *point* prediction is below `threshold` cycles,
    most-urgent-first into the next free slot (what a simple dashboard alert rule does)."""
    out = fleet.copy()
    slots = {d: cm.hangar_slots for d in range(cm.horizon_days)}
    day = np.full(len(out), np.nan)
    for e in np.argsort(out["pred_rul"].to_numpy()):
        if out["pred_rul"].iloc[e] < threshold:
            d = next((d for d in range(cm.horizon_days) if slots[d] > 0), None)
            if d is not None:
                slots[d] -= 1
                day[e] = d
    out["maint_day"] = day
    return out


def realized_cost(plan: pd.DataFrame, true_rul: np.ndarray, cm: CostModel) -> dict:
    """Score a plan against what actually happened (ground-truth RUL, known only in hindsight).

    An engine fails in service if its true RUL runs out before its maintenance day
    (or before the horizon end, if deferred)."""
    day = plan["maint_day"].to_numpy(float)
    deadline = np.where(np.isnan(day), cm.horizon_days, day) * cm.cycles_per_day
    failed = true_rul < deadline
    maintained = ~np.isnan(day) & ~failed
    cost = (failed * cm.failure
            + maintained * (cm.planned + cm.waste_per_cycle * np.maximum(0, true_rul - deadline)))
    return {"total_cost": float(cost.sum()), "in_service_failures": int(failed.sum()),
            "shop_visits": int(maintained.sum()),
            "wasted_cycles": float((maintained * np.maximum(0, true_rul - deadline)).sum())}

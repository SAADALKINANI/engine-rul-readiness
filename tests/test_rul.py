import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rul.conformal import MondrianConformal, SplitConformal, coverage_report  # noqa: E402
from rul.data import SENSOR_COLS, add_train_rul, informative_sensors, make_windows  # noqa: E402
from rul.explain import integrated_gradients  # noqa: E402
from rul.metrics import nasa_score  # noqa: E402
from rul.models import build_model, mc_dropout_predict  # noqa: E402
from rul.scheduler import CostModel, realized_cost, schedule, threshold_policy  # noqa: E402


def fake_fleet(n_units=5, rng=np.random.default_rng(0)):
    rows = []
    for u in range(1, n_units + 1):
        life = rng.integers(40, 90)
        for c in range(1, life + 1):
            rows.append({"unit": u, "cycle": c, **{s: rng.normal() + c * 0.01 for s in SENSOR_COLS}})
    return pd.DataFrame(rows)


def test_rul_target_is_capped_and_decreasing():
    df = add_train_rul(fake_fleet(), cap=50)
    assert df["rul"].max() == 50
    g = df[df.unit == 1]["rul"].to_numpy()
    assert (np.diff(g) <= 0).all() and g[-1] == 0


def test_windows_shapes_and_padding():
    df = add_train_rul(fake_fleet())
    feats = informative_sensors(df)
    X, y, u = make_windows(df, feats, window=30)
    assert X.shape[1:] == (30, len(feats)) and len(X) == len(y) == len(u)
    short = df[df.cycle <= 10]
    Xs, ys, _ = make_windows(short, feats, window=30, last_only=True)
    assert Xs.shape == (5, 30, len(feats))           # front-padded, one per engine


@pytest.mark.parametrize("name", ["lstm", "tcn", "transformer"])
def test_models_forward(name):
    m = build_model(name, 14, 30)
    out = m(torch.randn(4, 30, 14))
    assert out.shape == (4,)


def test_mc_dropout_returns_uncertainty():
    m = build_model("transformer", 14, 30)
    mu, sd = mc_dropout_predict(m, torch.randn(3, 30, 14), passes=5)
    assert mu.shape == sd.shape == (3,) and (sd >= 0).all()


def test_conformal_coverage_is_valid():
    rng = np.random.default_rng(1)
    y = rng.uniform(0, 125, 4000); p = y + rng.normal(0, 10, 4000)
    for cp in (SplitConformal(0.1), MondrianConformal(0.1)):
        cp.fit(y[:2000], p[:2000])
        lo, hi = cp.predict(p[2000:])
        assert coverage_report(y[2000:], lo, hi)["coverage"] >= 0.87


def test_nasa_score_penalises_late_more_than_early():
    assert nasa_score([50], [60]) > nasa_score([50], [40])


def test_integrated_gradients_completeness():
    """IG attributions sum (approximately) to f(x) - f(baseline)."""
    torch.manual_seed(0)
    m = build_model("lstm", 6, 10).eval()
    x, b = torch.randn(1, 10, 6), torch.zeros(10, 6)
    attr = integrated_gradients(m, x, b, steps=200)
    with torch.no_grad():
        diff = (m(x) - m(b[None])).item()
    assert abs(attr.sum().item() - diff) < 0.05 * max(1.0, abs(diff))


def test_scheduler_respects_capacity_and_beats_rule():
    rng = np.random.default_rng(0)
    fleet = pd.DataFrame({"unit": range(40), "pred_rul": rng.uniform(5, 120, 40)})
    resid = rng.normal(0, 12, 500)
    cm = CostModel(hangar_slots=2, horizon_days=20)
    plan = schedule(fleet, resid, cm)
    assert plan["maint_day"].value_counts().max() <= cm.hangar_slots
    truth = dict(zip(fleet.unit, fleet.pred_rul + rng.normal(0, 12, 40)))
    opt = realized_cost(plan, plan.unit.map(truth).to_numpy(), cm)
    naive = threshold_policy(fleet, cm, 30)
    base = realized_cost(naive, naive.unit.map(truth).to_numpy(), cm)
    assert opt["total_cost"] <= base["total_cost"] * 1.05

"""NASA C-MAPSS turbofan data loading, cleaning and windowing.

C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) contains run-to-failure
sensor trajectories for a fleet of turbofan engines. Each row is one engine cycle.

Key modelling choices (all documented in the README):
* Piecewise-linear RUL target capped at ``RUL_CAP`` cycles: early in life an engine is
  "healthy" and degradation is not yet observable, so predicting 300 vs 200 cycles is
  meaningless noise. This is the standard treatment in the RUL literature.
* Sensors that are constant in the training data are dropped (they carry no signal).
* Min-max scaling is fit on *training engines only* to avoid leakage.
* Splits are by **engine**, never by row, so no engine's history leaks across splits.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Human-readable sensor names (Saxena et al., 2008) - used in explanations and the app.
SENSOR_NAMES = {
    "s_2": "LPC outlet temp (T24)",
    "s_3": "HPC outlet temp (T30)",
    "s_4": "LPT outlet temp (T50)",
    "s_7": "HPC outlet pressure (P30)",
    "s_8": "Physical fan speed (Nf)",
    "s_9": "Physical core speed (Nc)",
    "s_11": "HPC static pressure (Ps30)",
    "s_12": "Fuel flow / Ps30 (phi)",
    "s_13": "Corrected fan speed (NRf)",
    "s_14": "Corrected core speed (NRc)",
    "s_15": "Bypass ratio (BPR)",
    "s_17": "Bleed enthalpy (htBleed)",
    "s_20": "HPT coolant bleed (W31)",
    "s_21": "LPT coolant bleed (W32)",
}

RUL_CAP = 125
DEFAULT_WINDOW = 30


def read_cmapss(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(ALL_COLS)]
    df.columns = ALL_COLS
    return df


def add_train_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["rul"] = np.minimum(max_cycle - df["cycle"], cap).astype(np.float32)
    return df


def add_test_rul(df: pd.DataFrame, rul_path: str | Path, cap: int = RUL_CAP) -> pd.DataFrame:
    """Test trajectories stop before failure; the RUL file gives true RUL at the last cycle."""
    final_rul = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0].to_numpy()
    df = df.copy()
    last = df.groupby("unit")["cycle"].transform("max")
    unit_final = dict(zip(sorted(df["unit"].unique()), final_rul))
    df["rul"] = np.minimum(df["unit"].map(unit_final) + (last - df["cycle"]), cap).astype(np.float32)
    df["true_final_rul"] = df["unit"].map(unit_final).astype(np.float32)  # uncapped, for reporting
    return df


@dataclass
class Scaler:
    cols: list[str]
    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def fit(cls, df: pd.DataFrame, cols: list[str]) -> "Scaler":
        return cls(cols, df[cols].min().to_numpy(), df[cols].max().to_numpy())

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rng = np.where(self.hi - self.lo == 0, 1.0, self.hi - self.lo)
        df[self.cols] = ((df[self.cols].to_numpy() - self.lo) / rng).astype(np.float32)
        return df


def informative_sensors(train: pd.DataFrame, tol: float = 5e-3) -> list[str]:
    std = train[SENSOR_COLS].std()
    return [c for c in SENSOR_COLS if std[c] > tol]


def make_windows(df: pd.DataFrame, features: list[str], window: int = DEFAULT_WINDOW,
                 last_only: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn per-cycle rows into (N, window, F) tensors.

    Engines shorter than ``window`` are front-padded with their first observation, which is a
    common, conservative choice (it does not invent degradation).
    Returns X, y (RUL at the window's last cycle) and the engine id of each window.
    """
    xs, ys, units = [], [], []
    for unit, g in df.groupby("unit", sort=True):
        arr = g[features].to_numpy(np.float32)
        rul = g["rul"].to_numpy(np.float32)
        if len(arr) < window:
            pad = np.repeat(arr[:1], window - len(arr), axis=0)
            arr = np.vstack([pad, arr])
            rul = np.concatenate([np.repeat(rul[:1], window - len(rul)), rul])
        if last_only:
            xs.append(arr[-window:]); ys.append(rul[-1]); units.append(unit)
            continue
        # sliding windows, stride 1
        idx = np.arange(window, len(arr) + 1)
        stacked = np.lib.stride_tricks.sliding_window_view(arr, (window, arr.shape[1]))[:, 0]
        xs.extend(stacked); ys.extend(rul[idx - 1]); units.extend([unit] * len(idx))
    return np.stack(xs).astype(np.float32), np.asarray(ys, np.float32), np.asarray(units)


@dataclass
class CMAPSSSplits:
    features: list[str]
    X_train: np.ndarray; y_train: np.ndarray; u_train: np.ndarray
    X_val: np.ndarray; y_val: np.ndarray                            # early stopping only
    X_cal: np.ndarray; y_cal: np.ndarray; u_cal: np.ndarray        # conformal calibration only
    X_test: np.ndarray; y_test: np.ndarray; u_test: np.ndarray      # last window per test engine
    test_frame: pd.DataFrame
    scaler: Scaler


def load_splits(data_dir: str | Path, subset: str = "FD001", window: int = DEFAULT_WINDOW,
                cal_frac: float = 0.2, val_frac: float = 0.1, seed: int = 7) -> CMAPSSSplits:
    data_dir = Path(data_dir)
    train = add_train_rul(read_cmapss(data_dir / f"train_{subset}.txt"))
    test = add_test_rul(read_cmapss(data_dir / f"test_{subset}.txt"), data_dir / f"RUL_{subset}.txt")

    features = informative_sensors(train)
    rng = np.random.default_rng(seed)
    units = train["unit"].unique()
    shuffled = rng.permutation(units)
    n_cal, n_val = max(1, int(len(units) * cal_frac)), max(1, int(len(units) * val_frac))
    cal_units, val_units = set(shuffled[:n_cal]), set(shuffled[n_cal:n_cal + n_val])
    tr = train[~train["unit"].isin(cal_units | val_units)]
    val = train[train["unit"].isin(val_units)]
    cal = train[train["unit"].isin(cal_units)]

    scaler = Scaler.fit(tr, features)
    tr, val, cal, test_s = (scaler.transform(d) for d in (tr, val, cal, test))

    Xtr, ytr, utr = make_windows(tr, features, window)
    Xv, yv, _ = make_windows(val, features, window)
    # Calibration set mimics the *test* situation: trajectories truncated at random points,
    # one window each. This makes conformal coverage valid for the deployment distribution.
    cal_trunc = _random_truncate(cal, rng, n_per_unit=10)
    Xc, yc, uc = make_windows_per_group(cal_trunc, features, window)
    Xte, yte, ute = make_windows(test_s, features, window, last_only=True)
    return CMAPSSSplits(features, Xtr, ytr, utr, Xv, yv, Xc, yc, uc, Xte, yte, ute, test, scaler)


def _random_truncate(df: pd.DataFrame, rng: np.random.Generator, n_per_unit: int) -> pd.DataFrame:
    parts = []
    for unit, g in df.groupby("unit"):
        n = len(g)
        cuts = rng.integers(low=min(n, 31), high=n + 1, size=n_per_unit)
        for k, cut in enumerate(cuts):
            part = g.iloc[:cut].copy()
            part["group"] = f"{unit}_{k}"
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def make_windows_per_group(df: pd.DataFrame, features: list[str], window: int):
    xs, ys, us = [], [], []
    for _, g in df.groupby("group", sort=False):
        arr = g[features].to_numpy(np.float32)
        if len(arr) < window:
            arr = np.vstack([np.repeat(arr[:1], window - len(arr), axis=0), arr])
        xs.append(arr[-window:]); ys.append(g["rul"].iloc[-1]); us.append(g["unit"].iloc[0])
    return np.stack(xs), np.asarray(ys, np.float32), np.asarray(us)

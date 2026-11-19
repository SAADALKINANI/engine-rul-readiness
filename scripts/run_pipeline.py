"""End-to-end pipeline: benchmark models -> conformal intervals -> explanations -> fleet schedule.

    python scripts/run_pipeline.py --subsets FD001 FD003 --seeds 0 1 2

Outputs (all committed so reviewers can read results without re-running):
    results/metrics.json, results/benchmark.md, results/figures/*.png,
    results/fleet_plan_<subset>.csv, artifacts/<subset>_best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import lightgbm as lgb  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from rul.conformal import MondrianConformal, SplitConformal, coverage_report  # noqa: E402
from rul.data import SENSOR_NAMES, load_splits  # noqa: E402
from rul.explain import integrated_gradients, sensor_importance  # noqa: E402
from rul.metrics import mae, nasa_score, rmse  # noqa: E402
from rul.models import build_model  # noqa: E402
from rul.scheduler import CostModel, realized_cost, schedule, threshold_policy  # noqa: E402
from rul.train import TrainConfig, fit, predict  # noqa: E402

FIG = ROOT / "results" / "figures"


def window_features(X: np.ndarray) -> np.ndarray:
    """Hand-crafted features for the gradient-boosting baseline: level, trend, volatility."""
    t = np.arange(X.shape[1]) - (X.shape[1] - 1) / 2
    slope = (X * t[None, :, None]).sum(1) / (t ** 2).sum()
    return np.concatenate([X[:, -1], X.mean(1), X.std(1), slope, X[:, -5:].mean(1)], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["FD001", "FD003"])
    ap.add_argument("--models", nargs="+", default=["lstm", "tcn", "transformer"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    FIG.mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("engine-rul")

    all_results: dict = {}
    for subset in args.subsets:
        sp = load_splits(ROOT / "data" / "raw", subset)
        nf, win = len(sp.features), sp.X_train.shape[1]
        print(f"\n=== {subset}: train windows={len(sp.X_train)} val={len(sp.X_val)} "
              f"cal={len(sp.X_cal)} test engines={len(sp.X_test)} features={nf}")
        res: dict = {"models": {}}

        # --- Gradient-boosting baseline on engineered window features -------------------------
        gbm = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
        gbm.fit(window_features(sp.X_train), sp.y_train,
                eval_set=[(window_features(sp.X_val), sp.y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        p = gbm.predict(window_features(sp.X_test))
        gbm_val = rmse(sp.y_val, gbm.predict(window_features(sp.X_val)))
        res["models"]["lightgbm"] = {"rmse": rmse(sp.y_test, p), "mae": mae(sp.y_test, p),
                                     "nasa_score": nasa_score(sp.y_test, p), "seeds": 1, "val_rmse": gbm_val}
        print(f"lightgbm      RMSE {res['models']['lightgbm']['rmse']:.2f}")

        # --- Deep sequence models, multiple seeds --------------------------------------------
        best_overall = (np.inf, None, None)
        for name in args.models:
            per_seed = []
            for seed in args.seeds:
                torch.manual_seed(seed)
                model = build_model(name, nf, win)
                ck = ROOT / "artifacts" / "runs" / f"{subset}_{name}_s{seed}.pt"
                if ck.exists():                      # resume: finished runs are cached (restart-safe)
                    c = torch.load(ck, weights_only=False)
                    model.load_state_dict(c["state"]); m, info = c["metrics"], c["info"]
                    per_seed.append(m)
                    print(f"{name:12s} seed {seed}  RMSE {m['rmse']:.2f}  (cached)")
                    if info["best_val_rmse"] < best_overall[0]:
                        best_overall = (info["best_val_rmse"], name, model)
                    continue
                with mlflow.start_run(run_name=f"{subset}-{name}-s{seed}"):
                    mlflow.log_params({"subset": subset, "model": name, "seed": seed, "window": win,
                                       "n_params": sum(p.numel() for p in model.parameters())})
                    model, info = fit(model, sp.X_train, sp.y_train, sp.X_val, sp.y_val,
                                      TrainConfig(epochs=args.epochs, seed=seed, lr=2e-3),
                                      log=lambda ep, tl, vr: mlflow.log_metrics(
                                          {"train_loss": tl, "val_rmse": vr}, step=ep))
                    p = predict(model, sp.X_test)
                    m = {"rmse": rmse(sp.y_test, p), "mae": mae(sp.y_test, p), "nasa_score": nasa_score(sp.y_test, p)}
                    mlflow.log_metrics({f"test_{k}": v for k, v in m.items()})
                per_seed.append(m)
                ck.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state": model.state_dict(), "metrics": m,
                            "info": {k: v for k, v in info.items() if k != "history"}}, ck)
                print(f"{name:12s} seed {seed}  RMSE {m['rmse']:.2f}  score {m['nasa_score']:.0f}  "
                      f"({info['epochs_run']} ep, {info['train_seconds']:.0f}s)")
                if info["best_val_rmse"] < best_overall[0]:      # select on *validation*, never test
                    best_overall = (info["best_val_rmse"], name, model)
            agg = {k: float(np.mean([s[k] for s in per_seed])) for k in per_seed[0]}
            agg.update({f"{k}_std": float(np.std([s[k] for s in per_seed])) for k in per_seed[0]})
            agg["seeds"] = len(per_seed)
            agg["n_params"] = int(sum(p.numel() for p in build_model(name, nf, win).parameters()))
            res["models"][name] = agg

        deep_val, best_name, best = best_overall
        # ---- model selection on VALIDATION among: best deep model, LightGBM, and their average --------------
        pg = lambda X: gbm.predict(window_features(X))  # noqa: E731
        cands = {best_name: (lambda X: predict(best, X)), "lightgbm": pg,
                 f"ensemble({best_name}+lightgbm)": (lambda X: 0.5 * predict(best, X) + 0.5 * pg(X))}
        val_scores = {k: rmse(sp.y_val, f(sp.X_val)) for k, f in cands.items()}
        sel = min(val_scores, key=val_scores.get)
        pred_fn = cands[sel]
        pe = cands[f"ensemble({best_name}+lightgbm)"](sp.X_test)
        res["models"]["ensemble"] = {"rmse": rmse(sp.y_test, pe), "mae": mae(sp.y_test, pe),
                                     "nasa_score": nasa_score(sp.y_test, pe), "seeds": 1, "members": f"{best_name}+lightgbm"}
        res["selection_val_rmse"] = val_scores
        res["selected_model"] = sel
        res["best_deep_model"] = best_name
        torch.save({"state_dict": best.state_dict(), "name": best_name, "selected": sel, "features": sp.features,
                    "window": win, "scaler_lo": sp.scaler.lo, "scaler_hi": sp.scaler.hi},
                   ROOT / "artifacts" / f"{subset}_best.pt")

        # --- Conformal prediction intervals ---------------------------------------------------
        p_cal, p_test = pred_fn(sp.X_cal), pred_fn(sp.X_test)
        res["selected_test"] = {"rmse": rmse(sp.y_test, p_test), "mae": mae(sp.y_test, p_test),
                                "nasa_score": nasa_score(sp.y_test, p_test)}
        sc = SplitConformal(args.alpha).fit(sp.y_cal, p_cal)
        mc = MondrianConformal(args.alpha).fit(sp.y_cal, p_cal)
        lo_s, hi_s = sc.predict(p_test)
        lo_m, hi_m = mc.predict(p_test)
        near = sp.y_test < 40
        res["conformal"] = {
            "target_coverage": 1 - args.alpha,
            "split": coverage_report(sp.y_test, lo_s, hi_s),
            "mondrian": coverage_report(sp.y_test, lo_m, hi_m),
            "split_near_failure_width": float((hi_s - lo_s)[near].mean()),
            "mondrian_near_failure_width": float((hi_m - lo_m)[near].mean()),
            "mondrian_band_halfwidths": {str(k): v for k, v in mc.q_.items()},
        }

        # --- Explanations (Integrated Gradients vs. healthy baseline) ------------------------
        healthy = torch.from_numpy(sp.X_train[sp.y_train >= 125].mean(0))
        idx = np.argsort(p_test)[:10]                     # the 10 most at-risk engines
        attr = integrated_gradients(best, torch.from_numpy(sp.X_test[idx]), healthy)
        imp_deep = sensor_importance(attr.detach().numpy(), sp.features)
        import shap  # tree SHAP for the gradient-boosting component
        sv = np.abs(shap.TreeExplainer(gbm).shap_values(window_features(sp.X_test[idx])))
        per = sv.reshape(len(idx), 5, len(sp.features)).sum((0, 1))     # 5 feature groups per sensor
        imp_gbm = dict(zip(sp.features, (per / per.sum()).round(4).tolist()))
        if sel == "lightgbm":
            imp = imp_gbm
        elif sel == best_name:
            imp = imp_deep
        else:
            imp = {k: round((imp_deep[k] + imp_gbm[k]) / 2, 4) for k in sp.features}
        imp = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
        res["explanation_method"] = {"lightgbm": "TreeSHAP", best_name: "Integrated Gradients"}.get(sel, "mean of IG (deep) and TreeSHAP (GBM)")
        res["sensor_importance_top_risk"] = {f"{k} - {SENSOR_NAMES.get(k, k)}": v for k, v in imp.items()}

        # --- Fleet maintenance plan -----------------------------------------------------------
        cm = CostModel()
        residuals = sp.y_cal - p_cal
        fleet = pd.DataFrame({"unit": sp.u_test, "pred_rul": p_test, "lower_90": lo_m, "upper_90": hi_m})
        truth = dict(zip(sp.u_test, sp.test_frame.groupby("unit")["true_final_rul"].first().loc[sp.u_test]))
        plan = schedule(fleet, residuals, cm)
        naive = threshold_policy(fleet, cm, threshold=30)
        r_opt = realized_cost(plan, plan["unit"].map(truth).to_numpy(), cm)
        r_naive = realized_cost(naive, naive["unit"].map(truth).to_numpy(), cm)
        res["fleet_plan"] = {"optimized": r_opt, "threshold_rule": r_naive,
                             "savings_usd": r_naive["total_cost"] - r_opt["total_cost"], "cost_model": cm.__dict__}
        plan.round(3).to_csv(ROOT / "results" / f"fleet_plan_{subset}.csv", index=False)
        fleet.round(3).to_csv(ROOT / "results" / f"fleet_predictions_{subset}.csv", index=False)
        np.save(ROOT / "artifacts" / f"{subset}_cal_residuals.npy", residuals)

        _plots(subset, sp, p_test, lo_m, hi_m, imp, res)
        all_results[subset] = res
        print(json.dumps({k: v for k, v in res.items() if k != "models"}, indent=1, default=float)[:1500])

    (ROOT / "results" / "metrics.json").write_text(json.dumps(all_results, indent=2, default=float))
    _benchmark_md(all_results)


def _plots(subset, sp, p, lo, hi, imp, res):
    order = np.argsort(sp.y_test)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(order))
    ax.fill_between(x, lo[order], hi[order], color="#8fb3e8", alpha=0.45, label="90% conformal interval")
    ax.plot(x, p[order], ".", color="#1f4e9c", label=f"Predicted RUL ({res['selected_model']})")
    ax.plot(x, sp.y_test[order], "-", color="#d1495b", lw=2, label="True RUL")
    ax.set_xlabel("Test engines (sorted by true RUL)"); ax.set_ylabel("Remaining useful life (cycles)")
    ax.set_title(f"{subset}: predictions with distribution-free 90% intervals "
                 f"(coverage {res['conformal']['mondrian']['coverage']:.0%})")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / f"{subset}_predictions.png", dpi=140); plt.close(fig)

    top = list(imp.items())[:8][::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([f"{k}  {SENSOR_NAMES.get(k, '')}" for k, _ in top], [v for _, v in top], color="#1f4e9c")
    ax.set_xlabel("Share of attribution (Integrated Gradients)")
    ax.set_title(f"{subset}: what drives risk for the 10 most at-risk engines")
    fig.tight_layout(); fig.savefig(FIG / f"{subset}_sensor_importance.png", dpi=140); plt.close(fig)


def _benchmark_md(all_results):
    lines = ["# Benchmark results (test set, last window of every test engine)\n",
             "Mean ± std over seeds. RMSE/MAE in cycles; NASA score is the asymmetric PHM08 score (lower is better).\n"]
    for subset, res in all_results.items():
        lines += [f"\n## {subset}\n", "| Model | RMSE | MAE | NASA score | Params |", "|---|---|---|---|---|"]
        for name, m in res["models"].items():
            sd = lambda k: f" ± {m[k + '_std']:.2f}" if f"{k}_std" in m else ""
            lines.append(f"| {name} | {m['rmse']:.2f}{sd('rmse')} | {m['mae']:.2f}{sd('mae')} | "
                         f"{m['nasa_score']:.0f} | {m.get('n_params', '-')} |")
        c = res["conformal"]
        st = res["selected_test"]
        lines += [f"\nSelected on validation RMSE: **{res['selected_model']}** "
                  f"(test RMSE {st['rmse']:.2f}, NASA score {st['nasa_score']:.0f})\n",
                  "| Interval method | Coverage (target 90%) | Mean width | Width near failure (RUL<40) |",
                  "|---|---|---|---|",
                  f"| Split conformal | {c['split']['coverage']:.1%} | {c['split']['mean_width']:.1f} | {c['split_near_failure_width']:.1f} |",
                  f"| Mondrian conformal | {c['mondrian']['coverage']:.1%} | {c['mondrian']['mean_width']:.1f} | {c['mondrian_near_failure_width']:.1f} |"]
        f = res["fleet_plan"]
        lines += ["\n| Maintenance policy | Realized cost | In-service failures | Shop visits | Wasted cycles |",
                  "|---|---|---|---|---|"]
        for k, lab in [("threshold_rule", "Alert rule (pred < 30)"), ("optimized", "Risk-aware MILP")]:
            r = f[k]
            lines.append(f"| {lab} | ${r['total_cost']:,.0f} | {r['in_service_failures']} | {r['shop_visits']} | {r['wasted_cycles']:.0f} |")
    (ROOT / "results" / "benchmark.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

"""Fleet Readiness dashboard.  Run:  streamlit run app.py   (after scripts/run_pipeline.py)"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
st.set_page_config(page_title="Fleet Readiness - Engine RUL", layout="wide")
metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
subset = st.sidebar.selectbox("Fleet (C-MAPSS subset)", list(metrics))
R = metrics[subset]
fleet = pd.read_csv(ROOT / "results" / f"fleet_predictions_{subset}.csv")
plan = pd.read_csv(ROOT / "results" / f"fleet_plan_{subset}.csv")

st.title("Fleet Readiness: Predictive Maintenance for Turbofan Engines")
st.caption(f"Model: **{R['selected_model']}** (selected on validation) · 90% conformal intervals · NASA C-MAPSS {subset}")
c1, c2, c3, c4 = st.columns(4)
f = R["fleet_plan"]
c1.metric("Engines < 30 cycles (predicted)", int((fleet.pred_rul < 30).sum()))
c2.metric("Interval coverage (target 90%)", f"{R['conformal']['mondrian']['coverage']:.0%}")
c3.metric("In-service failures (plan vs rule)", f"{f['optimized']['in_service_failures']} vs {f['threshold_rule']['in_service_failures']}")
c4.metric("Realized savings vs alert rule", f"${f['savings_usd'] / 1e6:,.1f}M")

st.subheader("Maintenance plan (MILP, capacity-constrained)")
st.dataframe(plan.head(40), width="stretch", height=360)

st.subheader("Engine drill-down")
unit = st.selectbox("Engine", fleet.sort_values("pred_rul").unit.tolist())
row = fleet[fleet.unit == unit].iloc[0]
fig = go.Figure()
fig.add_trace(go.Bar(x=["RUL"], y=[row.upper_90 - row.lower_90], base=[row.lower_90], name="90% interval", marker_color="#8fb3e8"))
fig.add_trace(go.Scatter(x=["RUL"], y=[row.pred_rul], mode="markers", marker=dict(size=16, color="#1f4e9c"), name="prediction"))
fig.update_layout(height=300, yaxis_title="cycles remaining")
st.plotly_chart(fig, width="stretch")
st.write(f"Guaranteed (90%) lower bound: **{row.lower_90:.0f} cycles** · point estimate {row.pred_rul:.0f}")

st.subheader("What drives risk (Integrated Gradients, top-10 at-risk engines)")
st.image(str(ROOT / "results" / "figures" / f"{subset}_sensor_importance.png"))
st.image(str(ROOT / "results" / "figures" / f"{subset}_predictions.png"))

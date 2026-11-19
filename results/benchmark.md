# Benchmark results (test set, last window of every test engine)

Mean ± std over seeds. RMSE/MAE in cycles; NASA score is the asymmetric PHM08 score (lower is better).


## FD001

| Model | RMSE | MAE | NASA score | Params |
|---|---|---|---|---|
| lightgbm | 12.20 | 9.43 | 216 | - |
| lstm | 13.41 ± 0.14 | 9.82 ± 0.04 | 309 | 55873 |
| tcn | 14.21 ± 0.69 | 10.12 ± 0.27 | 325 | 92289 |
| transformer | 14.65 ± 0.52 | 10.76 ± 0.24 | 412 | 105793 |
| ensemble | 12.37 | 9.02 | 234 | - |

Selected on validation RMSE: **ensemble(transformer+lightgbm)** (test RMSE 12.37, NASA score 234)

| Interval method | Coverage (target 90%) | Mean width | Width near failure (RUL<40) |
|---|---|---|---|
| Split conformal | 90.0% | 38.5 | 35.2 |
| Mondrian conformal | 90.0% | 37.7 | 19.1 |

| Maintenance policy | Realized cost | In-service failures | Shop visits | Wasted cycles |
|---|---|---|---|---|
| Alert rule (pred < 30) | $41,653,500 | 14 | 25 | 269 |
| Risk-aware MILP | $13,605,500 | 1 | 40 | 737 |

## FD003

| Model | RMSE | MAE | NASA score | Params |
|---|---|---|---|---|
| lightgbm | 12.63 | 9.07 | 303 | - |
| lstm | 14.02 ± 0.21 | 10.35 ± 0.29 | 396 | 56129 |
| tcn | 16.18 ± 0.34 | 11.06 ± 0.36 | 1088 | 92545 |
| transformer | 13.57 ± 0.60 | 9.33 ± 0.43 | 422 | 105857 |
| ensemble | 12.11 | 8.38 | 308 | - |

Selected on validation RMSE: **transformer** (test RMSE 12.84, NASA score 367)

| Interval method | Coverage (target 90%) | Mean width | Width near failure (RUL<40) |
|---|---|---|---|
| Split conformal | 92.0% | 44.2 | 40.4 |
| Mondrian conformal | 91.0% | 44.8 | 19.5 |

| Maintenance policy | Realized cost | In-service failures | Shop visits | Wasted cycles |
|---|---|---|---|---|
| Alert rule (pred < 30) | $60,063,500 | 22 | 19 | 209 |
| Risk-aware MILP | $19,072,500 | 2 | 49 | 1215 |

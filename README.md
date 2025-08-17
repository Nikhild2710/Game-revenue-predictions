# Game-revenue-predictions

# 🎮 Game Revenue Predictions (Paid PC Games)

Predict the **revenue of paid PC games** from game metadata using machine-learning models (Random Forest and XGBoost).  
The training dataset is **proprietary and not included**; the notebook runs with any file that matches the schema below.

---

## Why this project?

- **Business problem:** Forecasting revenue helps with portfolio planning, pricing, and marketing decisions.
- **Modeling challenge:** Game revenues are **extremely skewed** (many small titles, a few blockbusters).  
  Optimizing only absolute error favors big titles; optimizing only relative error favors small/indie titles.  
  We compare models/targets to make this trade-off explicit.

---

## Data schema (expected columns)

Sheet name: **`paid pc games`**

| Column            | Type     | Notes                                                                 |
|-------------------|----------|-----------------------------------------------------------------------|
| `price`           | float    | USD price at release/sale window                                     |
| `total_reviews`   | int      | Total Steam reviews                                                  |
| `total_positive`  | int      | Positive reviews                                                     |
| `total_negative`  | int      | Negative reviews                                                     |
| `reviewScore`     | float    | Steam review score (0–100 style)                                     |
| `peak_all_time`   | int      | All-time peak concurrent players                                     |
| `publisherClass`  | category | One of {AAA, AA, Indie, Hobbyist}                                    |
| `firstReleaseDate`| date     | *(optional)* used only if `release_year` is enabled                  |
| `revenue`         | float    | Target variable (USD)                                                |

Derived in notebook:
- `positive_ratio = total_positive / total_reviews` (safe division)
- *Optional:* `release_year = year(firstReleaseDate)`

**Excluded to avoid leakage:** `copiesSold` (it’s one-to-one with revenue at fixed price).

---

## Feature engineering & preprocessing

- Drop `copiesSold`
- Add `positive_ratio` (safe divide with zero-guard)
- One-hot encode `publisherClass`
- Median impute numeric, mode impute categorical
- No scaling (tree models are scale-invariant)
- **MAPE safety**: use `max(|y|, 1000)` in denominator to avoid tiny-value explosions; also report filtered MAPE for `revenue ≥ $1k`.

---

## Models compared

We train/evaluate **four** variants to surface the absolute vs relative accuracy trade-off:

1) **RandomForest (raw revenue)** — strong all-round baseline  
2) **RandomForest (log revenue)** — trains on `log1p(revenue)` and predicts with `expm1`  
3) **XGBoost (raw revenue)** — boosting, tuned modestly  
4) **XGBoost (log revenue)** — boosting on log target, predicts back to USD

Training details:
- 80/20 train-test split (fixed `random_state`)
- `RandomForestRegressor(n_estimators=150)`
- `XGBRegressor(tree_method="hist", subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)`  
  + early stopping when available

---

## Metrics & intuition

- **R²** *(higher is better)* — “How much better than predicting the mean?” (variance explained)
- **MAE** *(lower is better)* — average absolute error in dollars
- **RMSE** *(lower is better)* — penalizes large errors (typical big error)
- **MAPE** *(lower is better)* — average % error; can explode when actual revenue is tiny (we guard with ε=1000)

> **Rule-of-thumb:**  
> - Optimize **RMSE/MAE** when blockbusters matter most.  
> - Optimize **MAPE** when fair % accuracy across small/large titles matters.

---

## Results (this run)

| Model                | R²   | MAE ($) | RMSE ($) | MAPE (%) |
|---------------------|------|---------|----------|----------|
| RandomForest (raw)  | 0.761| 187,899 | 3,231,353| 52.75    |
| RandomForest (log)  | 0.739| 192,666 | 3,375,711| 37.75    |
| XGBoost (raw)       | 0.540| 192,510 | 4,482,505| 35.77    |
| XGBoost (log)       | 0.540| 192,510 | 4,482,505| 35.77    |

**Interpretation**
- **RF (raw)** explains the most variance and has the lowest RMSE → best absolute accuracy on average.
- **RF (log)** trades a little R² for much better MAPE → fairer relative performance across small/indie titles.
- **XGBoost** (in this configuration) didn’t beat RF on R²/RMSE, but achieves low MAPE (good relative accuracy).
- The log target makes models focus on **relative differences** (being off by 2× is equally bad for small and big titles).

---

## How to run (Colab)

1. Open the notebook in Colab: `Games_Revenue_Predictor.ipynb`
2. (Optional) **Mount Drive** and set your private dataset path:
3. Run cells top-to-bottom:
- Load data → preprocess → train/evaluate → compare models
4. Save the best model:

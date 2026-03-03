Below is a **full YAML model catalog template** you can use as the “single source of truth” for your DQ model layer (statistical + forecasting + ML + rules), **plus** a **normalization framework** that makes ~20 heterogeneous model outputs comparable on a single **0–100 severity** scale.

I’m assuming your catalog needs to work across **asset classes / rf hierarchies**, and support “plug & play” with libraries like **Merlion** (time-series anomaly/forecast/change-point) and **PyOD** (outlier detection), plus calibration tooling from **scikit-learn**. ([GitHub][1])

---

## 1) YAML model catalog template

```yaml
# dq_model_catalog.yaml
catalog_version: "1.0"
owner: "RFDM-DQ-ControlTower"
last_updated_utc: "2026-03-03T00:00:00Z"

# -------------------------
# Global conventions
# -------------------------
conventions:
  time:
    frequency: "B"              # business day
    timezone: "America/New_York"
    lookback_days_default: 260   # ~1y business days
  identifiers:
    rf_id_field: "risk_factor_id"
    rf_path_fields: ["rf_level1","rf_level2","rf_level3","rf_level4","rf_level5"]
  severity:
    scale: "0-100"
    buckets:
      - {name: "S0", range: [0, 19], meaning: "Normal"}
      - {name: "S1", range: [20, 39], meaning: "Mild"}
      - {name: "S2", range: [40, 59], meaning: "Moderate"}
      - {name: "S3", range: [60, 74], meaning: "High"}
      - {name: "S4", range: [75, 89], meaning: "Severe"}
      - {name: "S5", range: [90, 100], meaning: "Critical"}
  outputs:
    required_fields:
      - "timestamp"
      - "risk_factor_id"
      - "model_id"
      - "raw_score"          # model-native score (may be unbounded)
      - "p_value_or_p_anom"  # normalized probability-like signal (0..1)
      - "severity_0_100"
      - "flags"              # list of tags: ["missing", "stale", "jump", ...]
      - "explanations"       # key factors / SHAP top features etc.

# -------------------------
# Feature sets (reusable)
# -------------------------
feature_sets:

  fs_stat_core_v1:
    description: "Core rolling stats + shock metrics per series"
    inputs:
      series: ["x_t"]
    features:
      - {name: "delta_1d", formula: "x_t - x_t-1"}
      - {name: "pct_change_1d", formula: "(x_t - x_t-1)/max(|x_t-1|, eps)"}
      - {name: "roll_mean_20d", formula: "mean(x_{t-19..t})"}
      - {name: "roll_std_20d", formula: "std(x_{t-19..t})"}
      - {name: "z_20d", formula: "(x_t - roll_mean_20d)/max(roll_std_20d, eps)"}
      - {name: "ewma_0p94", formula: "EWMA(x, lambda=0.94)"}
      - {name: "staleness_days", formula: "days_since_last_change(x)"}
      - {name: "missing_run", formula: "consecutive_missing_count(x)"}
    guards:
      min_history: 60
      max_missing_frac: 0.2

  fs_peer_context_v1:
    description: "Cross-sectional peer deviations using rf hierarchy (sector/rating/tenor/curve)"
    inputs:
      series: ["x_t"]
      peer_group_query: "rf_level1..5 filters"
    features:
      - {name: "peer_median_t", formula: "median(peer_values_t)"}
      - {name: "peer_mad_t", formula: "MAD(peer_values_t)"}
      - {name: "peer_dev", formula: "x_t - peer_median_t"}
      - {name: "peer_robust_z", formula: "(x_t - peer_median_t)/max(peer_mad_t, eps)"}
      - {name: "peer_percentile", formula: "rank_percentile(x_t within peer_values_t)"}

  fs_curve_shape_v1:
    description: "Curve/surface consistency metrics (term structure & smiles)"
    inputs:
      nodes: ["x_t(tenor_i) for i in curve_nodes"]
    features:
      - {name: "slope_2s10s", formula: "x(10Y)-x(2Y)"}
      - {name: "curvature", formula: "x(5Y) - 0.5*(x(2Y)+x(10Y))"}
      - {name: "roughness_l2", formula: "sum_i (x(i+1)-x(i))^2"}
      - {name: "interp_error", formula: "abs(x(node) - spline_interp(neighbors))"}

# -------------------------
# Normalization framework (global policy)
# -------------------------
normalization:
  strategy: "two_stage"  # (A) distribution-to-probability, (B) probability-to-severity
  stage_A_score_to_prob:
    method: "empirical_cdf_tail"   # robust for heterogeneous raw scores
    scope: "per_model_per_series_family"
    series_family_keys:
      - "rf_level1"   # asset class
      - "rf_level2"   # currency (or region)
      - "rf_level4"   # subtype (optional)
    tail:
      enabled: true
      method: "evt_gpd"            # optional: fit Generalized Pareto to extreme tail
      tail_quantile_start: 0.95
  stage_B_prob_to_severity:
    method: "logit_scaled"
    mapping:
      clamp_eps: 1.0e-6
      formula: "severity = 100 * sigmoid(a * logit(p) + b)"
    parameters:
      a: 1.0
      b: 0.0
  calibration:
    enabled: true
    when_labels_available: true
    method: "isotonic_or_platt"
    isotonic:
      min_samples: 2000
    platt:
      fallback_when_data_scarce: true

# -------------------------
# Models
# -------------------------
models:

  - model_id: "stat_zscore_v1"
    family: "statistical"
    task: "point_anomaly"
    description: "Rolling z-score + staleness checks"
    feature_set: "fs_stat_core_v1"
    engine:
      type: "local"
      algorithm: "zscore_rules"
    scoring:
      raw_score_definition: "|z_20d|"
      higher_is_more_anomalous: true
    thresholds:
      warn: 3.0
      crit: 5.0
    normalization_profile: "default"
    tags: ["baseline","fast","explainable"]

  - model_id: "forecast_arima_resid_v1"
    family: "forecasting"
    task: "residual_anomaly"
    description: "ARIMA forecast; residual magnitude indicates anomaly"
    feature_set: "fs_stat_core_v1"
    engine:
      type: "statsmodels"
      algorithm: "ARIMA"
      hyperparams:
        auto_arima: true
        seasonal: false
    scoring:
      raw_score_definition: "|residual_t| / max(resid_std_60d, eps)"
      higher_is_more_anomalous: true
    normalization_profile: "default"
    tags: ["forecast","explainable"]

  - model_id: "pyod_iforest_v1"
    family: "ml_unsupervised"
    task: "outlier_detection"
    description: "Isolation Forest via PyOD; uses decision_function raw anomaly score"
    feature_set: "fs_stat_core_v1"
    engine:
      type: "pyod"
      algorithm: "IForest"
      hyperparams:
        n_estimators: 300
        contamination: "auto"
        random_state: 42
    scoring:
      raw_score_field: "decision_function"
      higher_is_more_anomalous: true
    normalization_profile: "default"
    tags: ["pyod","unsupervised"]

  - model_id: "xgb_suspect_classifier_v1"
    family: "ml_supervised"
    task: "classification"
    description: "XGBoost probability of suspect using historical labels/overrides"
    feature_sets: ["fs_stat_core_v1","fs_peer_context_v1","fs_curve_shape_v1"]
    label:
      name: "suspect_flag"
      positive_meaning: "bad_or_suspect"
      source: "RFDM_suspecting_overrides"
    engine:
      type: "xgboost"
      algorithm: "XGBClassifier"
      hyperparams:
        max_depth: 4
        n_estimators: 800
        learning_rate: 0.03
        subsample: 0.8
        colsample_bytree: 0.8
    scoring:
      raw_score_definition: "p_hat"     # model predicted probability (uncalibrated)
      higher_is_more_anomalous: true
    calibration_profile: "global_calibration_policy"
    explainability:
      method: "shap"
      top_k: 8
    normalization_profile: "probability_direct"  # skips Stage A; uses calibration + Stage B
    tags: ["xgb","supervised","shap"]

# -------------------------
# Execution & governance
# -------------------------
execution:
  schedule:
    daily_batch_time_local: "06:00"
    backfill:
      enabled: true
      max_days: 3650
  compute:
    parallelism:
      by_asset_class: true
      by_rf_partition: 256
  storage:
    scores_table: "dq_scores_by_rf_day"
    feature_store_table: "dq_features_by_rf_day"
    lineage_table: "dq_lineage"
  monitoring:
    drift:
      track_score_distribution: true
      psi_threshold: 0.2
    performance:
      track_precision_recall: true
      by_asset_class: true
    alerts:
      - when: "severity_0_100 >= 90"
        route: ["dq_slack","dq_email"]
        throttle_minutes: 60
```

### Why this structure works

* It supports **Merlion-style unified interfaces across anomaly + forecasting + change-point** (you can map your `engine.type` adapters to Merlion implementations). ([GitHub][1])
* It supports **PyOD “raw anomaly score” via `decision_function`** (which is exactly how PyOD standardizes scoring across detectors). ([pyod.readthedocs.io][2])
* It embeds a **global normalization policy** so every model, regardless of native score range, lands on the same severity scale.

---

## 2) Scoring normalization framework (20 models → comparable severity)

You want two things simultaneously:

1. **Comparability** across models (IForest score vs ARIMA residual vs rule z-score vs XGB probability)
2. **Stability** across regimes & asset classes (credit vs rates vs vol surfaces)

### Proposed approach: Two-stage normalization (robust + auditable)

#### Stage A — Convert model-native raw score → probability-like signal (0..1)

Use an **empirical CDF** per model, per “series family” (e.g., asset class + currency + subtype). Intuition:

> “How extreme is today’s raw score compared to this model’s historical scores for similar instruments?”

**Definition (empirical):**

* Let raw score be ( s_t )
* Build historical reference distribution ( {s_{t-L},...,s_{t-1}} )
* Define
  [
  p_t = \widehat{F}(s_t) \in [0,1]
  ]
  where (\widehat{F}) is the empirical CDF.

This makes wildly different scales comparable because everything becomes a **percentile / tail probability**.

**Tail option (recommended): EVT on top 5%**
For the extreme tail (e.g., above 95th percentile), fit a tail model (e.g., Generalized Pareto) so the “top 0.1%” is meaningfully separated from “top 2%”. (This is optional but helpful for prioritizing the worst events.)

> Practical benefit: you avoid having a dozen “all maxed-out” scores on stress days.

*(If you don’t want EVT, skip it—empirical percentile alone is already a big win.)*

#### Stage B — Convert probability-like signal → severity 0..100

Once you have (p_t), map it to severity using a monotonic transform that expands the tail:

* Compute logit: ( \text{logit}(p) = \ln\frac{p}{1-p} )
* Then:
  [
  \text{severity} = 100 \cdot \sigma(a \cdot \text{logit}(p) + b)
  ]
  where (\sigma) is the sigmoid, and (a,b) tune how aggressive you want the tail.

This yields:

* Smooth 0–100
* Strong differentiation in the 90–100 range
* Stable interpretation (“90+ is always extreme compared to history”)

---

## 3) Calibration when you have labels (critical for XGBoost and any supervised model)

Raw probabilities from classifiers are often **not calibrated**. Use **scikit-learn calibration**:

* **Platt scaling** (sigmoid) when data is scarce
* **Isotonic regression** when you have enough samples (more flexible but can overfit)

scikit-learn docs: probability calibration & isotonic regression ([Scikit-learn][3])
(Platt vs isotonic trade-off is widely discussed; isotonic can overfit with limited data.) ([MachineLearningMastery.com][4])

**Where calibration fits:**

* For supervised models: `p_hat` → calibrated `p_cal` → Stage B mapping to severity
* For unsupervised/forecast models: Stage A percentile gives `p` already (no labels needed)

---

## 4) How to make it operational in RFDM terms (what you actually store)

For each (rf_id, date, model_id), store:

* `raw_score` (model native)
* `p_anom` (after Stage A or after calibration)
* `severity_0_100` (after Stage B)
* plus metadata: peer group used, lookback window, drift stats, top explanations

This is aligned with PyOD’s concept that detectors produce **raw anomaly scores** (larger = more anomalous) but those scores are not inherently comparable across models—hence your normalization layer. ([pyod.readthedocs.io][2])

---

## 5) Source links (official docs)

```text
Merlion (framework overview): https://github.com/salesforce/Merlion
Merlion tutorials/docs: https://opensource.salesforce.com/Merlion/tutorials.html

PyOD model docs (decision_function anomaly score): https://pyod.readthedocs.io/en/latest/pyod.models.html
PyOD examples: https://pyod.readthedocs.io/en/latest/example.html

scikit-learn probability calibration: https://scikit-learn.org/stable/modules/calibration.html
scikit-learn isotonic regression: https://scikit-learn.org/stable/modules/isotonic.html
```

---

If you want, I can also give you:

* a **second YAML file** that defines **peer-group rules** (how to compute “sector/rating/tenor peers” per asset class), or
* a concrete “**20-model starter pack**” (model list + default hyperparams + which feature sets each uses) aligned to your rf_level1 buckets (CP/IR/EQ/FX/COMM rate/vol).

[1]: https://github.com/salesforce/Merlion?utm_source=chatgpt.com "Merlion: A Machine Learning Framework for Time Series ..."
[2]: https://pyod.readthedocs.io/en/latest/pyod.models.html?utm_source=chatgpt.com "All Models - pyod 2.0.6 documentation"
[3]: https://scikit-learn.org/stable/modules/calibration.html?utm_source=chatgpt.com "1.16. Probability calibration"
[4]: https://www.machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/?utm_source=chatgpt.com "How and When to Use a Calibrated Classification Model ..."

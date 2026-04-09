import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ─────────────────────────────────────────────────────────────────
# SCENARIO A: IR Rate — interpolate carry from existing curve
# ─────────────────────────────────────────────────────────────────

def strip_carry_ir(
    rf_series: pd.Series,                        # daily RF time series (bps)
    curve_history: dict[str, dict[float, float]],# {date_str: {tenor_days: rate_bps}}
    tau_days: float,                             # tenor of your risk factor in days
    N: int = 1,                                  # 1 for daily, 10 for 10d window
) -> pd.DataFrame:
    """
    Strip carry from a pre-computed IR risk factor series.
    Uses existing curve snapshots — no recomputation needed.
    """
    results = []

    dates = sorted(rf_series.index[1:])  # skip first (no prior day)

    for date in dates:
        prev_date = rf_series.index[rf_series.index.get_loc(date) - 1]

        raw_change = rf_series[date] - rf_series[prev_date]

        # Carry from prior-day curve
        if str(prev_date) in curve_history:
            curve = curve_history[str(prev_date)]
            tenors = sorted(curve.keys())
            rates  = [curve[t] for t in tenors]
            interp = interp1d(tenors, rates, kind='linear',
                              fill_value='extrapolate')

            rf_tau     = float(interp(tau_days))
            rf_tau_N   = float(interp(tau_days - N))
            carry      = rf_tau_N - rf_tau
        else:
            carry = np.nan

        results.append({
            "date":            date,
            "raw_change":      raw_change,
            "carry":           carry,
            "carry_adjusted":  raw_change - carry if not np.isnan(carry) else np.nan,
        })

    return pd.DataFrame(results).set_index("date")


# ─────────────────────────────────────────────────────────────────
# SCENARIO B: Vol Surface — strip theta as carry
# ─────────────────────────────────────────────────────────────────

def bs_theta_vol_carry(
    vol_bps: float,       # current implied vol in bps (e.g. 100 = 1%)
    expiry_days: float,   # time to expiry in days
    N: int = 1,
) -> float:
    """
    Approximate carry for an implied vol point using vol-of-vol theta.
    For a vol surface, carry ≈ change in vol due to expiry shortening by N days.

    Simplified: uses sqrt(T) scaling to estimate how vol changes as T shrinks.
    vol(T-N) ≈ vol(T) × sqrt(T / (T-N))  — flat term structure approximation.
    """
    vol_T    = vol_bps
    vol_T_N  = vol_bps * np.sqrt(expiry_days / (expiry_days - N))
    carry    = vol_T_N - vol_T   # positive: vol rises as expiry shortens (typically)
    return carry


def strip_carry_vol_surface(
    vol_series: pd.Series,    # daily vol time series (bps)
    expiry_days: float,       # expiry tenor of the vol point
    N: int = 1,
) -> pd.DataFrame:
    """
    Strip theta-carry from a pre-computed vol risk factor series.
    Uses BS sqrt(T) approximation when full surface not available.
    """
    results = []
    for i in range(1, len(vol_series)):
        date       = vol_series.index[i]
        prev_vol   = vol_series.iloc[i - 1]
        raw_change = vol_series.iloc[i] - prev_vol

        # Carry = theta approximation on prior day's expiry
        carry = bs_theta_vol_carry(prev_vol, expiry_days, N)

        results.append({
            "date":           date,
            "raw_change":     raw_change,
            "carry":          carry,
            "carry_adjusted": raw_change - carry,
        })

    return pd.DataFrame(results).set_index("date")


# ─────────────────────────────────────────────────────────────────
# SCENARIO C: RF series only — estimate carry statistically
# ─────────────────────────────────────────────────────────────────

def strip_carry_statistical(
    rf_series: pd.Series,
    window: int = 60,      # rolling window to estimate carry drift
) -> pd.DataFrame:
    """
    When no curve is available, estimate carry as the rolling median
    of daily changes. The median is robust to outlier shocks and
    approximates the deterministic drift component.

    carry_adjusted = raw_change - rolling_median(raw_change)
    """
    raw_changes = rf_series.diff().dropna()

    # Rolling median as carry estimator (robust to spikes)
    carry_est = raw_changes.rolling(window, min_periods=max(10, window // 3)).median()

    df = pd.DataFrame({
        "raw_change":     raw_changes,
        "carry":          carry_est,
        "carry_adjusted": raw_changes - carry_est,
    })
    return df


# ─────────────────────────────────────────────────────────────────
# USAGE EXAMPLE
# ─────────────────────────────────────────────────────────────────

np.random.seed(42)
dates = pd.date_range("2025-01-02", periods=252, freq="B")

# Simulate 2Y USD rate with roll-down drift + random shocks
carry_per_day = -0.164   # bps/day roll-down at 2Y
true_shocks   = np.random.normal(0, 2, 252)
rf_2y = pd.Series(
    420 + np.cumsum(carry_per_day + true_shocks),
    index=dates, name="IR_RATE_USD_2Y"
)

# --- Scenario A: with curve history ---
curve_history = {
    str(d.date()): {
        365: 470 + np.random.normal(0, 0.5),
        545: 450 + np.random.normal(0, 0.5),
        730: float(rf_2y[d]),               # anchor 2Y to actual RF
       1095: 400 + np.random.normal(0, 0.5),
    }
    for d in dates
}

result_A = strip_carry_ir(rf_2y, curve_history, tau_days=730, N=1)

# --- Scenario B: vol surface ---
vol_1y1y = pd.Series(
    100 + np.cumsum(np.random.normal(0.05, 1.5, 252)),
    index=dates, name="IR_VOL_USD_1Y1Y"
)
result_B = strip_carry_vol_surface(vol_1y1y, expiry_days=365, N=1)

# --- Scenario C: statistical (no curve) ---
result_C = strip_carry_statistical(rf_2y, window=60)

# --- Summary ---
for label, df in [("A — IR carry strip", result_A),
                  ("B — Vol theta strip", result_B),
                  ("C — Statistical strip", result_C)]:
    raw = df["raw_change"].std()
    adj = df["carry_adjusted"].dropna().std()
    print(f"{label}")
    print(f"  Raw change std      : {raw:.3f} bps")
    print(f"  Carry-adjusted std  : {adj:.3f} bps")
    print(f"  Carry removed       : {(raw - adj) / raw * 100:.1f}% of variance\n")

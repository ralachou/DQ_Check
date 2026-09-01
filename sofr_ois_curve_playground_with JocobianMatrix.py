from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import brentq

DEFAULT_QUOTES_PCT = {
    "1M": 4.30, "3M": 4.28, "6M": 4.25, "9M": 4.22,
    "1Y": 4.20, "2Y": 4.05, "3Y": 3.95, "4Y": 3.90,
    "5Y": 3.88, "10Y": 3.95, "20Y": 4.02, "30Y": 4.00,
}

TENOR_TO_YEARS = {
    "1M": 1/12, "3M": 3/12, "6M": 6/12, "9M": 9/12,
    "1Y": 1.0, "2Y": 2.0, "3Y": 3.0, "4Y": 4.0,
    "5Y": 5.0, "10Y": 10.0, "20Y": 20.0, "30Y": 30.0,
}


def fixed_payment_times(maturity: float, frequency: int = 1) -> np.ndarray:
    step = 1.0 / frequency
    if maturity <= step + 1e-12:
        return np.array([maturity], dtype=float)
    times = list(np.arange(step, maturity + 1e-12, step))
    if not math.isclose(times[-1], maturity, abs_tol=1e-10):
        times.append(maturity)
    return np.asarray(times, dtype=float)


def accrual_fractions(payment_times: np.ndarray) -> np.ndarray:
    return np.diff(np.concatenate(([0.0], payment_times)))


def log_linear_discount(query_times, node_times, node_dfs) -> np.ndarray:
    q = np.atleast_1d(np.asarray(query_times, dtype=float))
    times = np.concatenate(([0.0], np.asarray(node_times, dtype=float)))
    log_dfs = np.concatenate(([0.0], np.log(np.asarray(node_dfs, dtype=float))))
    if np.any(q < 0) or np.any(q > times[-1] + 1e-12):
        raise ValueError("Requested time is outside the solved curve range.")
    return np.exp(np.interp(q, times, log_dfs))


def zero_rate_cc(df: float, maturity: float) -> float:
    return -math.log(df) / maturity


def discount_factor_from_zero_cc(rate: float, maturity: float) -> float:
    return math.exp(-rate * maturity)


def simple_forward_rate(df_start: float, df_end: float, tau: float) -> float:
    return (df_start / df_end - 1.0) / tau


def continuous_forward_rate(df_start: float, df_end: float, tau: float) -> float:
    return math.log(df_start / df_end) / tau


def continuous_forward_from_zero_rates(z1: float, t1: float, z2: float, t2: float) -> float:
    return (z2 * t2 - z1 * t1) / (t2 - t1)


def par_ois_rate(maturity: float, node_times: np.ndarray, node_dfs: np.ndarray,
                 fixed_frequency: int = 1) -> float:
    pay_times = fixed_payment_times(maturity, fixed_frequency)
    accruals = accrual_fractions(pay_times)
    dfs = log_linear_discount(pay_times, node_times, node_dfs)
    annuity = float(np.dot(accruals, dfs))
    return (1.0 - dfs[-1]) / annuity


@dataclass
class BootstrapResult:
    curve: pd.DataFrame
    node_times: np.ndarray
    node_dfs: np.ndarray


def bootstrap_ois_curve(market_quotes: dict[str, float], fixed_frequency: int = 1) -> BootstrapResult:
    labels = sorted(market_quotes, key=lambda x: TENOR_TO_YEARS[x])
    solved_times, solved_dfs = [], []

    for label in labels:
        maturity = TENOR_TO_YEARS[label]
        quote = market_quotes[label]

        def objective(candidate_df: float) -> float:
            trial_times = np.asarray(solved_times + [maturity])
            trial_dfs = np.asarray(solved_dfs + [candidate_df])
            return par_ois_rate(maturity, trial_times, trial_dfs, fixed_frequency) - quote

        solved_df = brentq(objective, 1e-8, 3.0)
        solved_times.append(maturity)
        solved_dfs.append(solved_df)

    times = np.asarray(solved_times)
    dfs = np.asarray(solved_dfs)
    quotes = np.asarray([market_quotes[x] for x in labels])
    zeros = np.array([zero_rate_cc(df, t) for df, t in zip(dfs, times)])

    previous_times = np.concatenate(([0.0], times[:-1]))
    previous_dfs = np.concatenate(([1.0], dfs[:-1]))
    simple_fwds = np.array([
        simple_forward_rate(p1, p2, t2 - t1)
        for p1, p2, t1, t2 in zip(previous_dfs, dfs, previous_times, times)
    ])
    continuous_fwds = np.array([
        continuous_forward_rate(p1, p2, t2 - t1)
        for p1, p2, t1, t2 in zip(previous_dfs, dfs, previous_times, times)
    ])
    repriced = np.array([
        par_ois_rate(t, times[:i+1], dfs[:i+1], fixed_frequency)
        for i, t in enumerate(times)
    ])

    curve = pd.DataFrame({
        "Tenor": labels,
        "Time (Years)": times,
        "Market Par Rate": quotes,
        "Discount Factor": dfs,
        "Zero Rate (CC)": zeros,
        "Simple Forward": simple_fwds,
        "Continuous Forward": continuous_fwds,
        "Repriced Par Rate": repriced,
        "Repricing Error (bp)": (repriced - quotes) * 10000,
    })
    return BootstrapResult(curve, times, dfs)


def quote_to_forward_jacobian(market_quotes: dict[str, float], fixed_frequency: int,
                              bump_bp: float) -> pd.DataFrame:
    base = bootstrap_ois_curve(market_quotes, fixed_frequency).curve
    base_fwds = base["Simple Forward"].to_numpy()
    labels = base["Tenor"].tolist()
    jac = np.zeros((len(labels), len(labels)))
    bump = bump_bp / 10000

    for j, label in enumerate(labels):
        bumped = dict(market_quotes)
        bumped[label] += bump
        bumped_fwds = bootstrap_ois_curve(bumped, fixed_frequency).curve["Simple Forward"].to_numpy()
        jac[:, j] = (bumped_fwds - base_fwds) * 10000 / bump_bp

    return pd.DataFrame(jac,
        index=[f"Forward ending {x}" for x in labels],
        columns=[f"Quote {x}" for x in labels])


st.set_page_config(page_title="SOFR OIS Curve Playground", layout="wide")
st.title("SOFR OIS Forward Curve Playground")
st.caption("Market par quotes → discount factors → zero rates → forward rates → Jacobian")

with st.expander("Core formulas", expanded=False):
    st.latex(r"K(T)=\frac{1-P(0,T)}{\sum_i \alpha_iP(0,t_i)}")
    st.latex(r"z(T)=-\frac{\ln P(0,T)}{T}")
    st.latex(r"F(T_1,T_2)=\frac{P(0,T_1)/P(0,T_2)-1}{T_2-T_1}")
    st.latex(r"J_{ij}=\frac{\partial f_i}{\partial q_j}")

st.sidebar.header("Curve settings")
frequency = st.sidebar.selectbox("Fixed-leg frequency", [1, 2, 4],
    format_func=lambda x: {1:"Annual", 2:"Semiannual", 4:"Quarterly"}[x])
bump_bp = st.sidebar.number_input("Jacobian bump (bp)", 0.01, 100.0, 1.0, 0.25)

st.subheader("1. Edit hypothetical SOFR OIS par quotes")
quote_table = pd.DataFrame({"Tenor": DEFAULT_QUOTES_PCT.keys(), "Par Rate (%)": DEFAULT_QUOTES_PCT.values()})
edited = st.data_editor(quote_table, hide_index=True, use_container_width=True,
    disabled=["Tenor"], num_rows="fixed")
market_quotes = {r["Tenor"]: float(r["Par Rate (%)"]) / 100 for _, r in edited.iterrows()}

try:
    result = bootstrap_ois_curve(market_quotes, frequency)
except Exception as exc:
    st.error(f"Bootstrap failed: {exc}")
    st.stop()

curve = result.curve.copy()
display = curve.copy()
for col in ["Market Par Rate", "Zero Rate (CC)", "Simple Forward", "Continuous Forward", "Repriced Par Rate"]:
    display[col] *= 100

st.subheader("2. Bootstrapped curve")
st.dataframe(display.style.format({
    "Time (Years)":"{:.6f}", "Market Par Rate":"{:.4f}%", "Discount Factor":"{:.8f}",
    "Zero Rate (CC)":"{:.4f}%", "Simple Forward":"{:.4f}%",
    "Continuous Forward":"{:.4f}%", "Repriced Par Rate":"{:.4f}%",
    "Repricing Error (bp)":"{:.10f}"}), use_container_width=True)
st.download_button("Download curve CSV", display.to_csv(index=False), "sofr_curve.csv", "text/csv")

st.subheader("3. Plot the curve")
choice = st.selectbox("Measure", ["Market Par Rate", "Zero Rate (CC)", "Simple Forward", "Continuous Forward", "Discount Factor"])
fig, ax = plt.subplots()
y = curve[choice].to_numpy() if choice == "Discount Factor" else curve[choice].to_numpy() * 100
ax.plot(curve["Time (Years)"], y, marker="o")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel(choice if choice == "Discount Factor" else f"{choice} (%)")
ax.grid(True)
st.pyplot(fig)

st.subheader("4. Formula playground")
t1, t2, t3, t4 = st.tabs(["DF ↔ Zero", "Forward from DFs", "Forward from Zeros", "Forward from Curve"])

with t1:
    c1, c2 = st.columns(2)
    with c1:
        df = st.number_input("P(0,T)", 0.000001, 3.0, 0.95, 0.001)
        T = st.number_input("T (years)", 0.000001, 100.0, 2.0, 0.25, key="zt")
        st.metric("Continuous zero rate", f"{zero_rate_cc(df,T)*100:.6f}%")
    with c2:
        z = st.number_input("Zero rate (%)", -20.0, 50.0, 4.0, 0.1)
        Tz = st.number_input("T (years)", 0.0, 100.0, 2.0, 0.25, key="dft")
        st.metric("Discount factor", f"{discount_factor_from_zero_cc(z/100,Tz):.8f}")

with t2:
    p1 = st.number_input("P(0,T1)", 0.000001, 3.0, 0.96, 0.001)
    p2 = st.number_input("P(0,T2)", 0.000001, 3.0, 0.92, 0.001)
    tau = st.number_input("Year fraction", 0.000001, 100.0, 1.0, 0.25)
    a,b = st.columns(2)
    a.metric("Simple forward", f"{simple_forward_rate(p1,p2,tau)*100:.6f}%")
    b.metric("Continuous forward", f"{continuous_forward_rate(p1,p2,tau)*100:.6f}%")

with t3:
    c1,c2,c3,c4 = st.columns(4)
    T1 = c1.number_input("T1", 0.0, 100.0, 2.0, 0.25)
    Z1 = c2.number_input("z(T1) %", -20.0, 50.0, 4.0, 0.1)
    T2 = c3.number_input("T2", 0.000001, 100.0, 3.0, 0.25)
    Z2 = c4.number_input("z(T2) %", -20.0, 50.0, 4.2, 0.1)
    if T2 > T1:
        st.metric("Continuous forward", f"{continuous_forward_from_zero_rates(Z1/100,T1,Z2/100,T2)*100:.6f}%")
    else:
        st.warning("T2 must be greater than T1.")

with t4:
    max_t = float(result.node_times[-1])
    c1,c2 = st.columns(2)
    A = c1.number_input("Forward start T1", 0.0, max_t-0.000001, min(2.0,max_t/3), 0.25)
    B = c2.number_input("Forward end T2", 0.000001, max_t, min(3.0,max_t), 0.25)
    if B > A:
        pa = 1.0 if A == 0 else float(log_linear_discount(A,result.node_times,result.node_dfs)[0])
        pb = float(log_linear_discount(B,result.node_times,result.node_dfs)[0])
        c1,c2,c3 = st.columns(3)
        c1.metric("P(0,T1)", f"{pa:.8f}")
        c2.metric("P(0,T2)", f"{pb:.8f}")
        c3.metric("Simple forward", f"{simple_forward_rate(pa,pb,B-A)*100:.6f}%")
    else:
        st.warning("T2 must be greater than T1.")

st.subheader("5. Par-to-forward Jacobian")
jac = quote_to_forward_jacobian(market_quotes, frequency, bump_bp)
st.dataframe(jac.style.format("{:.6f}"), use_container_width=True)
st.download_button("Download Jacobian CSV", jac.to_csv(), "par_to_forward_jacobian.csv", "text/csv")
fig2, ax2 = plt.subplots(figsize=(10,7))
im = ax2.imshow(jac.to_numpy(), aspect="auto")
ax2.set_xticks(range(len(jac.columns))); ax2.set_xticklabels(jac.columns, rotation=90)
ax2.set_yticks(range(len(jac.index))); ax2.set_yticklabels(jac.index)
fig2.colorbar(im, ax=ax2, label="Forward move (bp) per quote bump")
fig2.tight_layout(); st.pyplot(fig2)

st.subheader("6. Single quote bump experiment")
label = st.selectbox("Quote to bump", list(market_quotes))
manual_bump = st.slider("Bump size (bp)", -100.0, 100.0, 1.0, 1.0)
bumped_quotes = dict(market_quotes); bumped_quotes[label] += manual_bump/10000
bumped_curve = bootstrap_ois_curve(bumped_quotes, frequency).curve
comparison = pd.DataFrame({
    "Tenor": curve["Tenor"],
    "Base Forward (%)": curve["Simple Forward"]*100,
    "Bumped Forward (%)": bumped_curve["Simple Forward"]*100,
})
comparison["Change (bp)"] = (comparison["Bumped Forward (%)"]-comparison["Base Forward (%)"])*100
st.dataframe(comparison.style.format({"Base Forward (%)":"{:.6f}%", "Bumped Forward (%)":"{:.6f}%", "Change (bp)":"{:.6f}"}), use_container_width=True)

fig3, ax3 = plt.subplots()
ax3.plot(curve["Time (Years)"], comparison["Base Forward (%)"], marker="o", label="Base")
ax3.plot(curve["Time (Years)"], comparison["Bumped Forward (%)"], marker="o", label="Bumped")
ax3.set_xlabel("Maturity (years)"); ax3.set_ylabel("Simple forward (%)"); ax3.legend(); ax3.grid(True)
st.pyplot(fig3)

st.subheader("7. Repricing validation")
max_error = float(np.max(np.abs(curve["Repricing Error (bp)"])))
st.metric("Maximum absolute repricing error", f"{max_error:.10f} bp")
if max_error < 1e-6:
    st.success("Every input par quote is reproduced to numerical precision.")

with st.expander("Educational limitations"):
    st.write("This PoC simplifies calendars, actual dates, Actual/360, payment lags, SOFR daily compounding details, futures convexity, stubs, bid/offer selection, and production interpolation governance.")

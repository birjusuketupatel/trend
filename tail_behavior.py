import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

LAG_MONTHS = 10
DAYS_IN_MONTH = 21

df = pd.read_csv(
    "data/us_stocks.csv",
    parse_dates=["Date"],
    index_col="Date",
).sort_index()

# Compute 21-day EWMA volatility
df["ewma_vol"] = (
    df["stock_mkt"]
    .ewm(span=DAYS_IN_MONTH, adjust=True)
    .std()
)

# Compute average return over past 10 months trend signal
month = df.index.to_period("M")

# Compound each calendar month's daily returns
monthly_returns = (
    df["stock_mkt"]
    .add(1)
    .groupby(month)
    .prod(min_count=1)
    .sub(1)
)

# Identify each month's first observed trading day
first_trading_dates = df.groupby(month).head(1).index

df["ret_monthly"] = np.nan

# Put the previous month's return on the current month's first trading day
df.loc[first_trading_dates, "ret_monthly"] = (
    monthly_returns.shift(1).to_numpy()
)
    
# Get simple average of past LAG_MONTHS months of monthly returns
lagged_avg_ret = (
    df.loc[first_trading_dates, "ret_monthly"]
    .rolling(window=LAG_MONTHS)
    .mean()
)

df.loc[first_trading_dates, "trend"] = lagged_avg_ret

df = df.resample('MS').first()

df["fwd_ret"] = df["ret_monthly"].shift(-1)

# Compute trend direction
# Coerce zero values to +1
df["trend_dir"] = np.sign(df["trend"]) + (df["trend"] == 0)

# Select relevant features
df = df[
    ["trend_dir", "fwd_ret"]
].dropna()

print(df.tail())

# Compute quintile rank of each month's return
# 0 = lowest forward-return quintile; 4 = highest
df["ret_rank"] = pd.qcut(
    df["fwd_ret"],
    q=5,
    labels=range(0, 5)
).astype(int)

# Print quintile boundaries
quantiles = df["fwd_ret"].quantile(np.linspace(0, 1, 6))
quantiles.index = ["Minimum", "20th", "40th", "60th", "80th", "Maximum"]

print("\nQuintile Boundaries:")
print(quantiles.map(lambda x: f"{x:.2%}").to_string())

# Compare lower-tail event probabilities across trend regimes using common
# percentile thresholds estimated from the full sample.
tail_percentiles = pd.Series(
    [0.20, 0.10, 0.05],
    index=["20th percentile", "10th percentile", "5th percentile"],
)
tail_thresholds = df["fwd_ret"].quantile(tail_percentiles.to_numpy())

trend_on_returns = df.loc[df["trend_dir"] == 1, "fwd_ret"]
trend_off_returns = df.loc[df["trend_dir"] == -1, "fwd_ret"]

trend_on_probabilities = np.array([
    (trend_on_returns < threshold).mean()
    for threshold in tail_thresholds
])
trend_off_probabilities = np.array([
    (trend_off_returns < threshold).mean()
    for threshold in tail_thresholds
])

x = np.arange(len(tail_percentiles))
bar_width = 0.36

fig, ax = plt.subplots(figsize=(9, 5))

trend_on_bars = ax.bar(
    x - bar_width / 2,
    trend_on_probabilities,
    bar_width,
    color="blue",
    alpha=0.5,
    label="Positive trend",
)
trend_off_bars = ax.bar(
    x + bar_width / 2,
    trend_off_probabilities,
    bar_width,
    color="red",
    alpha=0.5,
    label="Negative trend",
)

ax.bar_label(
    trend_on_bars,
    labels=[f"{probability:.1%}" for probability in trend_on_probabilities],
    padding=3,
)
ax.bar_label(
    trend_off_bars,
    labels=[f"{probability:.1%}" for probability in trend_off_probabilities],
    padding=3,
)
ax.set_xticks(
    x,
    [f"< {threshold:.2%}" for threshold in tail_thresholds],
)
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_xlabel("Full-Sample Forward-Return Threshold")
ax.set_ylabel("Empirical Probability")
ax.set_title("Crash Probability by Trend Regime")
ax.legend(frameon=False)
ax.set_ylim(
    0,
    max(trend_on_probabilities.max(), trend_off_probabilities.max()) * 1.15,
)
fig.tight_layout()
plt.show()

# Statistical tests to determine if disproportionate number of crashes and rallies occur during trend off months
num_obs = len(df)
crash_and_trend_on = len(df.query("trend_dir == 1 and ret_rank == 0"))
crash = len(df.query("ret_rank == 0"))
trend_on = len(df.query("trend_dir == 1"))

print(f"\nP(Crash) = {crash / num_obs:.2%}")
print(f"P(Crash | Trend On) = {crash_and_trend_on / trend_on:.2%}")
print(f"P(Crash | Trend Off) = {(crash - crash_and_trend_on) / (num_obs - trend_on):.2%}")
    
# Tests P(Crash | Trend On) < P(Crash | Trend Off)
z_stat, p_value = proportions_ztest(
    count=[crash_and_trend_on, crash - crash_and_trend_on],
    nobs=[trend_on, num_obs - trend_on],
    alternative="smaller"
)

print(f"\nz-statistic: {z_stat:.3f}")
print(f"One-sided p-value: {p_value:.4g}")

rally_and_trend_on = len(df.query("trend_dir == 1 and ret_rank == 4"))
rally = len(df.query("ret_rank == 4"))

print(f"\nP(Rally) = {rally / num_obs:.2%}")
print(f"P(Rally | Trend On) = {rally_and_trend_on / trend_on:.2%}")
print(f"P(Rally | Trend Off) = {(rally - rally_and_trend_on) / (num_obs - trend_on):.2%}")

# Tests P(Rally | Trend On) < P(Rally | Trend Off)
z_stat, p_value = proportions_ztest(
    count=[rally_and_trend_on, rally - rally_and_trend_on],
    nobs=[trend_on, num_obs - trend_on],
    alternative="smaller"
)

print(f"\nz-statistic: {z_stat:.3f}")
print(f"One-sided p-value: {p_value:.4g}")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_ind

LAG_MONTHS = 10

df = pd.read_csv(
    "data/us_stocks.csv",
    parse_dates=["Date"],
    index_col="Date",
).sort_index()


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

df = df[
    ["ret_monthly", "trend", "fwd_ret"]
].dropna()

df["trend_dir"] = np.sign(df["trend"]) + (df["trend"] == 0)

print(df.tail())

# Plot PACF of monthly returns
sm.graphics.tsa.plot_pacf(
    df["ret_monthly"],
    lags=18,
    alpha=0.05,
)

plt.title("Partial Autocorrelation of Monthly Stock Returns")
plt.show()

# Regress next month's return on the past 10-month average return
X = sm.add_constant(df["trend"])
y = df["fwd_ret"]

model = sm.OLS(y, X).fit()

print(model.summary())


# Scatter plot and fitted OLS line
x_plot = np.linspace(
    df["trend"].min(),
    df["trend"].max(),
    200,
)

y_plot = (
    model.params["const"]
    + model.params["trend"] * x_plot
)

plt.figure(figsize=(9, 5))

plt.scatter(
    df["trend"],
    df["fwd_ret"],
    alpha=0.5,
    s=25,
    color="steelblue",
)

plt.plot(
    x_plot,
    y_plot,
    color="firebrick",
    linewidth=2,
    label="OLS fit",
)

plt.axhline(0, color="black", linewidth=0.8, alpha=0.6)
plt.axvline(0, color="black", linewidth=0.8, alpha=0.6)

plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

plt.xlabel("Past 10-Month Average Return")
plt.ylabel("Forward Monthly Return")
plt.title("Past Returns and Next-Month Stock Returns")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# Overlapping histograms of forward returns by trend regime
trend_on = df.loc[df["trend_dir"] == 1, "fwd_ret"]
trend_off = df.loc[df["trend_dir"] == -1, "fwd_ret"]

# Common bins ensure the two distributions are directly comparable
bins = np.linspace(
    df["fwd_ret"].min(),
    df["fwd_ret"].max(),
    40,
)

plt.figure(figsize=(9, 5))

plt.hist(
    trend_on,
    bins=bins,
    density=True,
    alpha=0.55,
    color="steelblue",
    label=f"Positive trend (n={len(trend_on)})",
)

plt.hist(
    trend_off,
    bins=bins,
    density=True,
    alpha=0.55,
    color="firebrick",
    label=f"Negative trend (n={len(trend_off)})",
)

plt.axvline(0, color="black", linewidth=0.8, alpha=0.7)

plt.gca().xaxis.set_major_formatter(PercentFormatter(1))

plt.xlabel("Forward Monthly Return")
plt.ylabel("Probability Density")
plt.title("Distribution of Next-Month Returns by Trend")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# Test whether mean forward returns differ between trend regimes
trend_on = df.loc[df["trend_dir"] == 1, "fwd_ret"]
trend_off = df.loc[df["trend_dir"] == -1, "fwd_ret"]

t_stat, p_value = ttest_ind(
    trend_on,
    trend_off,
    equal_var=False,
    nan_policy="omit",
)

mean_on = trend_on.mean()
mean_off = trend_off.mean()
mean_difference = mean_on - mean_off

print("\nDifference in mean forward returns")
print(f"Positive-trend mean: {mean_on:.4%}")
print(f"Negative-trend mean: {mean_off:.4%}")
print(f"Mean difference:      {mean_difference:.4%}")
print(f"Welch t-statistic:    {t_stat:.3f}")
print(f"Two-sided p-value:    {p_value:.4f}")
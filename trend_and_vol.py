import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

DAYS_IN_MONTH = 21
LAG_MONTHS = 10

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

df["log_ewma_vol"] = np.log(df["ewma_vol"])

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

# Add forward volatility
df["log_fwd_vol"] = df["log_ewma_vol"].shift(-1)

df = df[
    ["log_ewma_vol", "trend", "log_fwd_vol"]
].dropna()

print(df.tail())

# ============================================================
# Regression 1: Forward log volatility on trend
# ============================================================

X1 = sm.add_constant(df["trend"])
y = df["log_fwd_vol"]

model1 = sm.OLS(y, X1).fit()

print("\nRegression 1: Trend versus forward log volatility")
print(model1.summary())

# Generate fitted line
x1_plot = np.linspace(
    df["trend"].min(),
    df["trend"].max(),
    200,
)

y1_plot = (
    model1.params["const"]
    + model1.params["trend"] * x1_plot
)

# Display first scatter plot
plt.figure(figsize=(9, 5))

plt.scatter(
    df["trend"],
    df["log_fwd_vol"],
    alpha=0.5,
    s=25,
)

plt.plot(
    x1_plot,
    y1_plot,
    color="red",
    linewidth=2,
    label="OLS fit",
)

plt.xlabel("Past 10-Month Average Return")
plt.ylabel("Forward Log Volatility")
plt.title("Trend and Forward Log Volatility")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Orthogonalize trend against current log volatility
# ============================================================

X_orth = sm.add_constant(df["log_ewma_vol"])

orthogonalization_model = sm.OLS(
    df["trend"],
    X_orth,
).fit()

# This is the component of trend unrelated to current volatility
df["orthogonal_trend"] = orthogonalization_model.resid


# ============================================================
# Regression 2: Forward log volatility on realized log volatility and
# orthogonalized trend
# ============================================================

X2 = sm.add_constant(
    df[["log_ewma_vol", "orthogonal_trend"]]
)

model2 = sm.OLS(
    df["log_fwd_vol"],
    X2,
).fit()

print(
    "\nRegression 2: Realized log volatility and "
    "orthogonalized trend versus forward log volatility"
)
print(model2.summary())

# Partial-regression y-axis: remove the effect of realized log volatility
# from forward log volatility. Since orthogonal_trend is already the
# residual from regressing trend on realized log volatility, its coefficient
# in the joint regression is also the slope in this partial plot.
fwd_vol_on_realized_vol = sm.OLS(
    df["log_fwd_vol"],
    sm.add_constant(df["log_ewma_vol"]),
).fit()

df["resid_fwd_log_vol"] = fwd_vol_on_realized_vol.resid

# Generate partial-regression fitted line
x2_plot = np.linspace(
    df["orthogonal_trend"].min(),
    df["orthogonal_trend"].max(),
    200,
)

y2_plot = (
    model2.params["orthogonal_trend"] * x2_plot
)

# Display partial-regression plot
plt.figure(figsize=(9, 5))

plt.scatter(
    df["orthogonal_trend"],
    df["resid_fwd_log_vol"],
    alpha=0.5,
    s=25,
)

plt.plot(
    x2_plot,
    y2_plot,
    color="red",
    linewidth=2,
    label="OLS fit",
)

plt.xlabel("Trend Orthogonal to Current Log Volatility")
plt.ylabel("Forward Log Volatility Residual")
plt.title("Partial Effect of Trend on Forward Log Volatility")
plt.legend()
plt.tight_layout()
plt.show()

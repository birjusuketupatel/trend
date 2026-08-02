import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

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

# Select first available observation each month
df = df.resample('MS').first()

# Add forward volatility estimate
df["fwd_vol"] = df["ewma_vol"].shift(-1)

df = df[["ewma_vol", "fwd_vol"]]

df = df.dropna()

print(df.head())

# Plot histogram of EWMA volatility
plt.figure(figsize=(9, 5))
plt.hist(df["ewma_vol"], bins=40, edgecolor="black", alpha=0.75)
plt.xlabel("EWMA Volatility")
plt.ylabel("Frequency")
plt.title("Distribution of Monthly EWMA Volatility Estimates")
plt.tight_layout()
plt.show()

# Log-transform current and forward volatility
df["log_ewma_vol"] = np.log(df["ewma_vol"])
df["log_fwd_vol"] = np.log(df["fwd_vol"])

# Regress forward log volatility on current log volatility
X = sm.add_constant(df["log_ewma_vol"])
y = df["log_fwd_vol"]

model = sm.OLS(y, X).fit()

print(model.summary())

# Scatter plot with OLS line of best fit
x_plot = np.linspace(
    df["log_ewma_vol"].min(),
    df["log_ewma_vol"].max(),
    200,
)
y_plot = model.params["const"] + model.params["log_ewma_vol"] * x_plot

plt.figure(figsize=(9, 5))
plt.scatter(
    df["log_ewma_vol"],
    df["log_fwd_vol"],
    alpha=0.5,
    s=25,
)
plt.plot(
    x_plot,
    y_plot,
    color="red",
    linewidth=2,
    label="OLS fit",
)
plt.xlabel("Log Current EWMA Volatility")
plt.ylabel("Log Forward EWMA Volatility")
plt.title("Current vs. Forward Log EWMA Volatility")
plt.legend()
plt.tight_layout()
plt.show()
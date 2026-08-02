import numpy as np
import pandas as pd
import statsmodels.api as sm

LAG_MONTHS = 10
DAYS_IN_MONTH = 21


# Load daily stock-market returns.
df = pd.read_csv(
    "data/us_stocks.csv",
    parse_dates=["Date"],
    index_col="Date",
).sort_index()

# Estimate current volatility from daily returns and take its logarithm.
df["ewma_vol"] = (
    df["stock_mkt"]
    .ewm(span=DAYS_IN_MONTH, adjust=True)
    .std()
)
df["log_ewma_vol"] = np.log(df["ewma_vol"])

# Compound daily returns into calendar-month returns.
month = df.index.to_period("M")
monthly_returns = (
    df["stock_mkt"]
    .add(1)
    .groupby(month)
    .prod(min_count=1)
    .sub(1)
)

# Store each completed month's return on the first trading day of the next
# month so that every signal uses only information available at that time.
first_trading_dates = df.groupby(month).head(1).index
df["ret_monthly"] = np.nan
df.loc[first_trading_dates, "ret_monthly"] = (
    monthly_returns.shift(1).to_numpy()
)

# Define trend as the average return over the previous ten months.
df.loc[first_trading_dates, "trend"] = (
    df.loc[first_trading_dates, "ret_monthly"]
    .rolling(window=LAG_MONTHS)
    .mean()
)

# Keep one observation per month and align predictors with next month's return.
df = df.resample("MS").first()
df["fwd_ret"] = df["ret_monthly"].shift(-1)

# Positive and zero trend values are trend on (+1); negative values are
# trend off (-1).
df["trend_dir"] = np.sign(df["trend"]) + (df["trend"] == 0)

df = df[["trend_dir", "fwd_ret", "log_ewma_vol"]].dropna()

# Rank forward returns into quintiles: 0 is the worst and 4 is the best.
df["ret_rank"] = pd.qcut(
    df["fwd_ret"],
    q=5,
    labels=range(5),
).astype(int)

# Binary outcomes for the bottom-quintile crash state and top-quintile rally
# state.
df["crash_state"] = (df["ret_rank"] == 0).astype(int)
df["rally_state"] = (df["ret_rank"] == 4).astype(int)

# Both logistic regressions use trend direction and current log volatility.
X = sm.add_constant(df[["trend_dir", "log_ewma_vol"]])

crash_model = sm.Logit(df["crash_state"], X).fit(disp=False)
print("\nLogistic Regression 1: Probability of a Crash State")
print(crash_model.summary())

rally_model = sm.Logit(df["rally_state"], X).fit(disp=False)
print("\nLogistic Regression 2: Probability of a Rally State")
print(rally_model.summary())

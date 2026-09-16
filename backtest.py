import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


DAYS_IN_MONTH = 21
LAG_MONTHS = 10
TREND_AVERAGE = "simple"  # "simple" or "linear"
TRADING_DAYS = 252
TARGET_VOL = 0.15
MAX_EXPOSURE = 1.00
VOL_TARGET_REBALANCING = "continuous"  # "coarse" or "continuous"
COARSE_REBALANCE_THRESHOLD = 0.1
ANNUAL_EXPENSE_RATIO = 0.004


def expanding_log_vol_forecast(log_annualized_volatility):
    """Predict 21-day-forward log annualized volatility with expanding OLS."""
    sampled_log_annualized_volatility = log_annualized_volatility.iloc[
        ::DAYS_IN_MONTH
    ]

    # Each regression pair is separated by 21 trading days. The pair ending
    # on a sampling date becomes available only on that date.
    current_log_annualized_volatility = (
        sampled_log_annualized_volatility.shift(1)
    )
    forward_log_annualized_volatility = sampled_log_annualized_volatility
    valid_pair = (
        current_log_annualized_volatility.notna()
        & forward_log_annualized_volatility.notna()
    )
    current_log_annualized_volatility = (
        current_log_annualized_volatility.where(valid_pair)
    )
    forward_log_annualized_volatility = (
        forward_log_annualized_volatility.where(valid_pair)
    )

    # These expanding moments give the intercept and slope from OLS while
    # avoiding a slow refit of the entire history on every date.
    mean_current_log_volatility = (
        current_log_annualized_volatility.expanding(min_periods=2).mean()
    )
    mean_forward_log_volatility = (
        forward_log_annualized_volatility.expanding(min_periods=2).mean()
    )
    variance_current_log_volatility = (
        current_log_annualized_volatility.expanding(min_periods=2).var()
    )
    covariance_log_volatility = current_log_annualized_volatility.expanding(
        min_periods=2
    ).cov(forward_log_annualized_volatility)

    volatility_persistence = (
        covariance_log_volatility / variance_current_log_volatility
    )
    log_volatility_intercept = (
        mean_forward_log_volatility
        - volatility_persistence * mean_current_log_volatility
    )

    # Carry the most recently estimable coefficients between sampling dates.
    daily_volatility_persistence = volatility_persistence.reindex(
        log_annualized_volatility.index
    ).ffill()
    daily_log_volatility_intercept = log_volatility_intercept.reindex(
        log_annualized_volatility.index
    ).ffill()
    return (
        daily_log_volatility_intercept
        + daily_volatility_persistence * log_annualized_volatility
    )


def equity_curve(returns):
    """Return the growth of one dollar."""
    return (1 + returns).cumprod()


def moving_average(values, window, method):
    """Return a simple or linearly weighted moving average."""
    if method == "simple":
        return values.rolling(window).mean()
    if method == "linear":
        weights = np.arange(1, window + 1, dtype=float)
        return values.rolling(window).apply(
            lambda observations: np.dot(observations, weights) / weights.sum(),
            raw=True,
        )
    raise ValueError(
        f"Unknown TREND_AVERAGE {method!r}; expected 'simple' or 'linear'."
    )


def volatility_target_exposure(
    uncapped_target,
    method,
    max_exposure,
    coarse_threshold,
):
    """Return capped exposure using daily or thresholded weekly rebalancing."""
    if method == "continuous":
        return uncapped_target.clip(lower=0, upper=max_exposure)
    if method != "coarse":
        raise ValueError(
            f"Unknown VOL_TARGET_REBALANCING {method!r}; expected "
            "'coarse' or 'continuous'."
        )

    # The first observation in each calendar week is the only day on which a
    # trade may occur. This also handles weeks whose Monday is a market holiday.
    week = uncapped_target.index.to_period("W-SUN")
    rebalance_day = pd.Series(week, index=uncapped_target.index).ne(
        pd.Series(week, index=uncapped_target.index).shift()
    )

    exposure = pd.Series(np.nan, index=uncapped_target.index, dtype=float)
    previous_equity_weight = np.nan
    for date, target in uncapped_target.items():
        if pd.isna(target) or not rebalance_day.loc[date] and pd.isna(
            previous_equity_weight
        ):
            continue
        if pd.isna(previous_equity_weight):
            selected_target = target
        elif rebalance_day.loc[date] and abs(
            target - previous_equity_weight
        ) > coarse_threshold and not np.isclose(
            abs(target - previous_equity_weight),
            coarse_threshold,
            rtol=0,
            atol=1e-12,
        ):
            selected_target = target
        else:
            selected_target = previous_equity_weight

        # Apply the exposure constraint only after the cadence and no-trade
        # threshold have selected the period's target weight.
        previous_equity_weight = np.clip(selected_target, 0, max_exposure)
        exposure.loc[date] = previous_equity_weight

    return exposure


def drawdown_depths(returns):
    """Return the trough depth of each distinct drawdown episode."""
    wealth = equity_curve(returns)
    running_peak = wealth.cummax().clip(lower=1.0)
    drawdown = wealth / running_peak - 1

    depths = []
    current_depth = None

    for value in drawdown:
        if value < 0:
            current_depth = value if current_depth is None else min(
                current_depth,
                value,
            )
        elif current_depth is not None:
            depths.append(current_depth)
            current_depth = None

    # Include a drawdown that has not recovered by the end of the sample.
    if current_depth is not None:
        depths.append(current_depth)

    return depths


def performance_statistics(returns, risk_free, exposure):
    """Calculate annualized return, risk, drawdown, and exposure statistics."""
    excess_returns = returns - risk_free
    log_returns = np.log1p(returns)
    depths = sorted(drawdown_depths(returns))
    five_worst = depths[:5]

    return {
        "Arithmetic mean": returns.mean() * TRADING_DAYS,
        "Arithmetic stddev": returns.std(ddof=1) * np.sqrt(TRADING_DAYS),
        "Arithmetic Sharpe": (
            excess_returns.mean()
            / excess_returns.std(ddof=1)
            * np.sqrt(TRADING_DAYS)
        ),
        "Geometric mean": np.expm1(log_returns.mean() * TRADING_DAYS),
        "Geometric stddev": np.expm1(
            log_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
        ),
        "Max drawdown": min(depths, default=0.0),
        "Average 5 worst drawdowns": np.mean(five_worst),
        "Average exposure": exposure.mean(),
        "Time in market": (exposure > 0).mean(),
        "Annual turnover": exposure.diff().abs().mean() * TRADING_DAYS,
    }


def print_statistics(statistics):
    """Print the statistics with percentages and Sharpe ratios formatted."""
    table = pd.DataFrame(statistics)
    formatted = table.copy().astype(object)

    for metric in table.index:
        if metric == "Arithmetic Sharpe":
            formatted.loc[metric] = table.loc[metric].map(lambda x: f"{x:.2f}")
        else:
            formatted.loc[metric] = table.loc[metric].map(lambda x: f"{x:.2%}")

    print("\nAnnualized Performance Statistics")
    print(formatted.to_string())


# Load total market and risk-free daily returns.
df = pd.read_csv(
    "data/us_stocks.csv",
    parse_dates=["Date"],
    index_col="Date",
).sort_index()

# Estimate daily volatility and train the predictive model in log-volatility
# space, where the persistence relationship is closer to linear.
df["ewma_vol"] = (
    df["stock_mkt"]
    .ewm(span=DAYS_IN_MONTH, adjust=True)
    .std()
)
df["annualized_ewma_vol"] = df["ewma_vol"] * np.sqrt(TRADING_DAYS)
df["log_annualized_ewma_vol"] = np.log(df["annualized_ewma_vol"])
df["predicted_log_annualized_vol_21_days_ahead"] = (
    expanding_log_vol_forecast(df["log_annualized_ewma_vol"])
)

# Compound daily returns into calendar-month returns. Each month's signal uses
# the configured average of the LAG_MONTHS fully completed months before it.
# For "linear", weights increase from 1 for the oldest month to LAG_MONTHS for
# the newest month.
month = df.index.to_period("M")
monthly_returns = (
    df["stock_mkt"]
    .add(1)
    .groupby(month)
    .prod(min_count=1)
    .sub(1)
)
trend_by_month = moving_average(
    monthly_returns.shift(1),
    LAG_MONTHS,
    TREND_AVERAGE,
)
df["trend"] = pd.Series(month, index=df.index).map(trend_by_month)
df["trend_gate"] = (df["trend"] >= 0).astype(float)

# The 21-day-ahead forecast made after date t is first usable for exposure on
# date t+1. Convert it from log volatility before sizing the position.
df["predicted_annual_vol_21_days_ahead"] = (
    np.exp(df["predicted_log_annualized_vol_21_days_ahead"].shift(1))
)

uncapped_target_exposure = (
    TARGET_VOL / df["predicted_annual_vol_21_days_ahead"]
) * df["trend_gate"]
df["exposure"] = volatility_target_exposure(
    uncapped_target_exposure,
    VOL_TARGET_REBALANCING,
    MAX_EXPOSURE,
    COARSE_REBALANCE_THRESHOLD,
)

# Begin only when both the monthly trend and prior-day volatility forecast are
# available. Cash earns RF, and exposure above one is financed at RF.
backtest = df[
    ["stock_mkt", "RF", "trend", "trend_gate", "exposure"]
].dropna().copy()

# Accrue the static annual expense ratio evenly across trading days for both
# active strategies, independent of their exposure changes.
daily_expense = ANNUAL_EXPENSE_RATIO / TRADING_DAYS
backtest["strategy_return"] = (
    backtest["RF"]
    + backtest["exposure"]
    * (backtest["stock_mkt"] - backtest["RF"])
    - daily_expense
)
backtest["trend_only_return"] = (
    backtest["RF"]
    + backtest["trend_gate"]
    * (backtest["stock_mkt"] - backtest["RF"])
    - daily_expense
)
backtest["buy_hold_return"] = backtest["stock_mkt"]

if (backtest["exposure"] < 0).any() or (
    backtest["exposure"] > MAX_EXPOSURE
).any():
    raise ValueError("Exposure is outside the permitted range.")

# Save daily returns, signals, and exposure for the backtested dates.
backtest.to_csv("data/returns.csv", index_label="Date")
print(f"wrote {len(backtest)} rows to 'data/returns.csv'")

# Print performance for the strategy and a fully invested benchmark.
statistics = {
    "Strategy": performance_statistics(
        backtest["strategy_return"],
        backtest["RF"],
        backtest["exposure"],
    ),
    "Trend Only": performance_statistics(
        backtest["trend_only_return"],
        backtest["RF"],
        backtest["trend_gate"],
    ),
    "Buy and Hold": performance_statistics(
        backtest["buy_hold_return"],
        backtest["RF"],
        pd.Series(1.0, index=backtest.index),
    ),
}
print_statistics(statistics)

# Plot growth of one dollar on a logarithmic scale so the full history remains
# readable.
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(
    backtest.index,
    equity_curve(backtest["strategy_return"]),
    color="blue",
    label="Strategy",
)
ax.plot(
    backtest.index,
    equity_curve(backtest["buy_hold_return"]),
    color="red",
    label="Buy and Hold",
)
ax.set_yscale("log")
ax.set_xlabel("Date")
ax.set_ylabel("Growth of $1 (log scale)")
ax.set_title("Strategy Equity Curve versus Buy and Hold")
ax.grid(alpha=0.2)
ax.legend(frameon=False)
fig.tight_layout()

# Plot the volatility-targeted strategy's daily equity exposure. The trend-only
# benchmark is intentionally omitted from this figure and the equity curve.
fig, ax = plt.subplots(figsize=(11, 5))
ax.fill_between(
    backtest.index,
    0,
    backtest["exposure"],
    color="steelblue",
    alpha=0.25,
    linewidth=0,
)
ax.plot(
    backtest.index,
    backtest["exposure"],
    color="steelblue",
    linewidth=0.5,
    alpha=0.65,
)
ax.axhline(
    1.0,
    color="dimgray",
    linewidth=0.7,
    linestyle="--",
    alpha=0.5,
)
ax.set_ylim(0, MAX_EXPOSURE * 1.05)
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_xlabel("Date")
ax.set_ylabel("Equity Exposure")
ax.set_title("Strategy Exposure Over Time")
ax.grid(alpha=0.2)
fig.tight_layout()

# Compound daily returns within each calendar month, then plot both monthly
# return distributions with identical bins.
monthly_plot_returns = (
    backtest[["strategy_return", "buy_hold_return"]]
    .add(1)
    .groupby(backtest.index.to_period("M"))
    .prod()
    .sub(1)
)

minimum_return = monthly_plot_returns[
    ["strategy_return", "buy_hold_return"]
].min().min()
maximum_return = monthly_plot_returns[
    ["strategy_return", "buy_hold_return"]
].max().max()
bins = np.linspace(minimum_return, maximum_return, 80)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(
    monthly_plot_returns["strategy_return"],
    bins=bins,
    density=True,
    alpha=0.5,
    color="blue",
    label="Strategy",
)
ax.hist(
    monthly_plot_returns["buy_hold_return"],
    bins=bins,
    density=True,
    alpha=0.5,
    color="red",
    label="Buy and Hold",
)
ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.set_xlabel("Monthly Return")
ax.set_ylabel("Probability Density")
ax.set_title("Distribution of Monthly Returns")
ax.legend(frameon=False)
fig.tight_layout()

plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

START = "1990-01-01"
END = "2026-12-31"
RANDOM_SEED = 42

df = pd.read_csv(
    "data/us_stocks.csv",
    parse_dates=["Date"],
    index_col="Date",
).sort_index()

returns = (
    df.loc[START:END, "stock_mkt"]
    .dropna()
    .astype(float)
)

if returns.empty:
    raise ValueError(f"No observations found between {START} and {END}.")

# Estimate the unconditional daily distribution.
mean_return = returns.mean()
standard_deviation = returns.std(ddof=1)

# Simulate independent returns with constant mean and variance.
rng = np.random.default_rng(RANDOM_SEED)

simulated_returns = pd.Series(
    rng.normal(
        loc=mean_return,
        scale=standard_deviation,
        size=len(returns),
    ),
    index=returns.index,
    name="simulated_return",
)

# Give both figures identical y-axis limits.
maximum_absolute_return = max(
    returns.abs().max(),
    simulated_returns.abs().max(),
)

y_limit = maximum_absolute_return * 1.05


def plot_daily_returns(series, title, color):
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.vlines(
        x=series.index,
        ymin=0,
        ymax=series,
        color=color,
        linewidth=0.4,
        alpha=0.8,
    )

    ax.axhline(
        y=0,
        color="black",
        linewidth=0.7,
    )

    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily return")
    ax.set_ylim(-y_limit, y_limit)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()


# Figure 1
plot_daily_returns(
    returns,
    f"Actual Daily U.S. Stock Market Returns, {START[:4]}–{END[:4]}",
    color="#2457A7",
)

# Figure 2
plot_daily_returns(
    simulated_returns,
    f"Simulated Daily Returns with Constant Volatility, {START[:4]}–{END[:4]}",
    color="#A7472A",
)

print(f"Historical daily mean: {mean_return:.4%}")
print(f"Historical daily standard deviation: {standard_deviation:.4%}")
# Calculate the 21-day EWMA standard deviation.
ewma_volatility = (
    returns
    .ewm(
        span=21,
        adjust=False,
        min_periods=21,
    )
    .std(bias=False)
)

# Figure 3: daily returns and EWMA volatility.
fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(14, 8),
    sharex=True,
)

# Top panel: daily returns.
axes[0].vlines(
    x=returns.index,
    ymin=0,
    ymax=returns,
    color="#2457A7",
    linewidth=0.4,
    alpha=0.8,
)

axes[0].axhline(
    y=0,
    color="black",
    linewidth=0.7,
)

axes[0].set_title(
    "Daily U.S. Stock Market Returns",
    loc="left",
    fontsize=13,
    fontweight="bold",
)

axes[0].set_ylabel("Daily return")
axes[0].set_ylim(-y_limit, y_limit)
axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
axes[0].grid(axis="y", alpha=0.2)

# Bottom panel: 21-day EWMA volatility.
axes[1].plot(
    ewma_volatility.index,
    ewma_volatility,
    color="#A7472A",
    linewidth=0.9,
)

axes[1].set_title(
    "21-Day EWMA Volatility",
    loc="left",
    fontsize=13,
    fontweight="bold",
)

axes[1].set_xlabel("Date")
axes[1].set_ylabel("Daily standard deviation")
axes[1].set_ylim(bottom=0)
axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
axes[1].grid(axis="y", alpha=0.2)

fig.suptitle(
    f"Daily Returns and Estimated Volatility, {START[:4]}–{END[:4]}",
    fontsize=15,
    fontweight="bold",
)

fig.tight_layout()


# Displays both figures without saving them.
plt.show()
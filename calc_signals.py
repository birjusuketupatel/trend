import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
from pandas_datareader import famafrench
import statsmodels.api as sm
import yfinance as yf


DAYS_IN_MONTH = 21
LAG_MONTHS = 10
TRADING_DAYS = 252
TARGET_VOL = 0.15
MAX_EXPOSURE = 1.00
FAMA_FRENCH_DATASET = "F-F_Research_Data_Factors_daily"
SP500_TOTAL_RETURN_TICKER = "^SP500TR"


# pandas-datareader still defines this endpoint with HTTP, while the Kenneth
# French data library is available over HTTPS.
famafrench._URL = famafrench._URL.replace("http://", "https://")


def load_fama_french_returns():
    """Download the Fama-French U.S. market total return series."""
    raw = pdr.DataReader(
        FAMA_FRENCH_DATASET,
        "famafrench",
        start="1900-01-01",
    )[0]
    returns = (raw["Mkt-RF"] + raw["RF"]) / 100
    if isinstance(returns.index, pd.PeriodIndex):
        returns.index = returns.index.to_timestamp()
    else:
        returns.index = pd.to_datetime(returns.index)
    returns.name = "total_return"
    return returns.sort_index()


def load_sp500_total_returns(start_date):
    """Download S&P 500 total-return index returns after start_date."""
    prices = yf.download(
        SP500_TOTAL_RETURN_TICKER,
        start=start_date.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    if prices.empty:
        return pd.Series(dtype=float, name="total_return")

    close = prices["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    returns = close.pct_change(fill_method=None).dropna()
    returns.name = "total_return"
    return returns


def load_combined_returns():
    """Splice recent S&P 500 total returns onto Fama-French history."""
    fama_french = load_fama_french_returns()
    last_fama_french_date = fama_french.index.max()

    # Starting on the last Fama-French date supplies the prior close needed to
    # calculate the first genuinely new daily S&P 500 total-index return.
    recent = load_sp500_total_returns(last_fama_french_date)
    recent = recent.loc[recent.index > last_fama_french_date]
    combined = pd.concat([fama_french, recent]).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined, last_fama_french_date


def calculate_signals(returns):
    """Calculate the monthly trend signal and annualized EWMA volatility."""
    signals = pd.DataFrame({"total_return": returns})
    signals["ewma_vol"] = (
        signals["total_return"]
        .ewm(span=DAYS_IN_MONTH, adjust=True)
        .std()
        * np.sqrt(TRADING_DAYS)
    )

    month = signals.index.to_period("M")
    monthly_returns = (
        signals["total_return"]
        .add(1)
        .groupby(month)
        .prod(min_count=1)
        .sub(1)
    )
    trend_by_month = (
        monthly_returns.shift(1)
        .rolling(LAG_MONTHS)
        .mean()
    )
    signals["trend"] = pd.Series(month, index=signals.index).map(
        trend_by_month
    )
    return signals, trend_by_month


def forecast_volatility(annualized_ewma_vol):
    """Fit 21-day log annualized-volatility persistence and forecast forward."""
    log_annualized_volatility = np.log(annualized_ewma_vol)

    # Sample once per 21 trading days so adjacent regression observations do
    # not reuse the same daily volatility estimates.
    sampled_log_annualized_volatility = log_annualized_volatility.iloc[
        ::DAYS_IN_MONTH
    ]
    regression_data = pd.concat(
        {
            "current_log_annualized_volatility": (
                sampled_log_annualized_volatility
            ),
            "forward_log_annualized_volatility": (
                sampled_log_annualized_volatility.shift(-1)
            ),
        },
        axis=1,
    ).dropna()
    if len(regression_data) < 2:
        raise ValueError("Not enough observations to train the volatility model.")

    regression_design = pd.DataFrame(
        {
            "intercept": 1.0,
            "current_log_annualized_volatility": regression_data[
                "current_log_annualized_volatility"
            ],
        },
        index=regression_data.index,
    )
    volatility_model = sm.OLS(
        regression_data["forward_log_annualized_volatility"],
        regression_design,
    ).fit()
    log_volatility_intercept = volatility_model.params["intercept"]
    volatility_persistence = volatility_model.params[
        "current_log_annualized_volatility"
    ]
    forecast_annualized_volatility = np.exp(
        log_volatility_intercept
        + volatility_persistence * log_annualized_volatility.iloc[-1]
    )
    return forecast_annualized_volatility, volatility_model


def print_signal_report(
    signals,
    trend_by_month,
    forecast_vol,
    volatility_model,
    last_fama_french_date,
):
    """Print current signals and volatility-regression results."""
    latest_date = signals.index.max()
    current_month = latest_date.to_period("M")
    trend = trend_by_month.loc[current_month]
    ewma_vol = signals.loc[latest_date, "ewma_vol"]
    leverage = np.clip(TARGET_VOL / forecast_vol, 0, MAX_EXPOSURE)

    if trend > 0:
        trend_direction = "Positive"
    elif trend < 0:
        trend_direction = "Negative"
    else:
        trend_direction = "Neutral"

    print(f"Signal date: {latest_date:%Y-%m-%d}")
    print(f"Fama-French data through: {last_fama_french_date:%Y-%m-%d}")
    print(f"EWMA volatility estimate: {ewma_vol:.2%}")
    print(f"21-trading-day-ahead volatility forecast: {forecast_vol:.2%}")
    print(f"Recommended leverage: {leverage:.2f}x")
    print(
        f"Trend signal at start of {current_month.strftime('%B %Y')}: "
        f"{trend:.2%}"
    )
    print(f"Trend direction: {trend_direction}")
    print("\n21-trading-day log annualized-volatility regression:")
    print(volatility_model.summary())


def plot_recent_signals(signals):
    """Plot trend and EWMA volatility over the latest 12 months."""
    latest_date = signals.index.max()
    plot_start = latest_date - pd.DateOffset(months=12)
    recent = signals.loc[signals.index >= plot_start]

    fig, (trend_ax, vol_ax) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
    )
    trend_ax.step(
        recent.index,
        recent["trend"],
        where="post",
        color="navy",
        linewidth=1.5,
    )
    trend_ax.axhline(0, color="dimgray", linewidth=0.8, linestyle="--")
    trend_ax.yaxis.set_major_formatter(PercentFormatter(1))
    trend_ax.set_ylabel("Average monthly return")
    trend_ax.set_title(f"{LAG_MONTHS}-Month Trend Signal")
    trend_ax.grid(alpha=0.2)

    vol_ax.plot(
        recent.index,
        recent["ewma_vol"],
        color="darkred",
        linewidth=1.5,
    )
    vol_ax.axhline(
        TARGET_VOL,
        color="dimgray",
        linewidth=0.8,
        linestyle="--",
        label=f"{TARGET_VOL:.0%} target",
    )
    vol_ax.yaxis.set_major_formatter(PercentFormatter(1))
    vol_ax.set_ylabel("Annualized volatility")
    vol_ax.set_xlabel("Date")
    vol_ax.set_title(f"{DAYS_IN_MONTH}-Day EWMA Volatility")
    vol_ax.grid(alpha=0.2)
    vol_ax.legend(frameon=False)

    fig.suptitle("U.S. Equity Signals — Past 12 Months")
    fig.tight_layout()
    plt.show()


def main():
    returns, last_fama_french_date = load_combined_returns()
    signals, trend_by_month = calculate_signals(returns)
    forecast_vol, volatility_model = forecast_volatility(signals["ewma_vol"])
    print_signal_report(
        signals,
        trend_by_month,
        forecast_vol,
        volatility_model,
        last_fama_french_date,
    )
    plot_recent_signals(signals)


if __name__ == "__main__":
    main()

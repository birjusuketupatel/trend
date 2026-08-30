import pandas as pd
from pandas_datareader import data as pdr
from pandas_datareader import famafrench


# pandas-datareader still defines this endpoint with HTTP, while the Kenneth
# French data library is available over HTTPS.
famafrench._URL = famafrench._URL.replace("http://", "https://")

raw = pdr.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start="1900-01-01")

factors = raw[0]

factors = factors.rename(columns={"Mkt-RF": "stock_mkt"})

# Keep stock_mkt as the total market return used by the existing scripts.
factors["stock_mkt"] += factors["RF"]
factors = factors[["stock_mkt", "RF"]] / 100

factors.to_csv("data/us_stocks.csv")
print(f"wrote {len(factors)} rows to 'data/us_stocks.csv'")

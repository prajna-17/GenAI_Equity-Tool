# fetch_stock.py
import yfinance as yf
import pandas as pd

def get_stock_history(ticker: str, period: str = "7d", interval: str = "1d"):
    """
    Fetch recent stock history for ticker symbol (e.g., 'AAPL').
    Returns a pandas DataFrame with Date index and Close prices.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    # keep only Close
    df = hist[["Close"]].copy()
    df = df.reset_index()
    df.rename(columns={"Date": "date", "Close": "close"}, inplace=True)
    return df

if __name__ == "__main__":
    print(get_stock_history("AAPL"))

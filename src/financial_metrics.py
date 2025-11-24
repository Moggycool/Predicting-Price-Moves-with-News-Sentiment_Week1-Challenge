"""
financial_metrics.py — Class-based module for financial metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt


class FinancialMetrics:
    """ Class to compute financial metrics for stocks."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def compute_daily_returns(self) -> pd.DataFrame:
        """Compute daily returns for each stock symbol."""
        self.df["Daily_Return"] = self.df.groupby(
            "Symbol")["Close"].pct_change()
        return self.df[["Symbol", "Date", "Daily_Return"]]

    def compute_cumulative_returns(self) -> pd.DataFrame:
        """Compute cumulative returns for each stock symbol."""
        self.df["Daily_Return"] = self.df.groupby(
            "Symbol")["Close"].pct_change()
        self.df["Cumulative_Return"] = self.df.groupby(
            "Symbol")["Daily_Return"].cumsum()
        return self.df[["Symbol", "Date", "Cumulative_Return"]]

    def compute_volatility(self, window: int = 14) -> pd.DataFrame:
        """Compute rolling volatility for each stock symbol."""
        self.df["Daily_Return"] = self.df.groupby(
            "Symbol")["Close"].pct_change()
        self.df["Volatility"] = self.df.groupby("Symbol")["Daily_Return"].rolling(
            window).std().reset_index(level=0, drop=True)
        return self.df[["Symbol", "Date", "Volatility"]]

    def plot_daily_returns(self, df_returns: pd.DataFrame) -> plt.Figure:
        """Plot daily returns for each stock symbol."""
        fig, ax = plt.subplots(figsize=(10, 5))
        for symbol, group in df_returns.groupby("Symbol"):
            ax.plot(group["Date"], group["Daily_Return"], label=symbol)
        ax.set_title("Daily Returns")
        ax.legend()
        return fig

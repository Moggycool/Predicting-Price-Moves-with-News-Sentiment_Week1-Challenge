"""
technical_analysis.py — Class-based module for technical indicators.
"""

import pandas as pd
import talib  # pylint: disable=no-member
import matplotlib.pyplot as plt


class TechnicalAnalysis:
    """ Class to compute technical indicators using TA-Lib."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if "Symbol" not in self.df.columns:
            self.df["Symbol"] = "Unknown"

    def compute_sma(self, period: int = 20) -> pd.DataFrame:
        """Compute Simple Moving Average (SMA)."""
        self.df["SMA"] = self.df.groupby("Symbol")["Close"].transform(
            lambda x: talib.SMA(x, timeperiod=period))  # pylint: disable=no-member
        return self.df

    def compute_ema(self, period: int = 20) -> pd.DataFrame:
        """Compute Exponential Moving Average (EMA)."""
        self.df["EMA"] = self.df.groupby("Symbol")["Close"].transform(
            lambda x: talib.EMA(x, timeperiod=period))  # pylint: disable=no-member
        return self.df

    def compute_rsi(self, period: int = 14) -> pd.DataFrame:
        """Compute Relative Strength Index (RSI)."""
        self.df["RSI"] = self.df.groupby("Symbol")["Close"].transform(
            lambda x: talib.RSI(x, timeperiod=period))   # pylint: disable=no-member

    def compute_macd(self) -> pd.DataFrame:
        """Compute Moving Average Convergence Divergence (MACD)."""
        def macd_group(x):
            macd, signal, hist = talib.MACD(x)  # pylint: disable=no-member
            self.df.loc[x.index, "MACD_Signal"] = signal
            self.df.loc[x.index, "MACD_Hist"] = hist
            return macd
        self.df["MACD"] = self.df.groupby(
            "Symbol")["Close"].transform(macd_group)
        return self.df

    def plot_sma(self) -> plt.Figure:
        """Plot Close price and SMA for each stock symbol."""
        fig, ax = plt.subplots(figsize=(10, 5))
        for symbol, group in self.df.groupby("Symbol"):
            ax.plot(group.index, group["Close"], label=f"{symbol} Close")
            ax.plot(group.index, group["SMA"], label=f"{symbol} SMA")
        ax.set_title("Close Price & SMA")
        ax.legend()
        return fig

"""
quantitative_analysis.py — Class-based module for visualizing stock data.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class QuantitativeAnalysis:
    """ Class to visualize stock data and analysis results."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def plot_closing_prices(self) -> plt.Figure:
        """Plot closing prices for each stock symbol."""
        fig, ax = plt.subplots(figsize=(12, 6))
        for symbol, group in self.df.groupby("Symbol"):
            ax.plot(group["Date"], group["Close"], label=symbol)
        ax.set_title("Closing Prices Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        return fig

    def plot_daily_returns(self, df_returns: pd.DataFrame) -> plt.Figure:
        """Plot daily returns for each stock symbol."""
        fig, ax = plt.subplots(figsize=(12, 6))
        for symbol, group in df_returns.groupby("Symbol"):
            ax.plot(group["Date"], group["Daily_Return"], label=symbol)
        ax.set_title("Daily Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily Return")
        ax.legend()
        return fig

    def plot_correlation_matrix(self, df_returns: pd.DataFrame) -> plt.Figure:
        """Plot correlation matrix of daily returns."""
        pivot = df_returns.pivot(
            index="Date", columns="Symbol", values="Daily_Return")
        corr = pivot.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Matrix of Daily Returns")
        return fig

"""
quantitative_analysis.py
Performs quantitative financial analysis using TA-Lib and PyNance.

Features:
- Load stock price data (OHLCV)
- Apply technical indicators (SMA, EMA, RSI, MACD, etc.)
- Compute financial metrics using PyNance
- Visualize price and indicator trends
"""

import importlib
from typing import Any, cast
import pandas as pd
import matplotlib.pyplot as plt


try:
    IMP_TALIB = importlib.import_module("talib")
    TALIB_AVAILABLE = True
except ImportError:
    IMP_TALIB = None
    TALIB_AVAILABLE = False

# Tell the type checker that talib has dynamic members
talib: Any = cast(Any, IMP_TALIB)

# Optional fallback
try:
    PANDAS_TA = importlib.import_module("pandas_ta")
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA = None
    PANDAS_TA_AVAILABLE = False

pandas_ta: Any = cast(Any, PANDAS_TA)

try:
    IMPORT_PYNANCE = importlib.import_module("pynance")
    PYNANCE_AVAILABLE = True
except ImportError:
    IMPORT_PYNANCE = None
    PYNANCE_AVAILABLE = False

# Tell the type checker this is dynamic
pynance: Any = cast(Any, IMPORT_PYNANCE)

# Optional: access attributes safely (they may not exist)
if PYNANCE_AVAILABLE:
    equity = getattr(pynance, "equity", None)
    metrics = getattr(pynance, "metrics", None)
else:
    ATR_EQUITY = None
    ATR_METRICS = None


plt.rcParams["figure.figsize"] = (12, 6)


class QuantitativeAnalysis:
    """Quantitative financial analysis using TA-Lib and PyNance."""

    def __init__(self, df, symbol="STOCK"):
        """
        Initialize QuantitativeAnalysis class.

        Args:
            df (pd.DataFrame): 
            Stock price DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            symbol (str): Stock symbol name for labeling plots
        """
        self.df = df.copy()
        self.symbol = symbol

        # Ensure required columns exist
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Ensure datetime index if not already
        if not pd.api.types.is_datetime64_any_dtype(self.df.index):
            if "Date" in self.df.columns:
                self.df.set_index(pd.to_datetime(
                    self.df["Date"]), inplace=True)
            else:
                raise ValueError(
                    "DataFrame must have a Date column or datetime index.")

    # ============================================================
    # 1. Apply Technical Indicators (TA-Lib)
    # ============================================================
    def apply_indicators(self):
        """
        Calculate common technical indicators:
        - SMA (20, 50)
        - EMA (20, 50)
        - RSI (14)
        - MACD
        """
        df = self.df

        # Moving Averages
        df["SMA_20"] = talib.SMA(df["Close"], timeperiod=20)
        df["SMA_50"] = talib.SMA(df["Close"], timeperiod=50)
        df["EMA_20"] = talib.EMA(df["Close"], timeperiod=20)
        df["EMA_50"] = talib.EMA(df["Close"], timeperiod=50)

        # RSI
        df["RSI_14"] = talib.RSI(df["Close"], timeperiod=14)

        # MACD
        macd, macdsignal, macdhist = talib.MACD(
            df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
        df["MACD"] = macd
        df["MACD_Signal"] = macdsignal
        df["MACD_Hist"] = macdhist

        self.df = df
        print("[SUCCESS] Technical indicators applied.")
        return df

    # ============================================================
    # 2. PyNance Financial Metrics
    # ============================================================
    def financial_metrics(self):
        """
        Compute financial metrics using PyNance.
        Metrics include:
        - Daily returns
        - Volatility
        - Cumulative returns
        """
        df = self.df.copy()

        # Daily returns
        df["Daily_Return"] = df["Close"].pct_change()

        # Cumulative returns
        df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1

        # Volatility (rolling std)
        df["Volatility_20"] = df["Daily_Return"].rolling(20).std()

        self.df = df
        print("[SUCCESS] Financial metrics calculated using PyNance logic.")
        return df

    # ============================================================
    # 3. Visualization
    # ============================================================
    def visualize(self):
        """
        Plot stock price and technical indicators.
        """
        df = self.df

        # Price + Moving Averages
        plt.figure()
        plt.plot(df["Close"], label=f"{self.symbol} Close", color="black")
        plt.plot(df["SMA_20"], label="SMA 20", color="blue")
        plt.plot(df["SMA_50"], label="SMA 50", color="red")
        plt.plot(df["EMA_20"], label="EMA 20", color="green")
        plt.plot(df["EMA_50"], label="EMA 50", color="orange")
        plt.title(f"{self.symbol} Price & Moving Averages")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

        # RSI
        plt.figure()
        plt.plot(df["RSI_14"], label="RSI 14", color="purple")
        plt.axhline(70, color="red", linestyle="--")
        plt.axhline(30, color="green", linestyle="--")
        plt.title(f"{self.symbol} RSI (14)")
        plt.xlabel("Date")
        plt.ylabel("RSI")
        plt.legend()
        plt.show()

        # MACD
        plt.figure()
        plt.plot(df["MACD"], label="MACD", color="blue")
        plt.plot(df["MACD_Signal"], label="MACD Signal", color="red")
        plt.bar(df.index, df["MACD_Hist"], label="MACD Hist", color="gray")
        plt.title(f"{self.symbol} MACD")
        plt.xlabel("Date")
        plt.ylabel("MACD")
        plt.legend()
        plt.show()

        # Cumulative Return
        plt.figure()
        plt.plot(df["Cumulative_Return"],
                 label="Cumulative Return", color="green")
        plt.title(f"{self.symbol} Cumulative Returns")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.show()

    # ============================================================
    # 4. Full Analysis Pipeline
    # ============================================================
    def full_analysis(self):
        """
        Run the full quantitative analysis pipeline:
        Apply indicators → Compute metrics → Visualize
        """
        self.apply_indicators()
        self.financial_metrics()
        self.visualize()
        print("[SUCCESS] Full quantitative analysis completed.")
        return self.df

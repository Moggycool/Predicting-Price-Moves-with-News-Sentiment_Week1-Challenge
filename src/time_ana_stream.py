"""
time_ana_stream.py — Streamlit-ready Time Series Analysis for news datasets
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


class TimeSeriesAnalyzer:
    """
    Performs time series analysis on news publication data.

    Features:
    - Daily publication trends
    - Spike detection
    - Hour-of-day publication analysis
    - Weekday-hour heatmap
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analyzer with a dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed news dataset with a 'date' column
        """
        self.df = df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        self.df = self.df.dropna(subset=["date"])
        self.daily_counts = None
        self.hourly_counts = None

    # -------------------------
    def compute_daily_counts(self) -> pd.Series:
        """Compute daily article counts."""
        self.daily_counts = self.df.groupby(self.df["date"].dt.date).size()
        return self.daily_counts

    # -------------------------
    def detect_spikes(self, threshold_std: float = 2) -> pd.Series:
        """Detect days with unusually high publication volume."""
        if self.daily_counts is None:
            self.compute_daily_counts()
        mean_count = self.daily_counts.mean()
        std_count = self.daily_counts.std()
        spike_threshold = mean_count + threshold_std * std_count
        spikes = self.daily_counts[self.daily_counts > spike_threshold]
        return spikes

    # -------------------------
    def plot_daily_trend(
        self,
        spikes: pd.Series | None = None,
        return_fig: bool = False
    ) -> plt.Figure | None:
        """Plot daily publication trend with optional spikes highlighted."""
        if self.daily_counts is None:
            self.compute_daily_counts()

        fig, ax = plt.subplots(figsize=(14, 5))
        self.daily_counts.plot(ax=ax, label="Daily Count")
        if spikes is not None and not spikes.empty:
            ax.scatter(spikes.index, spikes.values,
                       color="red", label="Spikes", zorder=5)
        ax.set_title("Daily News Publication Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Articles")
        ax.legend()
        plt.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()

    # -------------------------
    def compute_hourly_distribution(self) -> pd.Series:
        """Compute number of articles published by hour of the day."""
        self.df["hour"] = self.df["date"].dt.hour
        self.hourly_counts = self.df.groupby("hour").size()
        return self.hourly_counts

    # -------------------------
    def plot_hourly_distribution(self, return_fig: bool = False) -> plt.Figure | None:
        """Plot articles published by hour of the day."""
        if self.hourly_counts is None:
            self.compute_hourly_distribution()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=self.hourly_counts.index,
                    y=self.hourly_counts.values, palette="viridis", ax=ax)
        ax.set_title("Articles Published by Hour of Day")
        ax.set_xlabel("Hour (0-23)")
        ax.set_ylabel("Number of Articles")
        plt.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()

    # -------------------------
    def plot_weekday_hour_heatmap(self, return_fig: bool = False) -> plt.Figure | None:
        """Heatmap of article frequency by hour vs weekday."""
        self.df["hour"] = self.df["date"].dt.hour
        self.df["weekday"] = self.df["date"].dt.day_name()

        pivot = self.df.pivot_table(
            index="hour",
            columns="weekday",
            values="headline",
            aggfunc="count",
            fill_value=0
        )

        # Reorder columns to weekday order
        weekday_order = ["Monday", "Tuesday", "Wednesday",
                         "Thursday", "Friday", "Saturday", "Sunday"]
        pivot = pivot[weekday_order]

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt="d", ax=ax)
        ax.set_title("Heatmap: Hour of Day vs Weekday Publication Frequency")
        ax.set_xlabel("Weekday")
        ax.set_ylabel("Hour of Day")
        plt.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()

    # -------------------------
    def run(self) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Run full time series analysis pipeline."""
        daily_counts = self.compute_daily_counts()
        spikes = self.detect_spikes()
        hourly_counts = self.compute_hourly_distribution()
        return daily_counts, spikes, hourly_counts

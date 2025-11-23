"""
time_analysis.py — Class-based Time Series Analysis for news datasets.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class TimeSeriesAnalyzer:
    """
    Performs time series analysis on news publication data.

    Features:
    - Daily/weekly/monthly publication trends
    - Spike detection
    - Hour-of-day publication analysis
    """

    def __init__(self, data_path: str):
        """
        Initialize the analyzer with path to preprocessed CSV data.

        Parameters
        ----------
        data_path : str
            Path to 'preprocessed_data.csv'
        """
        self.data_path = data_path
        self.df = None
        self.daily_counts = None
        self.hourly_counts = None

    # ---------------------------------------------------------------
    def load_data(self):
        """Load preprocessed CSV and parse dates."""
        self.df = pd.read_csv(self.data_path)
        if "date" not in self.df.columns:
            raise ValueError("Dataset must contain a 'date' column!")

        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        self.df = self.df.dropna(subset=["date"])

    # ---------------------------------------------------------------
    def compute_daily_counts(self):
        """Compute daily article counts."""
        self.daily_counts = self.df.groupby(self.df["date"].dt.date).size()
        return self.daily_counts

    # ---------------------------------------------------------------
    def detect_spikes(self, threshold_std=2):
        """
        Detect days with unusually high publication volume.

        Parameters
        ----------
        threshold_std : float
            Number of standard deviations above mean to consider a spike.

        Returns
        -------
        pd.Series
            Days considered spikes with their article counts.
        """
        mean_count = self.daily_counts.mean()
        std_count = self.daily_counts.std()
        spike_threshold = mean_count + threshold_std * std_count
        spikes = self.daily_counts[self.daily_counts > spike_threshold]
        return spikes

    # ---------------------------------------------------------------
    def plot_daily_trend(self, spikes=None):
        """Plot daily publication trend with optional spikes highlighted."""
        plt.figure(figsize=(14, 5))
        self.daily_counts.plot(label="Daily Count")
        if spikes is not None:
            plt.scatter(spikes.index, spikes.values,
                        color="red", label="Spikes", zorder=5)
        plt.title("Daily News Publication Trend")
        plt.xlabel("Date")
        plt.ylabel("Number of Articles")
        plt.legend()
        plt.show()

    # ---------------------------------------------------------------
    def compute_hourly_distribution(self):
        """Compute number of articles published by hour of the day."""
        self.df["hour"] = self.df["date"].dt.hour
        self.hourly_counts = self.df.groupby("hour").size()
        return self.hourly_counts

    # ---------------------------------------------------------------
    def plot_hourly_distribution(self):
        """Plot articles published by hour of the day."""
        plt.figure(figsize=(10, 5))
        sns.barplot(x=self.hourly_counts.index,
                    y=self.hourly_counts.values, palette="viridis")
        plt.title("Articles Published by Hour of Day")
        plt.xlabel("Hour (0-23)")
        plt.ylabel("Number of Articles")
        plt.show()

    # ---------------------------------------------------------------
    def run(self):
        """Run full time series analysis pipeline."""
        print("Loading data...")
        self.load_data()

        print("Computing daily publication counts...")
        self.compute_daily_counts()

        print("Detecting publication spikes...")
        spikes = self.detect_spikes()
        print("Spikes detected on these dates:\n", spikes)

        print("Plotting daily trend...")
        self.plot_daily_trend(spikes=spikes)

        print("Computing hourly publication distribution...")
        self.compute_hourly_distribution()

        print("Plotting hourly distribution...")
        self.plot_hourly_distribution()

        return self.daily_counts, spikes, self.hourly_counts
    # ---------------------------------------------------------------

    def plot_weekday_hour_heatmap(self):
        """
        Plot a heatmap of article frequency by hour of the day vs weekday.
        """
        # Ensure weekday and hour columns exist
        self.df["hour"] = self.df["date"].dt.hour
        self.df["weekday"] = self.df["date"].dt.day_name()

        # Pivot table: rows=hour, columns=weekday, values=count of articles
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

        # Plot heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt="d")
        plt.title("Heatmap: Hour of Day vs Weekday Publication Frequency")
        plt.xlabel("Weekday")
        plt.ylabel("Hour of Day")
        plt.show()

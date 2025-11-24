"""Dashboard module for time series analysis of news datasets."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


class TimeSeriesAnalyzer:
    """Performs time series analysis on news datasets."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.daily_counts = None
        self.hourly_counts = None

    def compute_daily_counts(self):
        """Compute daily counts of news articles."""
        self.daily_counts = self.df.groupby(self.df["date"].dt.date).size()
        return self.daily_counts

    def detect_spikes(self, threshold_std=2):
        """Detect spikes in daily counts based on standard deviation threshold."""
        mean_count = self.daily_counts.mean()
        std_count = self.daily_counts.std()
        threshold = mean_count + threshold_std * std_count
        spikes = self.daily_counts[self.daily_counts > threshold]
        return spikes

    def plot_daily_trend(self, spikes=None, return_fig=False):
        """Plot daily news publication trend."""
        fig, ax = plt.subplots(figsize=(12, 5))
        self.daily_counts.plot(ax=ax, label="Daily Count")
        if spikes is not None:
            ax.scatter(spikes.index, spikes.values,
                       color="red", label="Spikes")
        ax.set_title("Daily News Publication Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Articles")
        ax.legend()
        if return_fig:
            return fig
        plt.show()

    def compute_hourly_distribution(self):
        """Compute hourly distribution of articles."""
        self.df["hour"] = self.df["date"].dt.hour
        self.hourly_counts = self.df.groupby("hour").size()
        return self.hourly_counts

    def plot_hourly_distribution(self, return_fig=False):
        """Plot hourly distribution of articles."""
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(self.hourly_counts.index,
               self.hourly_counts.values, color='skyblue')
        ax.set_title("Articles Published by Hour of Day")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Number of Articles")
        if return_fig:
            return fig
        plt.show()

    def plot_weekday_hour_heatmap(self, return_fig=False):
        """Plot heatmap of articles by weekday and hour."""
        self.df["hour"] = self.df["date"].dt.hour
        self.df["weekday"] = self.df["date"].dt.day_name()
        pivot = self.df.pivot_table(
            index="hour", columns="weekday", values="headline", aggfunc="count", fill_value=0
        )
        weekday_order = ["Monday", "Tuesday", "Wednesday",
                         "Thursday", "Friday", "Saturday", "Sunday"]
        pivot = pivot[weekday_order]
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt="d", ax=ax)
        ax.set_title("Heatmap: Hour of Day vs Weekday Publication Frequency")
        ax.set_xlabel("Weekday")
        ax.set_ylabel("Hour")
        if return_fig:
            return fig
        plt.show()

    def run(self):
        """Execute the full time series analysis pipeline."""
        print("Running time series analysis...")
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        self.df = self.df.dropna(subset=["date"])
        self.compute_daily_counts()
        spikes = self.detect_spikes()
        self.compute_hourly_distribution()
        return self.daily_counts, spikes, self.hourly_counts

"""
time_analysis.py
Performs time-based analysis on preprocessed analyst ratings data.

Features:
1. Publication frequency over time (daily/weekly/monthly)
2. Analysis of publishing times (hour of day)
"""

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 5)


class TimeAnalysis:
    """Time-based analysis on preprocessed data."""

    def __init__(self, df, date_column="date"):
        """
        Initialize TimeAnalysis class.

        Args:
            df (pd.DataFrame): Preprocessed dataframe from DataLoader
            date_column (str): Column containing datetime values
        """
        self.df = df.copy()
        self.date_column = date_column

        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(self.df[date_column]):
            self.df[date_column] = pd.to_datetime(
                self.df[date_column], errors="coerce")

        # Drop invalid dates
        self.df.dropna(subset=[date_column], inplace=True)

    # ============================================================
    # 1. Publication Frequency Over Time
    # ============================================================
    def publication_frequency(self):
        """
        Analyze article counts over time to identify spikes.

        Returns:
            dict: daily, weekly, monthly publication counts
        """
        df = self.df.set_index(self.date_column)

        daily = df.resample("D").size()
        weekly = df.resample("W").size()
        monthly = df.resample("M").size()

        # Plot daily trend
        plt.figure()
        daily.plot()
        plt.title("Daily Article Publication Frequency")
        plt.xlabel("Date")
        plt.ylabel("Number of Articles")
        plt.show()

        # Plot weekly trend
        plt.figure()
        weekly.plot(marker='o')
        plt.title("Weekly Article Publication Frequency")
        plt.xlabel("Week")
        plt.ylabel("Number of Articles")
        plt.show()

        # Plot monthly trend
        plt.figure()
        monthly.plot(marker='o')
        plt.title("Monthly Article Publication Frequency")
        plt.xlabel("Month")
        plt.ylabel("Number of Articles")
        plt.show()

        return {
            "daily": daily,
            "weekly": weekly,
            "monthly": monthly
        }

    # ============================================================
    # 2. Publishing Time Analysis
    # ============================================================
    def publishing_time_analysis(self):
        """
        Analyze publishing times (hour of day, weekday trends)

        Returns:
            dict: hourly and weekday distribution
        """
        df = self.df.copy()
        df["hour"] = df[self.date_column].dt.hour
        df["weekday"] = df[self.date_column].dt.day_name()

        # Hourly distribution
        hourly_counts = df["hour"].value_counts().sort_index()
        plt.figure()
        hourly_counts.plot(kind="bar")
        plt.title("Article Publications by Hour of Day")
        plt.xlabel("Hour")
        plt.ylabel("Number of Articles")
        plt.show()

        # Weekday distribution
        weekday_order = ["Monday", "Tuesday", "Wednesday",
                         "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_counts = df["weekday"].value_counts().reindex(weekday_order)
        plt.figure()
        weekday_counts.plot(kind="bar", color='orange')
        plt.title("Article Publications by Weekday")
        plt.xlabel("Weekday")
        plt.ylabel("Number of Articles")
        plt.show()

        return {
            "hourly": hourly_counts,
            "weekday": weekday_counts
        }

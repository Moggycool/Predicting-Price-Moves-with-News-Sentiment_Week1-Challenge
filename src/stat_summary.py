"""
stat_summary.py
Performs descriptive statistical analysis on the preprocessed analyst ratings dataset.

Functions:
1. Text length statistics (headline length)
2. Articles per publisher
3. Date-based trend analysis
"""

import pandas as pd


class StatSummary:
    """
    A reusable analysis class for summary statistics and simple EDA.
    Works with the cleaned DataFrame output from DataLoader.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with clean, preprocessed data.

        Args:
            df (pd.DataFrame): Preprocessed DataFrame from DataLoader.clean_preprocess()
        """
        self.df = df.copy()

        # Ensure "date" is datetime for trend analysis
        if not pd.api.types.is_datetime64_any_dtype(self.df["date"]):
            self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")

    # ----------------------------------------------------------------------
    # 1. Headline Text Length Statistics
    # ----------------------------------------------------------------------
    def headline_length_stats(self) -> pd.Series:
        """
        Get descriptive statistics for headline lengths.
        Returns:
            pd.Series: count, mean, std, min, max, etc.
        """
        self.df["headline_length"] = self.df["headline"].astype(str).str.len()
        headline_stats = self.df["headline_length"].describe()

        print("\n--- Headline Length Statistics ---")
        print(headline_stats)

        return headline_stats

    # ----------------------------------------------------------------------
    # 2. Articles Per Publisher
    # ----------------------------------------------------------------------
    def publisher_article_counts(self) -> pd.Series:
        """
        Count number of articles per publisher.

        Returns:
            pd.Series: Publisher counts sorted descending.
        """
        counts = self.df["publisher"].value_counts()

        print("\n--- Articles per Publisher ---")
        print(counts)

        return counts

    # ----------------------------------------------------------------------
    # 3. Date Trend Analysis (Daily / Weekly / Monthly)
    # ----------------------------------------------------------------------
    def publication_trends(self) -> dict:
        """
        Analyze publication frequency over time.

        Returns:
            dict: DataFrames for daily, weekly, and monthly trends.
        """
        # Ensure proper datetime index for resampling
        df = self.df.set_index("date")

        daily = df.resample("D").size()
        weekly = df.resample("W").size()
        monthly = df.resample("M").size()

        print("\n--- Publication Trends ---")
        print("Daily counts (first 10 rows):")
        print(daily.head(10))
        print("\nWeekly counts (first 10 rows):")
        print(weekly.head(10))
        print("\nMonthly counts (first 10 rows):")
        print(monthly.head(10))

        return {
            "daily": daily,
            "weekly": weekly,
            "monthly": monthly
        }


# --------------------------------------------------------------------------
# Example usage (run directly)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from load import DataLoader

    loader = DataLoader(
        r"D:\Python\Week-1\Data-Week-1\raw_analyst_ratings.csv"
    )

    # Load + clean data
    loader.load_data()
    df_clean = loader.clean_preprocess()

    # Perform stats
    stats = StatSummary(df_clean)

    stats.headline_length_stats()
    stats.publisher_article_counts()
    stats.publication_trends()

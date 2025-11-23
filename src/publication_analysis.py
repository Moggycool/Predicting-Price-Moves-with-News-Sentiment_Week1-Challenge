"""
publication_analysis.py — Class-based Publisher Analysis on news datasets
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid")


class PublisherAnalyzer:
    """
    Perform analysis of news publishers:
    - Count articles per publisher
    - Identify top contributors
    - Extract domains from email-like publisher names
    - Compare types of news per publisher
    """

    def __init__(self, data_path: str):
        """
        Initialize PublisherAnalyzer.

        :param data_path: Path to preprocessed CSV data
        """
        self.data_path = data_path
        self.df = None
        self.publisher_counts = None
        self.domain_counts = None

    # -------------------------
    def load_data(self):
        """Load the CSV file and basic validation."""
        self.df = pd.read_csv(self.data_path)
        if "publisher" not in self.df.columns:
            raise ValueError("Dataset must contain a 'publisher' column!")
        if "headline" not in self.df.columns:
            raise ValueError("Dataset must contain a 'headline' column!")

    # -------------------------
    def count_articles_per_publisher(self):
        """Count how many articles each publisher contributes."""
        self.publisher_counts = self.df["publisher"].value_counts()
        return self.publisher_counts

    # -------------------------
    def plot_top_publishers(self, top_n=10):
        """Plot top N publishers by article count."""
        if self.publisher_counts is None:
            self.count_articles_per_publisher()

        top_publishers = self.publisher_counts.head(top_n)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_publishers.values,
                    y=top_publishers.index, palette="viridis")
        plt.title(f"Top {top_n} Publishers by Number of Articles")
        plt.xlabel("Number of Articles")
        plt.ylabel("Publisher")
        plt.show()

    # -------------------------
    def extract_email_domains(self):
        """
        Extract domains if publisher names are email addresses.
        """
        def get_domain(publisher):
            if "@" in publisher:
                return publisher.split("@")[1].lower()
            return publisher.lower()

        self.df["publisher_domain"] = self.df["publisher"].astype(
            str).apply(get_domain)
        self.domain_counts = self.df["publisher_domain"].value_counts()
        return self.domain_counts

    # -------------------------
    def plot_top_domains(self, top_n=10):
        """Plot top N domains if emails are used as publisher names."""
        if self.domain_counts is None:
            self.extract_email_domains()

        top_domains = self.domain_counts.head(top_n)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_domains.values,
                    y=top_domains.index, palette="coolwarm")
        plt.title(f"Top {top_n} Publisher Domains")
        plt.xlabel("Number of Articles")
        plt.ylabel("Domain")
        plt.show()

    # -------------------------
    def analyze_news_types_per_publisher(self, column="stock"):
        """
        Analyze difference in type of news reported by publishers.
        Assumes there is a 'column' indicating news category (like 'stock' or 'news_type').
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataset!")

        pivot = self.df.pivot_table(
            index="publisher",
            columns=column,
            values="headline",
            aggfunc="count",
            fill_value=0
        )

        print("News type distribution per publisher:")
        print(pivot.head(10))  # show top 10 publishers
        return pivot

    # -------------------------
    def run_full_analysis(self, top_n=10, type_column="stock"):
        """
        Run the full analysis pipeline:
        - Publisher counts & plot
        - Domain extraction & plot
        - News type distribution
        """
        print("Loading data...")
        self.load_data()

        print("\nCounting articles per publisher...")
        self.count_articles_per_publisher()
        self.plot_top_publishers(top_n=top_n)

        print("\nExtracting domains from publishers...")
        self.extract_email_domains()
        self.plot_top_domains(top_n=top_n)

        print(
            f"\nAnalyzing news types per publisher using column '{type_column}'...")
        news_type_distribution = self.analyze_news_types_per_publisher(
            column=type_column)

        return self.publisher_counts, self.domain_counts, news_type_distribution

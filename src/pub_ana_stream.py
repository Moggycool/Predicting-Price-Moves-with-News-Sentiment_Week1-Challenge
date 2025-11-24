"""
pub_ana_stream.py — Publisher analysis module for news datasets (Streamlit-ready)
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

    def __init__(self, df: pd.DataFrame):
        """
        Initialize PublisherAnalyzer with a dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed news dataset
        """
        self.df = df.copy()
        self.publisher_counts = None
        self.domain_counts = None

    # -------------------------
    def count_articles_per_publisher(self) -> pd.Series:
        """Count how many articles each publisher contributes."""
        self.publisher_counts = self.df["publisher"].value_counts()
        return self.publisher_counts

    # -------------------------
    def plot_top_publishers(self, top_n: int = 10, return_fig: bool = False) -> plt.Figure | None:
        """Plot top N publishers by article count."""
        if self.publisher_counts is None:
            self.count_articles_per_publisher()

        top_publishers = self.publisher_counts.head(top_n)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_publishers.values,
                    y=top_publishers.index, palette="viridis", ax=ax)
        ax.set_title(f"Top {top_n} Publishers by Number of Articles")
        ax.set_xlabel("Number of Articles")
        ax.set_ylabel("Publisher")
        plt.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()

    # -------------------------
    def extract_email_domains(self) -> pd.Series:
        """Extract domains if publisher names are email addresses."""
        def get_domain(publisher: str) -> str:
            if "@" in publisher:
                return publisher.split("@")[1].lower()
            return publisher.lower()

        self.df["publisher_domain"] = self.df["publisher"].astype(
            str).apply(get_domain)
        self.domain_counts = self.df["publisher_domain"].value_counts()
        return self.domain_counts

    # -------------------------
    def plot_top_domains(self, top_n: int = 10, return_fig: bool = False) -> plt.Figure | None:
        """Plot top N domains if emails are used as publisher names."""
        if self.domain_counts is None:
            self.extract_email_domains()

        top_domains = self.domain_counts.head(top_n)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_domains.values, y=top_domains.index,
                    palette="coolwarm", ax=ax)
        ax.set_title(f"Top {top_n} Publisher Domains")
        ax.set_xlabel("Number of Articles")
        ax.set_ylabel("Domain")
        plt.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()

    # -------------------------
    def analyze_news_types_per_publisher(self, column: str = "stock") -> pd.DataFrame:
        """
        Analyze difference in type of news reported by publishers.

        Parameters
        ----------
        column : str
            Column indicating news category (e.g., 'stock' or 'news_type')

        Returns
        -------
        pd.DataFrame
            Pivot table of counts per publisher per news type
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
        return pivot

    # -------------------------
    def run_full_analysis(
        self,
        top_n: int = 10,
        type_column: str = "stock"
    ) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        Run the full analysis pipeline:
        - Publisher counts & plot
        - Domain extraction & plot
        - News type distribution
        """
        self.count_articles_per_publisher()
        # explicitly uses top_n
        self.plot_top_publishers(top_n=top_n, return_fig=True)
        self.extract_email_domains()
        self.plot_top_domains(top_n=top_n, return_fig=True)
        news_type_distribution = self.analyze_news_types_per_publisher(
            column=type_column)

        return self.publisher_counts, self.domain_counts, news_type_distribution

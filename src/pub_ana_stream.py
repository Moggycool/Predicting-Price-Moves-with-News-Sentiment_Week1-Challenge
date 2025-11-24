"""Publisher analysis module for news datasets."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


class PublisherAnalyzer:
    """Publisher analysis for news datasets."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.publisher_counts = None
        self.domain_counts = None

    def count_articles_per_publisher(self):
        """Count number of articles per publisher."""
        self.publisher_counts = self.df["publisher"].value_counts()
        return self.publisher_counts

    def plot_top_publishers(self, top_n=10, return_fig=False):
        """Plot top N publishers by article count."""
        if self.publisher_counts is None:
            self.count_articles_per_publisher()
        top = self.publisher_counts.head(top_n)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top.index, top.values, color='skyblue')
        ax.set_title(f"Top {top_n} Publishers")
        ax.set_xlabel("Number of Articles")
        ax.set_ylabel("Publisher")
        if return_fig:
            return fig
        plt.show()

    def extract_email_domains(self):
        """Extract email domains from publisher field."""
        def get_domain(pub):
            return pub.split("@")[1].lower() if "@" in pub else pub.lower()
        self.df["publisher_domain"] = self.df["publisher"].astype(
            str).apply(get_domain)
        self.domain_counts = self.df["publisher_domain"].value_counts()
        return self.domain_counts

    def plot_top_domains(self, top_n=10, return_fig=False):
        """Plot top N publisher domains by article count."""
        if self.domain_counts is None:
            self.extract_email_domains()
        top = self.domain_counts.head(top_n)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top.index, top.values, color='coral')
        ax.set_title(f"Top {top_n} Publisher Domains")
        ax.set_xlabel("Number of Articles")
        ax.set_ylabel("Domain")
        if return_fig:
            return fig
        plt.show()

    def analyze_news_types_per_publisher(self, column="stock"):
        """Analyze distribution of news types per publisher."""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found!")
        pivot = self.df.pivot_table(
            index="publisher", columns=column, values="headline", aggfunc="count", fill_value=0)
        return pivot

    def run_full_analysis(self, top_n=10, type_column="stock"):
        """Run full publisher analysis pipeline."""
        self.count_articles_per_publisher()
        self.extract_email_domains()
        news_type_dist = self.analyze_news_types_per_publisher(type_column)
        return self.publisher_counts, self.domain_counts, news_type_dist

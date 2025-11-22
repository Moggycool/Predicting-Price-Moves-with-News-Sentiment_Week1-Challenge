"""
publication_analysis.py
Performs publisher-level analysis on preprocessed analyst ratings data.

Features:
1. Identify most active publishers
2. Analyze differences in the type of news they report
3. Extract email domains if publisher names are emails
"""

import re
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 5)


class PublicationAnalysis:
    """Publisher-level analysis on preprocessed data."""

    def __init__(self, df, publisher_column="publisher", text_column="headline"):
        """
        Initialize PublicationAnalysis class.

        Args:
            df (pd.DataFrame): Preprocessed DataFrame from DataLoader
            publisher_column (str): Column containing publisher names
            text_column (str): Column containing headline or article text
        """
        self.df = df.copy()
        self.publisher_column = publisher_column
        self.text_column = text_column

    # ============================================================
    # 1. Most Active Publishers
    # ============================================================
    def most_active_publishers(self, top_n=10):
        """
        Count articles per publisher and visualize top contributors.

        Returns:
            pd.Series: top_n publisher counts
        """
        counts = self.df[self.publisher_column].value_counts()
        top_counts = counts.head(top_n)

        print(f"Top {top_n} Most Active Publishers:")
        print(top_counts)

        top_counts.plot(kind="bar")
        plt.title(f"Top {top_n} Publishers by Number of Articles")
        plt.xlabel("Publisher")
        plt.ylabel("Number of Articles")
        plt.xticks(rotation=45)
        plt.show()

        return top_counts

    # ============================================================
    # 2. News Type Analysis per Publisher
    # ============================================================
    def news_type_by_publisher(self):
        """
        Simple analysis of differences in headlines per publisher.
        Calculates average headline length per publisher as a proxy for type.

        Returns:
            pd.DataFrame: publisher vs avg headline length
        """
        self.df["headline_length"] = self.df[self.text_column].astype(
            str).str.len()
        avg_length = self.df.groupby(self.publisher_column)[
            "headline_length"].mean().sort_values(ascending=False)

        print("Average Headline Length per Publisher (as proxy for news type):")
        print(avg_length.head(10))

        avg_length.head(10).plot(kind="bar", color="orange")
        plt.title("Average Headline Length by Publisher")
        plt.xlabel("Publisher")
        plt.ylabel("Average Headline Length")
        plt.xticks(rotation=45)
        plt.show()

        return avg_length

    # ============================================================
    # 3. Email Domain Analysis
    # ============================================================
    def email_domain_analysis(self):
        """
        Identify unique email domains in publisher column if emails are used.

        Returns:
            pd.Series: counts of top domains
        """
        # Extract domain if publisher looks like email
        self.df["publisher_domain"] = self.df[self.publisher_column].apply(
            lambda x: re.search(
                r"@([\w.-]+)", x).group(1) if re.search(r"@([\w.-]+)", str(x)) else None
        )

        domain_counts = self.df["publisher_domain"].value_counts().dropna()
        top_domains = domain_counts.head(10)

        if not top_domains.empty:
            print("Top Publisher Email Domains:")
            print(top_domains)

            top_domains.plot(kind="bar", color="green")
            plt.title("Top Publisher Email Domains")
            plt.xlabel("Domain")
            plt.ylabel("Number of Articles")
            plt.xticks(rotation=45)
            plt.show()
        else:
            print("No email addresses found in publisher column.")

        return top_domains

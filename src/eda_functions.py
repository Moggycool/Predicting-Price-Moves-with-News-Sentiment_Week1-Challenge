"""Module providing Utility functions for 
loading, cleaning, and analyzing raw_analyst_ratings.csv."""
# eda_functions.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 1. Load Data
# ----------------------------


def load_data(file_path: str) -> pd.DataFrame:
    """Load large CSV efficiently."""
    return pd.read_csv(file_path, low_memory=False)

# ----------------------------
# 2. Preprocessing & Cleaning
# ----------------------------


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset: fix dates, drop missing/duplicates, create headline length."""

    # Standardize column names (lowercase)
    df.columns = df.columns.str.lower()

    # Convert date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

    # Remove duplicates
    df = df.drop_duplicates()

    # Critical fields
    critical_cols = [c for c in ['headline',
                                 'publisher', 'date'] if c in df.columns]
    df = df.dropna(subset=critical_cols)

    # Headline length
    if 'headline' in df.columns:
        df['headline_length'] = df['headline'].astype(str).apply(len)

    return df

# ----------------------------
# 3. Descriptive Statistics
# ----------------------------


def headline_length_stats(df: pd.DataFrame):
    """Return descriptive statistics for headline length."""
    if 'headline_length' in df.columns:
        return df['headline_length'].describe()


def plot_headline_length(df: pd.DataFrame):
    """Plot distribution of headline length."""
    plt.figure(figsize=(10, 5))
    sns.histplot(df['headline_length'], bins=50)
    plt.title('Headline Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.show()


def publisher_counts(df: pd.DataFrame):
    """Return count of articles per publisher."""
    return df['publisher'].value_counts()


def plot_top_publishers(df: pd.DataFrame, top_n=20):
    """Plot bar chart of top N publishers."""
    counts = df['publisher'].value_counts().head(top_n)
    plt.figure(figsize=(12, 6))
    counts.plot(kind='bar')
    plt.title(f'Top {top_n} Publishers by Article Count')
    plt.xlabel('Publisher')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


# ----------------------------
# 4. Publication Date Trends
# ----------------------------
def monthly_trends(df: pd.DataFrame):
    """Return article counts grouped by year-month."""
    df['year_month'] = df['date'].dt.to_period('M')
    return df.groupby('year_month').size()


def plot_monthly_trends(df: pd.DataFrame):
    """Plot monthly publication trends."""
    monthly = monthly_trends(df)
    monthly.plot(figsize=(12, 6))
    plt.title('Articles Over Time (Monthly)')
    plt.xlabel('Year-Month')
    plt.ylabel('Count')
    plt.show()

# ----------------------------
# 5. Save Processed Data
# ----------------------------


def save_processed(df: pd.DataFrame, output_path: str):
    """Save cleaned DataFrame to CSV."""
    df.to_csv(output_path, index=False)

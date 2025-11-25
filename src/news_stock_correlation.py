"""
news_stock_correlation.py
====================================================
Task-3: News Sentiment vs Stock Price Movement Analysis

Windows-friendly CLI and importable module.

Features:
- Load preprocessed news & stock datasets
- Normalize/align dates
- Sentiment analysis on headlines (TextBlob & VADER)
- Aggregate daily sentiment per stock
- Compute daily stock returns
- Merge datasets
- Pearson correlation analysis
- Save outputs (CSV + correlation report)
"""

import argparse
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure VADER lexicon is downloaded
nltk.download("vader_lexicon")


# ------------------------- Functions ------------------------- #

def compute_sentiment_textblob(text: str) -> float:
    """Compute TextBlob polarity (-1 to 1) for a headline."""
    if pd.isna(text) or not isinstance(text, str):
        return 0.0
    return TextBlob(text).sentiment.polarity


def compute_sentiment_vader(text: str, sia: SentimentIntensityAnalyzer) -> float:
    """Compute VADER compound sentiment (-1 to 1) for a headline."""
    if pd.isna(text) or not isinstance(text, str):
        return 0.0
    return sia.polarity_scores(text)["compound"]


def load_and_prepare_news(news_path: Path) -> pd.DataFrame:
    """Load news CSV, compute TextBlob & VADER sentiment, aggregate daily per stock."""
    df = pd.read_csv(news_path, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"]).dt.date

    sia = SentimentIntensityAnalyzer()
    df["sentiment_textblob"] = df["headline"].apply(compute_sentiment_textblob)
    df["sentiment_vader"] = df["headline"].apply(
        lambda x: compute_sentiment_vader(x, sia))

    daily = (
        df.groupby(["date", "stock"])[
            ["sentiment_textblob", "sentiment_vader"]]
        .mean()
        .reset_index()
    )
    return daily


def load_and_prepare_stock(stock_path: Path) -> pd.DataFrame:
    """Load stock CSV, compute daily returns per stock."""
    df = pd.read_csv(stock_path, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["daily_return"] = df.groupby("stock")["close"].pct_change()
    return df[["date", "stock", "daily_return"]]


def merge_and_analyze(news_df: pd.DataFrame, stock_df: pd.DataFrame) -> pd.DataFrame:
    """Merge daily sentiment and stock returns datasets."""
    merged = pd.merge(news_df, stock_df, on=["date", "stock"], how="inner")
    merged.dropna(inplace=True)
    return merged


def compute_correlation(df: pd.DataFrame, col1: str, col2: str) -> float:
    """Compute Pearson correlation coefficient between two columns."""
    corr, _ = pearsonr(df[col1], df[col2])
    return corr


# ------------------------- CLI ------------------------- #

def main():
    """Main function to run the analysis via CLI."""
    parser = argparse.ArgumentParser(
        description="Task-3: News Sentiment vs Stock Returns Analysis")
    parser.add_argument("--news", required=True,
                        help="Path to preprocessed news CSV")
    parser.add_argument("--stock", required=True,
                        help="Path to preprocessed stock CSV")
    parser.add_argument("--backend", default="textblob", choices=["textblob", "vader"],
                        help="Sentiment engine to prioritize for plots")
    args = parser.parse_args()

    # Windows-friendly path resolution
    news_path = Path(args.news).resolve()
    stock_path = Path(args.stock).resolve()
    output_dir = Path("data").resolve()
    output_dir.mkdir(exist_ok=True)

    # Load & process
    news_df = load_and_prepare_news(news_path)
    stock_df = load_and_prepare_stock(stock_path)
    merged_df = merge_and_analyze(news_df, stock_df)

    # Save merged CSV
    merged_df.to_csv(output_dir / "daily_sentiment_returns.csv",
                     index=False, encoding="utf-8")

    # Correlation analysis
    corr_textblob = compute_correlation(
        merged_df, "sentiment_textblob", "daily_return")
    corr_vader = compute_correlation(
        merged_df, "sentiment_vader", "daily_return")

    # Save correlation report
    report_file = output_dir / "correlation_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("Correlation Report\n")
        f.write("----------------------\n")
        f.write(f"TextBlob vs Returns: {corr_textblob:.4f}\n")
        f.write(f"VADER vs Returns:    {corr_vader:.4f}\n")

    print(f"✅ Analysis complete — outputs saved in {output_dir}")


if __name__ == "__main__":
    main()
# ------------------------- Streamlit Dashboard ------------------------- #

"""
app4.py
====================================================
Multi-Stock Sentiment & Stock Returns Dashboard

Features:
✅ Multi-stock overlay plots
✅ Dynamic sentiment engine selection (TextBlob / VADER)
✅ Moving averages & volatility bands per stock
✅ Rolling-window correlation heatmap
✅ KPIs update per selected engine and tickers
✅ Fully Windows-friendly and Pylint-clean
✅ Imports functions from src.news_stock_correlation
"""

from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Import functions from updated news_stock_correlation.py
from src.news_stock_correlation import (
    load_and_prepare_news,
    load_and_prepare_stock,
    merge_and_analyze,
    compute_correlation
)

# ---------------- Paths ---------------- #
data_dir = Path("data")
news_file = data_dir / "preprocessed_data.csv"
stock_file = data_dir / "preprocessed_stock_data.csv"
report_file = data_dir / "correlation_report.txt"

st.set_page_config(page_title="Multi-Stock Sentiment Dashboard", layout="wide")
st.title("📊 Multi-Stock Sentiment & Stock Returns Dashboard")

# ---------------- Sidebar Filters ---------------- #
st.sidebar.header("🔍 Dashboard Controls")
st.info("Loading and processing data...")

# Load & process data
news_df = load_and_prepare_news(news_file)
stock_df = load_and_prepare_stock(stock_file)
merged_df = merge_and_analyze(news_df, stock_df)

tickers = sorted(merged_df["stock"].unique())
selected_tickers = st.sidebar.multiselect(
    "Select Stock(s)", tickers, default=tickers[:2]
)

sentiment_engine = st.sidebar.radio(
    "Select Sentiment Engine", ("TextBlob", "VADER")
)
SENTIMENT_COL = "sentiment_textblob" if sentiment_engine == "TextBlob" else "sentiment_vader"

# Date slider
min_date, max_date = merged_df["date"].min(), merged_df["date"].max()
selected_dates = st.sidebar.date_input(
    "Select Date Range",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Moving averages & volatility bands
ma_window = st.sidebar.slider("Moving Average Window (days)", 2, 30, 7)
vol_band_multiplier = st.sidebar.slider(
    "Volatility Band Multiplier", 0.5, 3.0, 2.0)

# Filter data
df = merged_df[
    (merged_df["stock"].isin(selected_tickers)) &
    (merged_df["date"] >= selected_dates[0]) &
    (merged_df["date"] <= selected_dates[1])
].copy()

# ---------------- KPIs ---------------- #
st.subheader("📌 Key Performance Indicators (KPIs)")
cols = st.columns(min(4, len(selected_tickers)))

for idx, ticker in enumerate(selected_tickers):
    sub_df = df[df["stock"] == ticker]
    avg_sentiment = sub_df[SENTIMENT_COL].mean()
    avg_return = sub_df["daily_return"].mean()
    volatility = sub_df["daily_return"].std()
    corr = compute_correlation(sub_df, SENTIMENT_COL, "daily_return")
    col = cols[idx % 4]
    col.metric(f"{ticker} Avg Sentiment", f"{avg_sentiment:.3f}")
    col.metric(f"{ticker} Avg Return", f"{avg_return:.3%}")
    col.metric(f"{ticker} Volatility", f"{volatility:.3%}")
    col.metric(f"{ticker} Corr", f"{corr:.3f}")

# ---------------- Multi-Stock Overlay ---------------- #
st.subheader("📈 Multi-Stock Returns & Sentiment Overlay")
fig_overlay = go.Figure()

for ticker in selected_tickers:
    sub_df = df[df["stock"] == ticker].sort_values("date")
    # Sentiment line
    fig_overlay.add_trace(go.Scatter(
        x=sub_df["date"], y=sub_df[SENTIMENT_COL], mode="lines+markers",
        name=f"{ticker} {sentiment_engine} Sentiment"
    ))
    # Daily returns line
    fig_overlay.add_trace(go.Scatter(
        x=sub_df["date"], y=sub_df["daily_return"], mode="lines",
        name=f"{ticker} Return", yaxis="y2"
    ))
    # Moving average & volatility bands
    ma = sub_df["daily_return"].rolling(ma_window).mean()
    vol = sub_df["daily_return"].rolling(ma_window).std()
    fig_overlay.add_trace(go.Scatter(
        x=sub_df["date"], y=ma, mode="lines", line=dict(dash="dot"),
        name=f"{ticker} {ma_window}-day MA"
    ))
    fig_overlay.add_trace(go.Scatter(
        x=sub_df["date"], y=ma + vol * vol_band_multiplier, mode="lines",
        line=dict(dash="dash"), name=f"{ticker} Upper Band"
    ))
    fig_overlay.add_trace(go.Scatter(
        x=sub_df["date"], y=ma - vol * vol_band_multiplier, mode="lines",
        line=dict(dash="dash"), name=f"{ticker} Lower Band"
    ))

fig_overlay.update_layout(
    yaxis2=dict(overlaying="y", side="right"),
    title=f"Multi-Stock Overlay: {sentiment_engine} Sentiment & Returns with MA/Bands"
)
st.plotly_chart(fig_overlay, use_container_width=True)

# ---------------- Sentiment Distribution Histogram ---------------- #
st.subheader(f"📊 {sentiment_engine} Sentiment Distribution")
fig_hist = px.histogram(df, x=SENTIMENT_COL,
                        color="stock", nbins=30, marginal="box")
st.plotly_chart(fig_hist, use_container_width=True)

# ---------------- Rolling-Window Correlation Heatmap ---------------- #
st.subheader("📊 Rolling-Window Correlation Heatmap")
window_days = st.sidebar.slider("Rolling Correlation Window (days)", 2, 30, 7)
pivot_corr = pd.DataFrame()

for ticker in selected_tickers:
    sub_df = df[df["stock"] == ticker].sort_values("date")
    rolling_corr = sub_df[SENTIMENT_COL].rolling(
        window_days).corr(sub_df["daily_return"])
    pivot_corr[ticker] = rolling_corr

fig_corr = px.imshow(
    pivot_corr.T.corr(),
    text_auto=True, color_continuous_scale="RdBu_r",
    title=f"Rolling {window_days}-Day Sentiment vs Return Correlation"
)
st.plotly_chart(fig_corr, use_container_width=True)

# ---------------- Correlation Report ---------------- #
if report_file.exists():
    st.subheader("📄 Correlation Report")
    with open(report_file, "r", encoding="utf-8") as f:
        st.text(f.read())
else:
    st.warning("⚠ correlation_report.txt not found.")

st.markdown("---")
st.caption(
    f"✅ Dashboard with MA, volatility bands & rolling correlation using {sentiment_engine}")
st.caption(
    "Select different stocks and date ranges to explore sentiment vs stock returns.")
st.caption("Developed by Moges Behailu")

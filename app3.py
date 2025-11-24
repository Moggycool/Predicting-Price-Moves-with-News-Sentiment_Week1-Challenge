"""
Streamlit Dashboard: Stock Analysis
Preprocessing, Technical Indicators, Financial Metrics & Visualization
"""

# --------------------------
# Standard library imports
from pathlib import Path
import pandas as pd

# --------------------------
# Data visualization
# import matplotlib.pyplot as plt  # pylint: disable=import-error
import seaborn as sns
import streamlit as st

# --------------------------
# Custom module imports
from src.stock_preprocessor import StockPreprocessor
from src.technical_analysis import TechnicalAnalysis
from src.financial_metrics import FinancialMetrics
from src.quantitative_analysis import QuantitativeAnalysis

# --------------------------
# Streamlit configuration
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
sns.set_style("whitegrid")

# --------------------------
# Sidebar controls
st.sidebar.title("Stock Analysis Dashboard")
run_preprocessing = st.sidebar.checkbox("Run preprocessing", value=True)
compute_technical = st.sidebar.checkbox(
    "Compute Technical Indicators", value=True)
compute_metrics = st.sidebar.checkbox("Compute Financial Metrics", value=True)
visualize_data = st.sidebar.checkbox(
    "Visualize Quantitative Analysis", value=True)

# --------------------------
# Set data folder path
DATA_DIR = Path("data")  # CSVs are under data/

# --------------------------
# Load or preprocess data
if run_preprocessing:
    st.info("Preprocessing stock data...")
    preprocessor = StockPreprocessor(DATA_DIR)
    df = preprocessor.preprocess_all()
    preprocessor.save_preprocessed(df)
    st.success("Preprocessing complete!")
else:
    csv_file = DATA_DIR / "preprocessed_stock_data.csv"
    if not csv_file.exists():
        st.error("Preprocessed CSV not found. Please run preprocessing first.")
        st.stop()
    df = pd.read_csv(csv_file)
    df["Date"] = pd.to_datetime(df["Date"])
    st.success("Loaded preprocessed stock data!")

st.write(f"Data preview: {df.shape[0]} rows, {df.shape[1]} columns")
st.dataframe(df.head())

# --------------------------
# Stock selector for interactive plots
symbols = df["Symbol"].unique().tolist()
selected_symbols = st.sidebar.multiselect(
    "Select stocks to display", options=symbols, default=symbols
)
df_selected = df[df["Symbol"].isin(selected_symbols)]

# --------------------------
# Date range selector
min_date = df_selected["Date"].min()
max_date = df_selected["Date"].max()
start_date, end_date = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
df_selected = df_selected[(df_selected["Date"] >= pd.to_datetime(start_date)) &
                          (df_selected["Date"] <= pd.to_datetime(end_date))]

# --------------------------
# Technical Analysis
if compute_technical:
    # pylint: disable=C0103
    st.header("📈 Technical Analysis")
    ta = TechnicalAnalysis(df_selected)
    df_selected = ta.compute_sma(period=20)
    df_selected = ta.compute_ema(period=20)
    ta.compute_rsi(period=14)

    df_selected = ta.compute_macd()
    st.write("Technical indicators added to the DataFrame")
    st.dataframe(df_selected.head())
    st.subheader("Close Price & SMA Plot")
    st.pyplot(ta.plot_sma())

# --------------------------
# Financial Metrics
if compute_metrics:
    st.header("💰 Financial Metrics")
    fm = FinancialMetrics(df_selected)
    df_daily = fm.compute_daily_returns()
    df_cum = fm.compute_cumulative_returns()
    df_vol = fm.compute_volatility(window=14)
    st.write("Daily returns, cumulative returns, and volatility calculated")
    st.subheader("Daily Returns Plot")
    st.pyplot(fm.plot_daily_returns(df_daily))

# --------------------------
# Quantitative Analysis & Visualization
if visualize_data:
    st.header("📊 Quantitative Visualization")
    qa = QuantitativeAnalysis(df_selected)

    st.subheader("Closing Prices")
    st.pyplot(qa.plot_closing_prices())

    st.subheader("Daily Returns")
    st.pyplot(qa.plot_daily_returns(df_daily))

    st.subheader("Correlation Matrix of Daily Returns")
    st.pyplot(qa.plot_correlation_matrix(df_daily))

st.success("Dashboard is fully interactive!")

"""Application entry point for the News Sentiment Analysis Dashboard."""

import importlib
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pyLDAvis
from pyLDAvis import sklearn as pyLDAvis_sklearn

# -------------------------
# Custom modules (updated names)
# -------------------------
from pub_ana_stream import PublisherAnalyzer
from tex_ana_stream import TopicModeler
from time_ana_stream import TimeSeriesAnalyzer

import pub_ana_stream
import tex_ana_stream
import time_ana_stream

# Reload modules to reflect updates
importlib.reload(pub_ana_stream)
importlib.reload(tex_ana_stream)
importlib.reload(time_ana_stream)

# -------------------------
# Streamlit configuration
# -------------------------
st.set_page_config(
    page_title="News Sentiment Analysis Dashboard", layout="wide")

sns.set(style="whitegrid")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("News Sentiment Analysis Dashboard")

data_file = st.sidebar.file_uploader("Upload preprocessed CSV", type=["csv"])
num_topics = st.sidebar.slider("Number of Topics (LDA)", 2, 10, 4)
top_publishers = st.sidebar.slider("Top Publishers", 5, 20, 10)

# -------------------------
# Load CSV
# -------------------------
if data_file is None:
    st.warning("Please upload a preprocessed CSV to continue.")
    st.stop()

df = pd.read_csv(data_file)
st.success(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# -------------------------
# Time Series Analysis
# -------------------------
st.header("📈 Time Series Analysis")

ts = TimeSeriesAnalyzer(df)
daily_counts, spikes, hourly_counts = ts.run()

# Daily trend
st.subheader("Daily Article Trend")
fig = ts.plot_daily_trend(spikes=spikes, return_fig=True)
st.pyplot(fig)

# Hourly distribution
st.subheader("Hourly Distribution of Articles")
fig = ts.plot_hourly_distribution(return_fig=True)
st.pyplot(fig)

# Weekday-Hour heatmap
st.subheader("Weekday-Hour Heatmap")
fig = ts.plot_weekday_hour_heatmap(return_fig=True)
st.pyplot(fig)

# -------------------------
# Topic Modeling
# -------------------------
st.header("📝 Topic Modeling (LDA)")

topic_model = TopicModeler(
    df=df, num_topics=num_topics, max_features=1000, sample_size=1000)
topics, dt_matrix, vectorizer, lda_model = topic_model.run()

st.subheader("Top Words per Topic")
st.text(topics)

# -------------------------
# LDA Interactive Visualization
# -------------------------
st.subheader("Interactive LDA Visualization")
try:
    vis_data = pyLDAvis_sklearn.prepare(
        lda_model, dt_matrix, vectorizer, mds='tsne')
    st.components.v1.html(pyLDAvis.prepared_data_to_html(vis_data), height=900)
except Exception as e:
    st.error("pyLDAvis failed to render LDA visualization.")
    st.info(str(e))

# -------------------------
# Publisher Analysis
# -------------------------
st.header("🏢 Publisher Analysis")

pub = PublisherAnalyzer(df=df)
pub_counts, domain_counts, news_type_dist = pub.run_full_analysis(
    top_n=top_publishers, type_column="stock")

# Top publishers
st.subheader(f"Top {top_publishers} Publishers")
fig = pub.plot_top_publishers(top_n=top_publishers, return_fig=True)
st.pyplot(fig)

# Top domains
st.subheader("Top Domains")
fig = pub.plot_top_domains(top_n=top_publishers, return_fig=True)
st.pyplot(fig)

# News type distribution
st.subheader("News Type Distribution per Publisher")
st.dataframe(news_type_dist.head(top_publishers))

st.success("All visualizations loaded successfully!")

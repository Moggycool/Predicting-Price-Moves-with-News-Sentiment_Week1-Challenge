"""Application entry point for the News Sentiment Analysis Dashboard."""
# import necessary libraries
import os  # pylint: disable=unused-import
from pathlib import Path
import importlib
import sys
# pylint: disable=wrong-import-position
# Data visualization libraries
import pyLDAvis.sklearn   # ✔ FIXED: sklearn_models removed
import pyLDAvis
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Custom module imports
from publication_analysis import PublisherAnalyzer
from topic_modeling import TopicModeler
from time_analysis import TimeSeriesAnalyzer
import publication_analysis
import topic_modeling
import time_analysis
# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

# Reload custom modules

importlib.reload(time_analysis)
importlib.reload(topic_modeling)
importlib.reload(publication_analysis)
# Streamlit and data viz imports

# Set visualization styles
plt.style.use("seaborn-darkgrid")
sns.set(style="whitegrid")
# --------------------------
# Streamlit Configuration
# --------------------------
st.set_page_config(
    page_title="News Sentiment Analysis Dashboard",
    layout="wide"
)


# --------------------------
# Sidebar
# --------------------------
st.sidebar.title("News Sentiment Analysis Dashboard")

data_file = st.sidebar.file_uploader(
    "Upload preprocessed CSV", type=["csv"], accept_multiple_files=False
)

num_topics = st.sidebar.slider("Number of Topics (LDA)", 2, 10, 3)
top_publishers = st.sidebar.slider("Top Publishers", 5, 20, 10)

# --------------------------
# Load data
# --------------------------
if data_file is not None:
    df = pd.read_csv(data_file)
    st.success(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
else:
    st.warning("Please upload a preprocessed CSV to continue.")
    st.stop()

# --------------------------
# Time Series Analysis
# --------------------------
st.header("📈 Time Series Analysis")

ts_analyzer = TimeSeriesAnalyzer(df)   # ✔ uses dataframe instead of file path
daily_counts, spikes, hourly_counts = ts_analyzer.run()

st.subheader("Daily Article Trend")
fig, ax = plt.subplots(figsize=(10, 5))
ts_analyzer.plot_daily_trend(spikes=spikes)
st.pyplot(fig)

st.subheader("Hourly Distribution of Articles")
fig, ax = plt.subplots(figsize=(10, 4))
ts_analyzer.plot_hourly_distribution()
st.pyplot(fig)

st.subheader("Weekday-Hour Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
ts_analyzer.plot_weekday_hour_heatmap()
st.pyplot(fig)

# --------------------------
# Topic Modeling
# --------------------------
st.header("📝 Topic Modeling (LDA)")

topic_modeler_obj = TopicModeler(
    data=df,                 # ✔ use dataframe directly
    num_topics=num_topics,
    max_features=1000,
    sample_size=1000
)

topics = topic_modeler_obj.run()

st.subheader("Top Words per Topic")
st.text(topics)

# --------------------------
# LDA Visualization
# --------------------------
st.subheader("Interactive Topic Visualization")

try:
    lda_vis_data = pyLDAvis.sklearn.prepare(
        topic_modeler_obj.lda_model,
        topic_modeler_obj.doc_term_matrix,
        topic_modeler_obj.vectorizer,
        mds='tsne'
    )

    st.components.v1.html(
        pyLDAvis.prepared_data_to_html(lda_vis_data),
        height=800
    )

except Exception as e:
    st.error("pyLDAvis failed to render the LDA visualization.")
    st.info(f"Error: {e}")
    st.info("Try: pip install pyLDAvis==3.4.0")

# --------------------------
# Publisher Analysis
# --------------------------
st.header("🏢 Publisher Analysis")

pub_analyzer = PublisherAnalyzer(df)  # ✔ use dataframe directly
publisher_counts, domain_counts, news_type_dist = pub_analyzer.run_full_analysis(
    top_n=top_publishers,
    type_column="stock"
)

st.subheader(f"Top {top_publishers} Publishers")
st.bar_chart(publisher_counts.head(top_publishers))

st.subheader("Top Domains")
st.bar_chart(domain_counts.head(top_publishers))

st.subheader("News Type Distribution per Publisher")
st.dataframe(news_type_dist.head(top_publishers))

st.success("All visualizations are interactive!")

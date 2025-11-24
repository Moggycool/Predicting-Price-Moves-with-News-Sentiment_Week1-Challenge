"""
Application entry point for the News Sentiment Analysis Dashboard
(using Streamlit-ready modules)
"""
# Standard library imports
from pathlib import Path
import importlib
import sys
# ----------------==========
# Data visualization and manipulation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
# ==========================
# pyLDAvis for LDA visualization
import pyLDAvis
import pyLDAvis.sklearn  # needed for sklearn integration
# --------------------------
# Custom module imports
from src.pub_ana_stream import PublisherAnalyzer
from src.tex_ana_stream import TopicModeler
from src.time_ana_stream import TimeSeriesAnalyzer
# ==========================

# Add src to path
# --------------------------
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

# --------------------------
# Custom module imports
# --------------------------

# Reload modules to reflect any changes
importlib.reload(sys.modules.get("pub_ana_stream"))
importlib.reload(sys.modules.get("tex_ana_stream"))
importlib.reload(sys.modules.get("time_ana_stream"))

# --------------------------
# Streamlit Configuration
# --------------------------
st.set_page_config(
    page_title="News Sentiment Analysis Dashboard",
    layout="wide"
)
sns.set(style="whitegrid")
plt.style.use("seaborn-darkgrid")

# --------------------------
# Sidebar
# --------------------------
st.sidebar.title("News Sentiment Analysis Dashboard")

data_file = st.sidebar.file_uploader(
    "Upload preprocessed CSV", type=["csv"]
)

num_topics = st.sidebar.slider("Number of Topics (LDA)", 2, 10, 3)
top_publishers = st.sidebar.slider("Top Publishers", 5, 20, 10)

# --------------------------
# Load Data
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
ts_analyzer = TimeSeriesAnalyzer(df)
daily_counts, spikes, hourly_counts = ts_analyzer.run()

st.subheader("Daily Article Trend")
fig = ts_analyzer.plot_daily_trend(spikes=spikes, return_fig=True)
st.pyplot(fig)

st.subheader("Hourly Distribution of Articles")
fig = ts_analyzer.plot_hourly_distribution(return_fig=True)
st.pyplot(fig)

st.subheader("Weekday-Hour Heatmap")
fig = ts_analyzer.plot_weekday_hour_heatmap(return_fig=True)
st.pyplot(fig)

# --------------------------
# Topic Modeling
# --------------------------
st.header("📝 Topic Modeling (LDA)")
topic_modeler = TopicModeler(
    df, num_topics=num_topics, max_features=1000, sample_size=1000)
topics = topic_modeler.run()  # pylint: disable=invalid-name

st.subheader("Top Words per Topic")
st.text(topics)

# --------------------------
# Interactive LDA Visualization
# --------------------------
st.subheader("Interactive Topic Visualization")
try:
    lda_vis_data = pyLDAvis.sklearn.prepare(
        topic_modeler.lda_model,
        topic_modeler.vectorizer.transform(df["headline"].astype(str)),
        topic_modeler.vectorizer,
        mds='tsne'
    )
    st.components.v1.html(
        pyLDAvis.prepared_data_to_html(lda_vis_data), height=800)

except (AttributeError, ValueError, TypeError) as e:
    st.error("pyLDAvis failed to render the LDA visualization.")
    st.info(f"Error: {e}")
    st.info("Try: pip install pyLDAvis==3.4.0")

# --------------------------
# Publisher Analysis
# --------------------------
st.header("🏢 Publisher Analysis")
pub_analyzer = PublisherAnalyzer(df)
publisher_counts, domain_counts, news_type_dist = pub_analyzer.run_full_analysis(
    top_n=top_publishers,
    type_column="stock"
)

st.subheader(f"Top {top_publishers} Publishers")
fig = pub_analyzer.plot_top_publishers(top_n=top_publishers, return_fig=True)
st.pyplot(fig)

st.subheader("Top Domains")
fig = pub_analyzer.plot_top_domains(top_n=top_publishers, return_fig=True)
st.pyplot(fig)

st.subheader("News Type Distribution per Publisher")
st.dataframe(news_type_dist.head(top_publishers))

st.success("All visualizations are interactive!")

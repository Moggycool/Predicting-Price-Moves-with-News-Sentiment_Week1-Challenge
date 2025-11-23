"""Application Entry Point for News Sentiment Analysis Dashboard."""
import sys
from pathlib import Path
# Topic Modeling visualization
import pyLDAvis
import pyLDAvis.sklearn
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path (if src is at root level)
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

# Import custom modules
try:
    from publication_analysis import PublisherAnalyzer
    from topic_modeling import TopicModeler
    from time_analysis import TimeSeriesAnalyzer
except ImportError:
    # Try scripts/src structure
    src_path = Path(__file__).parent / "scripts" / "src"
    sys.path.append(str(src_path))

    # Re-import custom modules
    # ------------------------------------------
    # pylint: disable=import-error
    # pyright: ignore[reportMissingImports]
    from publication_analysis import PublisherAnalyzer
    from topic_modeling import TopicModeler
    from time_analysis import TimeSeriesAnalyzer
# --------------------------
# Streamlit & Visualization Config

# Set Streamlit page config
st.set_page_config(
    page_title="News Sentiment Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Set seaborn style
sns.set(style="whitegrid")


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
ts_analyzer = TimeSeriesAnalyzer(data_file)
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
    data_path=data_file,
    num_topics=num_topics,
    max_features=1000,
    sample_size=1000
)
topics = topic_modeler_obj.run()
st.subheader("Top Words per Topic")
st.text(topics)

st.subheader("Interactive Topic Visualization")
lda_vis_data = pyLDAvis.sklearn.prepare(
    topic_modeler_obj.lda_model,
    topic_modeler_obj.doc_term_matrix,
    topic_modeler_obj.vectorizer,
    mds='tsne'
)
# Display in Streamlit
st.components.v1.html(pyLDAvis.prepared_data_to_html(lda_vis_data), height=800)

# --------------------------
# Publisher Analysis
# --------------------------
st.header("🏢 Publisher Analysis")

pub_analyzer = PublisherAnalyzer(data_file)
publisher_counts, domain_counts, news_type_dist = pub_analyzer.run_full_analysis(
    top_n=top_publishers, type_column="stock")

st.subheader(f"Top {top_publishers} Publishers")
st.bar_chart(publisher_counts.head(top_publishers))

st.subheader("Top Domains")
st.bar_chart(domain_counts.head(top_publishers))

st.subheader("News Type Distribution per Publisher")
st.dataframe(news_type_dist.head(top_publishers))

st.success("All visualizations are interactive!")

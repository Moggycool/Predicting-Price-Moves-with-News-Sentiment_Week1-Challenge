"""
Application entry point for the News Sentiment Analysis Dashboard
(using Streamlit-ready modules)
"""

# --------------------------
# Standard library imports
from pathlib import Path
import importlib
import sys
import inspect

# --------------------------
# Data visualization and manipulation
import matplotlib.pyplot as plt  # pylint: disable=import-error
import seaborn as sns
import pandas as pd
import streamlit as st

# --------------------------
# pyLDAvis for LDA visualization
import pyLDAvis
# import pyLDAvis.sklearn

# --------------------------
# Streamlit configuration
st.set_page_config(
    page_title="News Sentiment Analysis Dashboard",
    layout="wide"
)
sns.set_theme(style="darkgrid")  # or "whitegrid"

# --------------------------
# Add src folder to path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

# --------------------------
# Module configuration for safe import/reload
modules_info = {
    "src.time_ana_stream": "time_ana_stream",
    "src.tex_ana_stream": "tex_ana_stream",
    "src.pub_ana_stream": "pub_ana_stream"
}

modules = {}

# ==========================
# Import or reload modules safely
for module_path, var_name in modules_info.items():
    if module_path in sys.modules:
        module = importlib.reload(sys.modules[module_path])
    else:
        module = importlib.import_module(module_path)
    modules[var_name] = module

# Assign modules to variables
time_ana_stream = modules["time_ana_stream"]
tex_ana_stream = modules["tex_ana_stream"]
pub_ana_stream = modules["pub_ana_stream"]

# ==========================
# Helper function to check if a class can be instantiated safely


def can_instantiate(cls):
    """Check if a class can be instantiated without required parameters."""
    sig = inspect.signature(cls)
    for param in sig.parameters.values():
        if param.default is param.empty and param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):  # pylint: disable=line-too-long
            return False
    return True


# ==========================
# Set max upload size to 1 GB (1024 MB)
# Set.set_option('server.maxUploadSize', 1024)
# ==========================

# Sidebar
st.sidebar.title("News Sentiment Analysis Dashboard")
data_file = st.sidebar.file_uploader("Upload preprocessed CSV", type=["csv"])
num_topics = st.sidebar.slider("Number of Topics (LDA)", 2, 10, 3)
top_publishers = st.sidebar.slider("Top Publishers", 5, 20, 10)

# ==========================
# Load data with fallback
local_csv = Path(__file__).parent / "data/processed/preprocessed_data.csv"
if data_file is not None:
    df = pd.read_csv(data_file)
    st.success(f"Data loaded from uploaded file: {df.shape}")
elif local_csv.exists():
    df = pd.read_csv(local_csv)
    st.success(f"Data loaded from local CSV: {df.shape}")
else:
    st.warning("Please upload a CSV or ensure local CSV exists.")
    st.stop()

# ==========================
# Time Series Analysis
st.header("📈 Time Series Analysis")
ts_analyzer = time_ana_stream.TimeSeriesAnalyzer(df)
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
st.header("📝 Topic Modeling (LDA)")
topic_modeler = tex_ana_stream.TopicModeler(
    df, num_topics=num_topics, max_features=1000, sample_size=1000
)
topics = topic_modeler.run()
st.subheader("Top Words per Topic")
st.text(topics)
# --------------------------
# Interactive LDA Visualization
st.subheader("Interactive Topic Visualization")
try:
    lda_vis_data = pyLDAvis.sklearn.prepare(
        topic_modeler.lda_model,
        topic_modeler.vectorizer.transform(df["headline"].astype(str)),
        topic_modeler.vectorizer.get_feature_names_out(),  # direct array, no Series
        mds='tsne'
    )
    st.components.v1.html(
        pyLDAvis.prepared_data_to_html(lda_vis_data), height=800
    )
except (AttributeError, ValueError, TypeError) as e:
    st.error("pyLDAvis failed to render the LDA visualization.")
    st.info(f"Error: {e}")
    st.info("Ensure CountVectorizer.get_feature_names_out() is used (sklearn >=1.0)")

# ==========================
# Publisher Analysis
st.header("🏢 Publisher Analysis")
pub_analyzer = pub_ana_stream.PublisherAnalyzer(df)
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

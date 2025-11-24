"""
tex_ana_stream.py — Streamlit-ready Topic Modeling on news headlines
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download("stopwords", quiet=True)


class TopicModeler:
    """Class for topic modeling on news headlines using LDA."""

    def __init__(
        self,
        df: pd.DataFrame,
        num_topics: int = 4,
        max_features: int = 1000,
        sample_size: int = 1000
    ):
        """
        Initialize TopicModeler.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed news dataset
        num_topics : int
            Number of topics to extract
        max_features : int
            Max vocabulary size for CountVectorizer
        sample_size : int
            Number of rows to sample (None = use all)
        """
        self.df = df.copy()
        self.num_topics = num_topics
        self.max_features = max_features
        self.sample_size = sample_size
        self.vectorizer = None
        self.lda_model = None

        # Sample if dataset is too large
        if self.sample_size is not None and len(self.df) > self.sample_size:
            self.df = self.df.sample(
                self.sample_size, random_state=42).reset_index(drop=True)

    # -------------------------
    def clean_headlines(self):
        """Clean text for NLP."""
        stop_words = set(stopwords.words("english"))

        def clean(text: str) -> str:
            if pd.isna(text) or str(text).strip() == "":
                return ""
            text = str(text).lower()
            text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
            text = re.sub(r"[^a-zA-Z ]", " ", text)    # Keep letters only
            text = " ".join([word for word in text.split()
                            if word not in stop_words])
            return text

        self.df["clean_headline"] = self.df["headline"].astype(
            str).apply(clean)
        self.df = self.df[self.df["clean_headline"].str.strip()
                          != ""].reset_index(drop=True)

    # -------------------------
    def vectorize(self) -> CountVectorizer:
        """Vectorize text using CountVectorizer."""
        self.vectorizer = CountVectorizer(
            max_features=self.max_features, stop_words="english")
        dt_matrix = self.vectorizer.fit_transform(self.df["clean_headline"])
        return dt_matrix

    # -------------------------
    def fit_lda(self, dt_matrix):
        """Fit LDA."""
        self.lda_model = LatentDirichletAllocation(
            n_components=self.num_topics,
            random_state=42,
            learning_method="batch",
            max_iter=15
        )
        self.lda_model.fit(dt_matrix)

    # -------------------------
    def display_topics(self, num_words: int = 10) -> str:
        """Return top words per topic as string."""
        feature_names = self.vectorizer.get_feature_names_out()
        topics_output = []

        for idx, topic in enumerate(self.lda_model.components_):
            top_words = [feature_names[i]
                         for i in topic.argsort()[-num_words:][::-1]]
            topics_output.append(f"Topic #{idx + 1}: {', '.join(top_words)}")

        return "\n".join(topics_output)

    # -------------------------
    def run(self) -> str:
        """Run full topic modeling pipeline."""
        if "headline" not in self.df.columns or self.df.empty:
            raise ValueError(
                "Dataset must contain non-empty 'headline' column!")

        self.clean_headlines()
        dt_matrix = self.vectorize()
        self.fit_lda(dt_matrix)
        topics = self.display_topics()
        return topics

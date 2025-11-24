"""
topic_modeler.py — Efficient Topic Modeling on news headlines
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download("stopwords", quiet=True)

# pylint: disable=too-many-instance-attributes


class TopicModeler:
    """ Class for topic modeling on news headlines using LDA. """

    def __init__(self, data_path: str, num_topics: int = 4, max_features: int = 1000, sample_size: int = 1000):  # pylint: disable=R0913
        """
        Initialize TopicModeler.

        :param data_path: Path to preprocessed_data.csv
        :param num_topics: Number of topics to extract
        :param max_features: Max vocabulary size for CountVectorizer
        :param sample_size: Number of rows to sample (None = use all)
        """
        self.data_path = data_path
        self.num_topics = num_topics
        self.max_features = max_features
        self.sample_size = sample_size
        self.vectorizer = None
        self.lda_model = None
        self.df = None

    # -------------------------
    def load_data(self):
        """Load preprocessed data."""
        self.df = pd.read_csv(self.data_path)
        if "headline" not in self.df.columns:
            raise ValueError("Dataset must contain 'headline' column!")
        # Sample if dataset is too large
        if self.sample_size is not None and len(self.df) > self.sample_size:
            self.df = self.df.sample(
                self.sample_size, random_state=42).reset_index(drop=True)

    # -------------------------
    def clean_headlines(self):
        """Clean text for NLP."""
        stop_words = set(stopwords.words("english"))

        def clean(text):
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
        # Remove empty cleaned headlines
        self.df = self.df[self.df["clean_headline"].str.strip()
                          != ""].reset_index(drop=True)

    # -------------------------
    def vectorize(self):
        """Vectorize text using CountVectorizer."""
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            stop_words="english"
        )
        dt_matrix = self.vectorizer.fit_transform(self.df["clean_headline"])
        return dt_matrix

    # -------------------------
    def fit_lda(self, dt_matrix):
        """Fit LDA with verbose output."""
        print(
            f"Fitting LDA with {self.num_topics} topics on {dt_matrix.shape[0]} documents...")
        self.lda_model = LatentDirichletAllocation(
            n_components=self.num_topics,
            random_state=42,
            learning_method="batch",
            verbose=1,  # show progress
            max_iter=15
        )
        self.lda_model.fit(dt_matrix)
        print("LDA fitting complete.")

    # -------------------------
    def display_topics(self, num_words=10):
        """Print top words per topic."""
        feature_names = self.vectorizer.get_feature_names_out()
        topics_output = []

        for idx, topic in enumerate(self.lda_model.components_):
            top_words = [feature_names[i]
                         for i in topic.argsort()[-num_words:][::-1]]
            topics_output.append(f"Topic #{idx+1}: {', '.join(top_words)}")

        return "\n".join(topics_output)

    # -------------------------
    def run(self):
        """Run the full topic modeling pipeline."""
        print("Loading data...")
        self.load_data()

        print("Cleaning headlines...")
        self.clean_headlines()
        if len(self.df) == 0:
            raise ValueError("No valid headlines left after cleaning!")

        print("Vectorizing text...")
        dt_matrix = self.vectorize()

        print("Fitting LDA...")
        self.fit_lda(dt_matrix)

        print("\nExtracted Topics:")
        topics = self.display_topics()
        print(topics)
        return topics
# -------------------------

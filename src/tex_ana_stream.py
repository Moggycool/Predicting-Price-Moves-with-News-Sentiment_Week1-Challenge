"""Topic modeling module for news headlines using LDA."""
import re  # pylint: disable=import-error
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download("stopwords", quiet=True)


class TopicModeler:
    """Topic modeling using LDA on news headlines."""

    def __init__(self, df: pd.DataFrame, num_topics=4, max_features=1000, sample_size=1000):
        self.df = df.copy()
        self.num_topics = num_topics
        self.max_features = max_features
        self.sample_size = sample_size
        self.vectorizer = None
        self.lda_model = None
        self.doc_term_matrix = None

    def clean_headlines(self):
        """Clean headlines by removing stopwords, URLs, and non-alphabetic characters."""
        stop_words = set(stopwords.words("english"))

        def clean(text):
            if pd.isna(text):
                return ""
            text = str(text).lower()
            text = re.sub(r"http\S+|www\S+", "", text)
            text = re.sub(r"[^a-zA-Z ]", " ", text)
            return " ".join([w for w in text.split() if w not in stop_words])
        self.df["clean_headline"] = self.df["headline"].astype(
            str).apply(clean)
        self.df = self.df[self.df["clean_headline"].str.strip()
                          != ""].reset_index(drop=True)

    def vectorize(self):
        """Convert cleaned headlines to document-term matrix."""
        self.vectorizer = CountVectorizer(
            max_features=self.max_features, stop_words="english")
        self.doc_term_matrix = self.vectorizer.fit_transform(
            self.df["clean_headline"])
        return self.doc_term_matrix

    def fit_lda(self):
        """Fit LDA model to the document-term matrix."""
        self.lda_model = LatentDirichletAllocation(
            n_components=self.num_topics,
            random_state=42,
            learning_method="batch",
            max_iter=15,
            verbose=1
        )
        self.lda_model.fit(self.doc_term_matrix)

    def display_topics(self, num_words=10):
        """Display top words for each topic."""
        feature_names = self.vectorizer.get_feature_names_out()
        topics_output = []
        for idx, topic in enumerate(self.lda_model.components_):
            top_words = [feature_names[i]
                         for i in topic.argsort()[-num_words:][::-1]]
            topics_output.append(f"Topic #{idx+1}: {', '.join(top_words)}")
        return "\n".join(top_words)

    def run(self):
        """Execute the full topic modeling pipeline."""
        if self.sample_size is not None and len(self.df) > self.sample_size:
            self.df = self.df.sample(
                self.sample_size, random_state=42).reset_index(drop=True)
        self.clean_headlines()
        if len(self.df) == 0:
            raise ValueError("No valid headlines left after cleaning!")
        self.vectorize()
        self.fit_lda()
        topics = self.display_topics()
        return topics, self.doc_term_matrix, self.vectorizer, self.lda_model

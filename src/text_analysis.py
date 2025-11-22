""" Text Analysis Module"""
# ============================================================
# text_analysis.py
# NLP + Topic Modeling for Analyst Ratings Headlines
# ============================================================

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class TextAnalysis:
    """ A reusable text analysis class for NLP tasks on financial news headlines."""

    def __init__(self, df, text_column="headline", n_topics=5):
        """
        Initialize the TextAnalysis class.

        Parameters:
        df (DataFrame): Preprocessed dataframe
        text_column (str): Column that contains text data
        n_topics (int): Number of topics to extract using LDA
        """
        self.df = df
        self.text_column = text_column
        self.n_topics = n_topics
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

    # ============================================================
    # 1. Basic Text Cleaning
    # ============================================================
    def clean_text(self, text):
        """ Clean text data: lowercase, remove punctuation, stopwords, and stem."""
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        tokens = text.split()

        tokens = [
            self.stemmer.stem(word)
            for word in tokens
            if word not in self.stop_words and len(word) > 2
        ]

        return " ".join(tokens)

    def preprocess(self):
        """ Preprocess the text data in the specified column."""
        print("Cleaning text data...")
        self.df["clean_text"] = self.df[self.text_column].astype(
            str).apply(self.clean_text)
        return self.df

    # ============================================================
    # 2. Keyword Extraction (TF-IDF)
    # ============================================================
    def extract_keywords(self, top_n=20):
        """ Extract top N keywords using TF-IDF."""
        print("Extracting keywords using TF-IDF...")

        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(self.df["clean_text"])

        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.sum(axis=0).A1

        keyword_df = (
            pd.DataFrame({"keyword": feature_names, "score": scores})
            .sort_values(by="score", ascending=False)
            .head(top_n)
        )

        return keyword_df

    # ============================================================
    # 3. Topic Modeling (LDA)
    # ============================================================
    def topic_modeling(self, top_words=10):
        """ Perform LDA topic modeling and return top words per topic."""
        print(f"Running LDA Topic Modeling (topics={self.n_topics})...")

        vectorizer = CountVectorizer(max_features=2000)
        dt_matrix = vectorizer.fit_transform(self.df["clean_text"])
        vocab = vectorizer.get_feature_names_out()

        lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            learning_method="batch",
        )
        lda.fit(dt_matrix)

        topics = {}
        for idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-top_words:]
            topics[f"Topic {idx+1}"] = [vocab[i] for i in top_indices]

        return topics

    # ============================================================
    # 4. Combined Report
    # ============================================================
    def full_analysis(self):
        """ Run full text analysis: cleaning, keyword extraction, and topic modeling."""
        print("============================================================")
        print("Running Full Text Analysis: Cleaning → Keywords → Topics")
        print("============================================================")

        self.preprocess()

        keywords = self.extract_keywords()
        topics = self.topic_modeling()

        return {
            "keywords": keywords,
            "topics": topics
        }

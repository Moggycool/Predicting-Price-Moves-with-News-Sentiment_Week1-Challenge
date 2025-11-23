"""
data_preprocess.py — Class-based preprocessing for news datasets.
"""

from pathlib import Path
import pandas as pd


class NewsDataPreprocessor:
    """
    Performs preprocessing on news datasets:
    - missing value handling
    - text cleaning
    - date formatting
    - saving preprocessed output
    """

    def __init__(self, save_dir: str | Path):
        """
        Initialize preprocessor with a directory to save output.

        Parameters
        ----------
        save_dir : str or Path
            Folder where preprocessed CSV will be saved.
        """
        self.save_dir = Path(save_dir)

        if not self.save_dir.exists():
            raise FileNotFoundError(
                f"Save directory does not exist: {self.save_dir}")

    # --------------------------------------------------------------------------
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning: strip whitespace, remove multiple spaces.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """
        if pd.isna(text):
            return ""

        text = str(text).strip()
        text = " ".join(text.split())  # remove extra whitespace
        return text

    # --------------------------------------------------------------------------
    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values by filling them with empty strings for text
        and forward filling for dates when possible.

        Returns
        -------
        pd.DataFrame
        """
        df = df.copy()

        # Fill missing text fields
        for col in ["headline", "publisher", "url", "stock"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        # Attempt to forward-fill dates if present
        if "date" in df.columns:
            df["date"] = df["date"].fillna(method="ffill")

        return df

    # --------------------------------------------------------------------------
    def format_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts date column to official format: YYYY-MM-DD HH:MM

        Returns
        -------
        pd.DataFrame
        """
        df = df.copy()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M")

        return df

    # --------------------------------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the full preprocessing pipeline.

        Returns
        -------
        pd.DataFrame
        """
        df = df.copy()

        # 1. Handle missing values
        df = self.handle_missing(df)

        # 2. Clean text fields
        for col in ["headline", "publisher", "url", "stock"]:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text)

        # 3. Format dates
        df = self.format_dates(df)

        return df

    # --------------------------------------------------------------------------
    def save(self, df: pd.DataFrame, filename: str = "preprocessed_data.csv") -> Path:
        """
        Save dataframe to CSV inside the save directory.

        Returns
        -------
        Path : Full path of saved file
        """
        save_path = self.save_dir / filename
        df.to_csv(save_path, index=False)
        print(f"Preprocessed data saved to: {save_path}")
        return save_path


# --------------------------------------------------------------------------
# Example usage (optional)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    BASE_DATA_DIR = Path(
        r"D:\Python\Week-1"
        r"\Predicting Price Moves with News Sentiment"
        r"\Predicting-Price-Moves-with-News-Sentiment_Week1-Challenge"
        r"\data"
    )

    # Example: Load data from previous loader
    from data_loader import NewsDataLoader
    loader = NewsDataLoader(BASE_DATA_DIR)
    df_raw = loader.load("raw_analyst_ratings.csv")

    # Preprocess
    processor = NewsDataPreprocessor(BASE_DATA_DIR)
    df_clean = processor.preprocess(df_raw)

    # Save
    processor.save(df_clean)

"""
data_loader.py — Class-based loader for news datasets.
"""

from pathlib import Path
import pandas as pd


class NewsDataLoader:
    """
    A reusable class for loading CSV news datasets
    from a predefined base directory.
    """

    def __init__(self, base_data_dir: str | Path):
        """
        Initialize the data loader with a base directory.

        Parameters
        ----------
        base_data_dir : str or Path
            Directory path where the raw CSV files are stored.
        """
        self.base_data_dir = Path(base_data_dir)

        if not self.base_data_dir.exists():
            raise FileNotFoundError(
                f"Base data directory does not exist: {self.base_data_dir}"
            )

    def load(self, filename: str) -> pd.DataFrame:
        """
        Load a CSV file from the base directory.

        Parameters
        ----------
        filename : str
            CSV file name (e.g., 'raw_analyst_ratings.csv').

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame with cleaned column names.
        """
        file_path = self.base_data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        df = pd.read_csv(file_path)   # pylint: disable=redefined-outer-name

        # Clean column names
        df.columns = df.columns.str.strip()

        # Optional debug display
        print("Preview of loaded data:")
        print(df.head())
        print("Shape:", df.shape)
        print("Columns:", list(df.columns))

        return df

    def list_files(self) -> list[str]:
        """
        List all CSV files available in the base data directory.

        Returns
        -------
        list of str
            Filenames of CSV files in the directory.
        """
        return [f.name for f in self.base_data_dir.glob("*.csv")]


# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    BASE_DATA_DIR = Path(
        r"D:\Python\Week-1"
        r"\Predicting Price Moves with News Sentiment"
        r"\Predicting-Price-Moves-with-News-Sentiment_Week1-Challenge"
        r"\data"
    )

    loader = NewsDataLoader(BASE_DATA_DIR)
    df = loader.load("raw_analyst_ratings.csv")
    print(df.info())

"""Module for preprocessing stock CSV files into a combined DataFrame."""
from pathlib import Path
import pandas as pd


class StockPreprocessor:
    """
    Loads, cleans, preprocesses, and combines multiple stock CSVs
    into a single DataFrame.
    """

    # List of stock CSV files to preprocess
    TARGET_FILES = ["NVDA.csv", "AAPL.csv",
                    "AMZN.csv", "GOOG.csv", "META.csv", "MSFT.csv"]

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory does not exist: {self.data_dir}")

    def load_and_clean_stock(self, file_name: str) -> pd.DataFrame:
        """
        Load a single stock CSV and perform basic cleaning.
        """
        file_path = self.data_dir / file_name
        if not file_path.exists():
            print(f"Warning: {file_path} not found. Skipping.")
            return None

        df = pd.read_csv(file_path)

        # Ensure required columns exist
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in {file_name}: {missing_cols}")

        # Drop rows with missing essential values
        df = df.dropna(subset=required_cols)

        # Convert Date column to datetime
        df["Date"] = pd.to_datetime(df["Date"])

        # Add Symbol column based on filename
        df["Symbol"] = file_name.replace(".csv", "")

        return df

    def preprocess_all(self) -> pd.DataFrame:
        """
        Load, clean, preprocess all target CSVs, and combine them.
        """
        dfs = []
        for file_name in self.TARGET_FILES:
            df = self.load_and_clean_stock(file_name)
            if df is not None:
                dfs.append(df)

        if not dfs:
            raise FileNotFoundError(
                "No target stock CSVs were found in the data folder.")

        # Combine all stock data
        combined_df = pd.concat(dfs, axis=0).reset_index(drop=True)

        # Optional: sort by Date and Symbol
        combined_df = combined_df.sort_values(
            by=["Symbol", "Date"]).reset_index(drop=True)
        return combined_df

    def save_preprocessed(self, df: pd.DataFrame, filename: str = "preprocessed_stock_data.csv"):
        """
        Save the preprocessed DataFrame to the data directory.
        """
        output_file = self.data_dir / filename
        df.to_csv(output_file, index=False)
        print(f"Preprocessed data saved to: {output_file}")


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    DATA_DIR = Path("data")  # Path to your project/data folder
    preprocessor = StockPreprocessor(DATA_DIR)
    combined_df_object = preprocessor.preprocess_all()
    preprocessor.save_preprocessed(combined_df_object)
    print(combined_df_object.head())

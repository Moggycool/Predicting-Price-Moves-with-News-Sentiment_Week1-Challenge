"""
load.py
Reusable class-based module for:
1. Loading raw_analyst_ratings.csv
2. Cleaning & preprocessing (with standardized datetime format)
3. Saving preprocessed data as pre_processed data.csv
"""

import logging
import os
import pandas as pd

# --------------------------------------------------------------
# Configure Logging
# --------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)


class DataLoader:
    """
    A reusable loader class for analyst ratings dataset.
    """

    REQUIRED_COLUMNS = ["headline", "url", "publisher", "date", "stock"]

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.directory = os.path.dirname(file_path)
        self.df = None

    # ----------------------------------------------------------
    # 1. LOAD DATA
    # ----------------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            df = pd.read_csv(self.file_path, encoding="utf-8",
                             low_memory=False)
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load CSV: {e}") from e

        # Check missing required columns
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise RuntimeError(f"[ERROR] Missing required columns: {missing}")

        self.df = df
        logging.info("Loaded data: %d rows, %d columns",
                     df.shape[0], df.shape[1])
        return df

    # ----------------------------------------------------------
    # 2. CLEAN + PREPROCESS
    # ----------------------------------------------------------
    def clean_preprocess(self) -> pd.DataFrame:
        """Clean and preprocess the dataset"""
        if self.df is None:
            raise RuntimeError("Data not loaded. Run load_data() first.")

        df = self.df.copy()

        # Remove duplicates
        df.drop_duplicates(inplace=True)

        # Strip whitespace
        text_columns = ["headline", "url", "publisher", "stock"]
        for col in text_columns:
            df[col] = df[col].astype(str).str.strip()

        # Date handling
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Drop invalid dates
        df = df.dropna(subset=["date"])

        # Drop rows missing essential fields
        df.dropna(subset=["headline", "publisher"], inplace=True)

        # Keep "date" column as datetime for EDA
        df["formatted_date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M")

        self.df = df
        logging.info("Cleaning & preprocessing completed.")

        return df

    # ----------------------------------------------------------
    # 3. SAVE PREPROCESSED DATA
    # ----------------------------------------------------------
    def save_preprocessed(self):
        """Save the preprocessed data to CSV"""
        if self.df is None:
            raise RuntimeError(
                "No cleaned data to save. Run clean_preprocess().")

        output_path = os.path.join(self.directory, "pre_processed data.csv")

        # Save formatted_date instead of datetime for output
        save_df = self.df.copy()
        save_df["date"] = save_df["formatted_date"]
        save_df.drop(columns=["formatted_date"], inplace=True)

        try:
            save_df.to_csv(output_path, index=False, encoding="utf-8")
            logging.info("Preprocessed data saved at: %s", output_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save CSV: {e}") from e

    # ----------------------------------------------------------
    # 4. RUN THE FULL PROCESS PIPELINE
    # ----------------------------------------------------------
    def process(self):
        """Run full load → clean → save pipeline"""
        self.load_data()
        self.clean_preprocess()
        self.save_preprocessed()


# --------------------------------------------------------------------------
# Run when executed directly
# --------------------------------------------------------------------------
if __name__ == "__main__":
    loader = DataLoader(
        r"D:\Python\Week-1\Data-Week-1\raw_analyst_ratings.csv"
    )
    loader.process()

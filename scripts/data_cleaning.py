import logging
from pathlib import Path

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load raw telecom churn dataset.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Dataset loaded successfully: {df.shape}")
        return df

    except Exception as e:
        logging.exception(f"Failed to load dataset: {e}")
        raise


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean TotalCharges column and remove invalid rows.
    """
    try:
        df["TotalCharges"] = (
            df["TotalCharges"]
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
        )

        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

        before_rows = len(df)

        df = df.dropna(subset=["TotalCharges"])

        after_rows = len(df)

        logging.info(
            f"Dropped {before_rows - after_rows} rows "
            f"with invalid TotalCharges values"
        )

        return df

    except Exception as e:
        logging.exception(f"Error cleaning TotalCharges: {e}")
        raise


def encode_churn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode churn column.
    """
    try:
        df["Churn"] = df["Churn"].map(
            {
                "No": 0,
                "Yes": 1
            }
        )

        logging.info("Churn encoded successfully")

        return df

    except Exception as e:
        logging.exception(f"Error encoding Churn: {e}")
        raise


def create_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create tenure groups for analysis.
    """
    try:
        bins = [0, 12, 24, 48, 72]
        labels = [
            "0-12 Months",
            "13-24 Months",
            "25-48 Months",
            "49-72 Months"
        ]

        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        logging.info("Tenure groups created")

        return df

    except Exception as e:
        logging.exception(f"Error creating tenure groups: {e}")
        raise


def save_cleaned_data(
    df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Save cleaned dataset.
    """
    try:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            output_path,
            index=False
        )

        logging.info(
            f"Cleaned dataset saved to: {output_path}"
        )

    except Exception as e:
        logging.exception(f"Error saving file: {e}")
        raise


def main() -> None:
    """
    Execute complete cleaning pipeline.
    """
    try:
        input_file = Path(
            "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )

        output_file = Path(
            "data/cleaned/telco_churn_cleaned.csv"
        )

        df = load_data(input_file)

        logging.info(
            f"Initial dataset shape: {df.shape}"
        )

        df = clean_total_charges(df)

        df = encode_churn(df)

        df = create_tenure_group(df)

        logging.info(
            f"Final dataset shape: {df.shape}"
        )

        save_cleaned_data(
            df,
            output_file
        )

        logging.info(
            "Data cleaning pipeline completed successfully"
        )

    except Exception as e:
        logging.exception(
            f"Pipeline failed: {e}"
        )


if __name__ == "__main__":
    main()
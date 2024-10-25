# data_utils.py
import pandas as pd
from sklearn.model_selection import train_test_split

def raw_taxi_df(filename: str) -> pd.DataFrame:
    """
    Loads raw taxi data from a Parquet file.

    Args:
        filename (str): Path to the Parquet file containing raw taxi data.

    Returns:
        pd.DataFrame: A DataFrame containing the raw taxi data.
    """
    return pd.read_parquet(path=filename)

def clean_taxi_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw taxi data by removing NaN values and trips that are too long.

    Assumptions:
    - We remove NaNs to ensure there are no missing values in the dataset.
    - Trips longer than 100 miles are likely outliers, so they are removed.

    Args:
        raw_df (pd.DataFrame): Raw taxi data DataFrame.

    Returns:
        pd.DataFrame: A cleaned DataFrame ready for further processing.
    """
    clean_df = raw_df.dropna()
    clean_df = clean_df[clean_df["trip_distance"] < 100]
    
    # Create a new column for time duration in minutes
    clean_df["time_deltas"] = clean_df["tpep_dropoff_datetime"] - clean_df["tpep_pickup_datetime"]
    clean_df["time_mins"] = pd.to_numeric(clean_df["time_deltas"]) / 6**10
    return clean_df

def split_taxi_data(clean_df: pd.DataFrame, 
                    x_columns: list[str], 
                    y_column: str, 
                    train_size: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into training and testing sets.

    Args:
        clean_df (pd.DataFrame): Cleaned taxi DataFrame.
        x_columns (list[str]): List of feature columns to use for training.
        y_column (str): Target column to predict.
        train_size (int): Number of rows to include in the training set.

    Returns:
        Tuple: Four DataFrames (X_train, X_test, y_train, y_test).
    """
    return train_test_split(clean_df[x_columns], clean_df[[y_column]], train_size=train_size)


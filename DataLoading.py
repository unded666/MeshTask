import requests
import zipfile
import pandas as pd
import os
import io

DATASET_SOURCE = './Data/study_performance.csv'

def load_data(path = DATASET_SOURCE) -> pd.DataFrame:
    """
    This function loads a dataset from a given path into a pandas DataFrame.

    Parameters:
    path (str): The path to the dataset file. By default, it uses the value of DATASET_SOURCE.

    Returns:
    pd.DataFrame: The loaded dataset as a pandas DataFrame.

    Raises:
    FileNotFoundError: If the file specified by path does not exist.
    """
    # Check if the file exists at the given path
    if not os.path.exists(path):
        # If the file does not exist, raise a FileNotFoundError
        raise FileNotFoundError(f"File not found at {path}")

    # If the file exists, load it into a pandas DataFrame and return it
    return pd.read_csv(path)


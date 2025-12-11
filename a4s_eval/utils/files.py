"""File handling utilities for A4S evaluation.

This module provides utilities for downloading, caching, and reading dataset and model
files. It handles file downloads from the API server and manages a local cache to
avoid redundant downloads.
"""

import os
import pandas as pd
from a4s_eval.utils import env
import requests

# Directory names for caching different types of files
DATASET_DIR = "datasets"
MODEL_DIR = "models"


def download_file(url: str, path: str) -> None:
    """Download a file from a URL and save it to the specified path.

    Args:
        url (str): The URL to download the file from
        path (str): The local path where the file should be saved

    Raises:
        requests.exceptions.RequestException: If the download fails
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # Check for errors
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)


def get_dataset_files(dataset_file: str) -> str:
    """Retrieve a dataset file, downloading it if not already cached.

    Args:
        dataset_file (str): Name of the dataset file to retrieve

    Returns:
        str: Local path to the dataset file

    Note:
        Files are cached in the CACHE_DIR/datasets directory
    """
    cache_dir = f"{env.CACHE_DIR}/{DATASET_DIR}"
    os.makedirs(cache_dir, exist_ok=True)

    url = f"{env.API_URL}/api/dataset_file?file_name={dataset_file}"
    file_path = f"{cache_dir}/{dataset_file}"
    if not os.path.exists(file_path):
        download_file(url, file_path)
    return file_path


def get_model_files(model_file: str) -> str:
    """Retrieve a model file, downloading it if not already cached.

    Args:
        model_file (str): Name of the model file to retrieve

    Returns:
        str: Local path to the model file

    Note:
        Files are cached in the CACHE_DIR/models directory
    """
    cache_dir = f"{env.CACHE_DIR}/{MODEL_DIR}"
    os.makedirs(cache_dir, exist_ok=True)

    url = f"{env.API_URL}/api/model_file?file_name={model_file}"
    file_path = f"{cache_dir}/{model_file}"
    if not os.path.exists(file_path):
        download_file(url, file_path)
    return file_path


def auto_read_dataset_file(dataset_file: str) -> pd.DataFrame:
    """Automatically read a dataset file based on its extension.

    Args:
        dataset_file (str): Path to the dataset file

    Returns:
        pd.DataFrame: The loaded dataset

    Raises:
        ValueError: If the file format is not supported
    """
    if dataset_file.endswith(".csv"):
        return pd.read_csv(dataset_file)
    elif dataset_file.endswith(".parquet"):
        return pd.read_parquet(dataset_file)
    else:
        raise ValueError(f"Unsupported file format: {dataset_file}")


def auto_get_read_dataset_file(dataset_file: str) -> pd.DataFrame:
    """Download (if needed) and read a dataset file in one operation.

    This is a convenience function that combines get_dataset_files() and
    auto_read_dataset_file() into a single operation.

    Args:
        dataset_file (str): Name of the dataset file to retrieve and read

    Returns:
        pd.DataFrame: The loaded dataset
    """
    file_path = get_dataset_files(dataset_file)
    return auto_read_dataset_file(file_path)

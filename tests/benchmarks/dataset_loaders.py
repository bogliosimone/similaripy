"""
Dataset loaders for benchmarking similaripy.

This module provides unified dataset loading functionality for various
recommendation datasets (MovieLens, Yandex Music Yambda, etc.).
"""

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import scipy.sparse as sp


# Dataset configuration
DATASET_DIR = Path("datasets_bench")

# MovieLens configurations
MOVIELENS_CONFIGS = {
    "25m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        "name": "ml-25m",
    },
    "32m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
        "name": "ml-32m",
    },
}

# Yambda configurations
YAMBDA_CONFIGS = {
    "50m": {
        "data_dir": "flat/50m",
        "data_files": "multi_event.parquet",
    },
    "500m": {
        "data_dir": "flat/500m",
        "data_files": "multi_event.parquet",
    },
}


def load_movielens(version="32m", verbose=True):
    """
    Load MovieLens dataset and convert to sparse URM (User Rating Matrix).

    Parameters
    ----------
    version : str, optional
        Dataset version: "25m" or "32m" (default: "32m")
    verbose : bool, optional
        Print progress information (default: True)

    Returns
    -------
    scipy.sparse.csr_matrix
        User-item rating matrix in CSR format (users x items)
    dict
        Dataset metadata with keys:
        - 'n_users': number of users
        - 'n_items': number of items
        - 'n_ratings': number of ratings
        - 'density': matrix density
        - 'user_id_map': mapping from original user IDs to indices
        - 'item_id_map': mapping from original movie IDs to indices
    """
    if version not in MOVIELENS_CONFIGS:
        raise ValueError(f"Unknown MovieLens version '{version}'. Available: {list(MOVIELENS_CONFIGS.keys())}")

    config = MOVIELENS_CONFIGS[version]
    dataset_path = DATASET_DIR / config["name"]

    # Download if needed
    _download_movielens(version, verbose)

    # Load ratings
    ratings_file = dataset_path / "ratings.csv"
    if not ratings_file.exists():
        raise FileNotFoundError(
            f"Ratings file not found at {ratings_file}. "
            f"Please ensure the dataset is properly downloaded."
        )

    if verbose:
        print(f"Loading MovieLens {version} from {ratings_file}...")

    df = pd.read_csv(ratings_file)

    if verbose:
        print(f"Loaded {len(df)} ratings")

    # Create mappings for user and item IDs
    unique_users = df['userId'].unique()
    unique_items = df['movieId'].unique()

    user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item_id_map = {item_id: idx for idx, item_id in enumerate(unique_items)}

    # Map IDs to indices
    df['user_idx'] = df['userId'].map(user_id_map)
    df['item_idx'] = df['movieId'].map(item_id_map)

    # Create sparse matrix
    n_users = len(unique_users)
    n_items = len(unique_items)

    if verbose:
        print(f"Creating sparse URM: {n_users} users x {n_items} items")

    URM = sp.csr_matrix(
        (df['rating'].values, (df['user_idx'].values, df['item_idx'].values)),
        shape=(n_users, n_items),
        dtype=np.float32
    )

    density = URM.nnz / (n_users * n_items)

    if verbose:
        print(f"URM shape: {URM.shape}")
        print(f"URM density: {density:.4%}")
        print(f"URM nnz: {URM.nnz}")

    metadata = {
        'n_users': n_users,
        'n_items': n_items,
        'n_ratings': URM.nnz,
        'density': density,
        'user_id_map': user_id_map,
        'item_id_map': item_id_map,
    }

    return URM, metadata


def load_yambda(version="50m", event_type="multi_event", verbose=True):
    """
    Load Yandex Music Yambda dataset and convert to sparse URM.

    Requires: pip install datasets pyarrow

    Parameters
    ----------
    version : str, optional
        Dataset version: "50m" or "500m" (default: "50m")
    event_type : str, optional
        Event type: "likes", "listens", or "multi_event" (default: "multi_event")
    verbose : bool, optional
        Print progress information (default: True)

    Returns
    -------
    scipy.sparse.csr_matrix
        User-item interaction matrix in CSR format (users x items)
    dict
        Dataset metadata with keys:
        - 'n_users': number of users
        - 'n_items': number of items
        - 'n_interactions': number of interactions
        - 'density': matrix density
        - 'user_id_map': mapping from original user IDs to indices
        - 'item_id_map': mapping from original item IDs to indices
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Yambda dataset requires the 'datasets' package. "
            "Install it with: pip install datasets pyarrow"
        )

    if version not in YAMBDA_CONFIGS:
        raise ValueError(f"Unknown Yambda version '{version}'. Available: {list(YAMBDA_CONFIGS.keys())}")

    config = YAMBDA_CONFIGS[version]
    data_files = f"{event_type}.parquet"

    if verbose:
        print(f"Loading Yandex Music Yambda {version} ({event_type})...")

    # Load from HuggingFace
    ds = load_dataset(
        "yandex/yambda",
        data_dir=config["data_dir"],
        data_files=data_files
    )
    df = ds["train"].to_pandas()

    if verbose:
        print(f"Loaded {len(df)} interactions")

    # Keep only necessary columns
    df = df[["uid", "item_id"]].astype({"uid": "int64", "item_id": "int64"})

    # Map raw IDs to contiguous indices using pandas Categorical
    u_cat = pd.Categorical(df["uid"])
    i_cat = pd.Categorical(df["item_id"])

    u_codes = u_cat.codes.astype(np.int64)
    i_codes = i_cat.codes.astype(np.int64)
    n_users = len(u_cat.categories)
    n_items = len(i_cat.categories)

    if verbose:
        print(f"Creating sparse URM: {n_users} users x {n_items} items")

    # Build COO -> CSR with implicit 1s
    data = np.ones(len(df), dtype=np.float32)
    URM = sp.coo_matrix((data, (u_codes, i_codes)), shape=(n_users, n_items)).tocsr()

    density = URM.nnz / (n_users * n_items)

    if verbose:
        print(f"URM shape: {URM.shape}")
        print(f"URM density: {density:.4%}")
        print(f"URM nnz: {URM.nnz}")

    # Create ID mappings
    user_id_map = dict(zip(u_cat.categories.astype(np.int64), range(n_users)))
    item_id_map = dict(zip(i_cat.categories.astype(np.int64), range(n_items)))

    metadata = {
        'n_users': n_users,
        'n_items': n_items,
        'n_interactions': URM.nnz,
        'density': density,
        'user_id_map': user_id_map,
        'item_id_map': item_id_map,
    }

    return URM, metadata


def _download_movielens(version, verbose=True):
    """Download and extract MovieLens dataset if not already present."""
    DATASET_DIR.mkdir(exist_ok=True)

    config = MOVIELENS_CONFIGS[version]
    dataset_path = DATASET_DIR / config["name"]

    if dataset_path.exists():
        if verbose:
            print(f"Dataset already exists at {dataset_path}")
        return

    zip_path = DATASET_DIR / f"{config['name']}.zip"

    if not zip_path.exists():
        if verbose:
            print(f"Downloading MovieLens {version} from {config['url']}...")
        urlretrieve(config['url'], zip_path)
        if verbose:
            print(f"Downloaded to {zip_path}")

    if verbose:
        print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATASET_DIR)
    if verbose:
        print(f"Extracted to {dataset_path}")


def load_URM(dataset, version=None, verbose=True, **kwargs):
    """
    Unified dataset loader that dispatches to the appropriate loader.

    Parameters
    ----------
    dataset : str
        Dataset name: "movielens" or "yambda"
    version : str, optional
        Dataset version (default depends on dataset):
        - MovieLens: "25m" (default) or "32m"
        - Yambda: "50m" (default) or "500m"
    verbose : bool, optional
        Print progress information (default: True)
    **kwargs : dict
        Additional dataset-specific arguments:
        - For Yambda: event_type ("likes", "listens", "multi_event")

    Returns
    -------
    scipy.sparse.csr_matrix
        User-item matrix in CSR format (users x items)
    dict
        Dataset metadata

    Examples
    --------
    >>> # Load MovieLens 25M
    >>> URM, meta = load_URM("movielens", version="25m")

    >>> # Load Yambda 50M with multi-event interactions
    >>> URM, meta = load_URM("yambda", version="50m", event_type="multi_event")
    """
    dataset = dataset.lower()

    if dataset == "movielens":
        version = version or "32m"
        return load_movielens(version=version, verbose=verbose)

    elif dataset == "yambda":
        version = version or "50m"
        event_type = kwargs.get("event_type", "multi_event")
        return load_yambda(version=version, event_type=event_type, verbose=verbose)

    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Available datasets: movielens, yambda"
        )

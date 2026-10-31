"""
Benchmark datasets: one users x items CSR (float32) per (dataset, version).

Real datasets are built once and cached as .npz in one directory shared by
every checkout and worktree (`SIMILARIPY_BENCH_DATA`, default
~/.cache/similaripy-bench), so every build, 0.7.0 included, loads the same
bytes and needs no pandas once the cache exists. Each cached URM is checked
against the shape and nnz recorded in EXPECTED: a changed source or loader
fails the load instead of silently timing other data.

Not named datasets.py: scripts run by path put this folder first on
sys.path, and that name would shadow Hugging Face's `datasets`.
"""

import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import scipy.sparse as sp

CACHE = Path(os.environ.get("SIMILARIPY_BENCH_DATA", Path.home() / ".cache" / "similaripy-bench"))

DEFAULT_VERSIONS = {"synthetic": "10k", "movielens": "32m", "yambda": "50m", "hm": "full"}

# (shape, nnz) of the URM as built below, by cache key
EXPECTED = {
    "movielens-32m": ((200948, 84432), 32000204),
    "yambda-50m-multi_event": ((10000, 934057), 13607096),
    "yambda-500m-multi_event": ((100000, 3004578), 137922244),
    "hm-full": ((1362281, 104547), 27306439),  # 31,788,324 purchases
}

MOVIELENS_URLS = {
    "25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
    "32m": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
}
YAMBDA_VERSIONS = ("50m", "500m")
YAMBDA_EVENTS = ("likes", "listens", "multi_event")
# H&M Personalized Fashion Recommendations (Kaggle competition data: download it yourself, after
# accepting the competition rules; numbers on it stay internal until the rules are checked, D75 Q1)
HM_CSV = CACHE / "h-and-m" / "transactions_train.csv"


def load(dataset, version=None, event_type="multi_event", verbose=True):
    """(URM, info): the users x items CSR and {dataset, version, key, shape, nnz}."""
    version = version or DEFAULT_VERSIONS[dataset]
    if dataset == "synthetic":
        if version != "10k":
            raise ValueError("synthetic has one version: 10k")
        key, URM = "synthetic-10k", make_urm()
    else:
        if dataset == "movielens":
            if version not in MOVIELENS_URLS:
                raise ValueError(f"movielens versions: {sorted(MOVIELENS_URLS)}")
            key, build = f"movielens-{version}", lambda: _movielens(version, verbose)
        elif dataset == "yambda":
            if version not in YAMBDA_VERSIONS or event_type not in YAMBDA_EVENTS:
                raise ValueError(f"yambda versions: {YAMBDA_VERSIONS}, event types: {YAMBDA_EVENTS}")
            key, build = f"yambda-{version}-{event_type}", lambda: _yambda(version, event_type, verbose)
        elif dataset == "hm":
            if version != "full":
                raise ValueError("hm has one version: full")
            key, build = "hm-full", lambda: _hm(verbose)
        else:
            raise ValueError(f"unknown dataset {dataset!r}: synthetic, movielens, yambda, hm")
        URM = _cached(key, build, verbose)
    info = {"dataset": dataset, "version": version, "key": key, "shape": list(URM.shape), "nnz": int(URM.nnz)}
    if dataset == "yambda":
        info["event_type"] = event_type
    return URM, info


def _cached(key, build, verbose):
    path = CACHE / f"{key}.npz"
    if path.exists():
        URM = sp.csr_array(sp.load_npz(path))
    else:
        URM = build()
        # One index dtype for every dataset: int32 while it fits (scipy's rule; pandas codes are int64 or int32)
        if URM.nnz <= np.iinfo(np.int32).max and max(URM.shape) <= np.iinfo(np.int32).max:
            URM.indices, URM.indptr = URM.indices.astype(np.int32), URM.indptr.astype(np.int32)
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        sp.save_npz(tmp, URM, compressed=False)
        tmp.replace(path)  # no half-written cache after an interrupt
        if verbose:
            print(f"cached {key} at {path}")
    expected = EXPECTED.get(key)
    if expected is not None and (tuple(URM.shape), URM.nnz) != expected:
        raise RuntimeError(
            f"{key}: shape {URM.shape} nnz {URM.nnz}, expected {expected}; "
            f"the source or the loader changed (delete {path} to rebuild after checking)"
        )
    return URM


def make_urm(n_users=10_000, n_items=10_000, nnz=1_000_000, zipf_a=0.8, seed=42):
    """Seeded users x items matrix with Zipf-like item popularity, ratings 1..5."""
    rng = np.random.default_rng(seed)
    popularity = 1.0 / np.arange(1, n_items + 1) ** zipf_a
    items = rng.choice(n_items, size=nnz, p=popularity / popularity.sum())
    # Shuffle item ids so popularity isn't already sorted by column
    items = rng.permutation(n_items)[items]
    users = rng.integers(0, n_users, size=nnz)
    ratings = rng.integers(1, 6, size=nnz).astype(np.float32)
    urm = sp.csr_array((ratings, (users, items)), shape=(n_users, n_items))
    urm.sum_duplicates()
    # int32 indices: 0.7.0 array-mode filter_cols/target_cols crash on int64
    urm.indices = urm.indices.astype(np.int32)
    urm.indptr = urm.indptr.astype(np.int32)
    return urm


def _movielens(version, verbose):
    import pandas as pd

    name = f"ml-{version}"
    ratings = CACHE / name / "ratings.csv"
    if not ratings.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        zip_path = CACHE / f"{name}.zip"
        if not zip_path.exists():
            if verbose:
                print(f"downloading {MOVIELENS_URLS[version]}")
            urlretrieve(MOVIELENS_URLS[version], zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(CACHE)
    df = pd.read_csv(ratings, usecols=["userId", "movieId", "rating"])
    # Ids numbered by first appearance
    users, _ = pd.factorize(df["userId"])
    items, _ = pd.factorize(df["movieId"])
    shape = (int(users.max()) + 1, int(items.max()) + 1)
    return sp.csr_array((df["rating"].to_numpy(), (users, items)), shape=shape, dtype=np.float32)


def _yambda(version, event_type, verbose):
    import pandas as pd

    data_dir, data_file = f"flat/{version}", f"{event_type}.parquet"
    # Local HF cache first: no network call (the Hub API can rate-limit or fail)
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("yandex/yambda", f"{data_dir}/{data_file}", repo_type="dataset", local_files_only=True)
        df = pd.read_parquet(path, columns=["uid", "item_id"])
    except (ImportError, FileNotFoundError):  # not cached yet: download
        from datasets import load_dataset

        if verbose:
            print(f"downloading yandex/yambda {data_dir}/{data_file}")
        ds = load_dataset("yandex/yambda", data_dir=data_dir, data_files=data_file)
        df = ds["train"].to_pandas()[["uid", "item_id"]]
    users = pd.Categorical(df["uid"].astype("int64"))
    items = pd.Categorical(df["item_id"].astype("int64"))
    data = np.ones(len(df), dtype=np.float32)  # implicit: repeated events sum
    shape = (len(users.categories), len(items.categories))
    return sp.coo_array((data, (users.codes, items.codes)), shape=shape).tocsr()


def _hm(verbose):
    """Purchases: users x articles, value = purchase count (repeated purchases summed)."""
    import pandas as pd

    from similaripy.preprocessing import interactions_to_urm

    if not HM_CSV.exists():
        raise FileNotFoundError(
            f"{HM_CSV} not found: download transactions_train.csv from the Kaggle competition "
            "'h-and-m-personalized-fashion-recommendations' (account + rules acceptance) and put it there"
        )
    if verbose:
        print(f"reading {HM_CSV}")
    df = pd.read_csv(HM_CSV, usecols=["customer_id", "article_id"], dtype={"customer_id": str, "article_id": "int64"})
    users, _ = pd.factorize(df["customer_id"])  # 64-char hex ids -> integer codes
    URM, _, _ = interactions_to_urm(users, df["article_id"].to_numpy())
    return sp.csr_array(URM, dtype=np.float32)

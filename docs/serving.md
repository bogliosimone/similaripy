# 🛎️ Serving

`ItemKNN` serves a similarity matrix S computed by any similarity function.

```python
from similaripy.serving import ItemKNN
```

## Offline → Save → Online

```python
import similaripy as sim
from similaripy.serving import ItemKNN

# offline: compute S, build the model, save it
S = sim.cosine(URM.T, k=100)
ItemKNN(S.T, item_ids=item_ids).save("models/v3")

# online, in each worker: load (mmap: one copy of S shared by every process)
model = ItemKNN.load("models/v3")
ids, scores = model.recommend(history, k=20)       # history: your item ids
```

`score[j] = sum_i w_i * S[i, j]` over the history items `i` (weights 1.0 by
default): the best `k` items with a positive score, best first. The model is
immutable: one instance can be shared by any number of threads.

## Methods

| Method | Description |
|--------|-------------|
| `ItemKNN(S)` | Builds the model from a square item-item matrix S. |
| `recommend(history)` | The top-k items for one history: `(ids, scores)`. |
| `recommend_batch(histories)` | `recommend` for a list of histories, in parallel: two lists, one array per user. |
| `view(include=, exclude=)` | The same model (S shared, no copy) with availability rules for every request. |
| `save(path)` / `ItemKNN.load(path)` | A saved model, memory-mapped by default. |
| `item_ids` | The id of each item (column of S). |

## Parameters

Everything after the first argument is keyword-only.

**`ItemKNN(S, ...)`**

| Parameter | Description |
|-----------|-------------|
| `S` | Square sparse item-item matrix, e.g. `sim.cosine(URM.T, k=100)` or its transpose (see [Which orientation of S](#which-orientation-of-s)). |
| `item_ids` | One unique int64 id per item (column of S); histories, results and rules use them. `None` = S's column indices. *(default: `None`)* |
| `candidate_items` | Only these items can ever be recommended; the others are dropped from the model (histories still use them). Changing it means a rebuild. *(default: `None` = all)* |
| `block_size` | As in the similarity functions: `'auto'`, `None` (no blocking) or an int > 0. Same results at any value; blocking is faster under concurrent requests, `None` for a lone request on an idle machine. *(default: `'auto'`)* |
| `num_threads` | Threads for the build. `0` = all cores. *(default: `0`)* |

**`recommend(history, ...)`, `recommend_batch(histories, ...)`**

| Parameter | Description |
|-----------|-------------|
| `history` | Item ids. Order doesn't matter; a duplicate adds its weight; ids the model doesn't have are ignored. In batch: a list with one array per user. |
| `weights` | One finite weight per history id. In batch: one entry per user (an entry may be `None`). *(default: `None` = all 1.0)* |
| `k` | Number of items wanted; fewer when fewer items score > 0. *(default: `10`)* |
| `exclude_seen` | Never recommend a history item. *(default: `True`)* |
| `include` | Only these items can be recommended. In batch: one entry per user. *(default: `None` = all)* |
| `exclude` | Never recommend these items. In batch: one entry per user. *(default: `None`)* |
| `sorted` | Best first; `False` skips the final sort (a few % faster at k in the thousands). *(default: `True`)* |
| `num_threads` | Batch only. `0` = all cores (one shared pool), `n` = `n` threads. *(default: `0`)* |

## Single and batch requests

`recommend` serves one user; `recommend_batch` serves a list of users in
parallel, and its element `u` equals `recommend(histories[u], ...)` bit for
bit. A rule shared by every user goes in a [view](#availability).

```python
ids, scores = model.recommend(history, weights=w, k=20)              # one user (online)
ids, scores = model.recommend_batch(histories, weights=ws, k=20)     # many users (offline)
for user, i, s in zip(users, ids, scores):
    cache[user] = (i, s)                                             # e.g. precomputed results
```

From an interaction log, results in your own ids: `pd.factorize` gives the
`item_ids` array (`item_ids[j]` = your id for column `j` of S), and the batch
output is already in those ids. In chunks when there are millions of users:

```python
user_codes, user_ids = pd.factorize(log.user)
item_codes, item_ids = pd.factorize(log.item)
URM = sps.csr_matrix((np.ones(len(log)), (user_codes, item_codes)))   # duplicates summed

model = ItemKNN(sim.cosine(URM.T, k=100).T, item_ids=item_ids)

histories = np.split(item_ids[URM.indices], URM.indptr[1:-1])   # your ids per user
weights = np.split(URM.data, URM.indptr[1:-1])
chunk = 100_000
for a in range(0, len(histories), chunk):
    ids, scores = model.recommend_batch(histories[a:a + chunk], weights=weights[a:a + chunk], k=20)
    pd.DataFrame({
        "user": np.repeat(user_ids[a:a + chunk], [len(i) for i in ids]),
        "item": np.concatenate(ids),
        "score": np.concatenate(scores),
    }).to_parquet(f"recs/part-{a // chunk}.parquet")   # one file per chunk
```

With `ItemKNN(S)`, the results equal `sim.dot_product(URM, S, k=k, filter_cols=URM)`
up to float rounding.

## Availability

Three levels, from permanent to per request. They combine with AND.

| Level | How | Cost | Use for |
|-------|-----|------|---------|
| Build | `ItemKNN(S, candidate_items=ids)` | rebuild to change | items that are never recommended in this model version (e.g. inactive items in a huge catalogue) |
| View | `model.view(include=ids, exclude=ids)` | one bit per item, S shared | rules shared by many requests that change during the day (region, stock, package) |
| Call | `model.recommend(h, include=ids, exclude=ids)` | proportional to the ids | per user or per request rules (e.g. items already shown) |

`include` = only these items can be recommended, `exclude` = never these.
A view of a view narrows: it keeps its parent's rules and adds its own.

```python
# Build, per model version: only the active items can be recommended
model = ItemKNN(S.T, item_ids=item_ids, candidate_items=active_items)

# View, per country: the items available in the region, minus those out of stock
by_country = {c: model.view(include=catalogue[c], exclude=out_of_stock) for c in countries}

# View of a view, per package: the country's items AND the package's (e.g. "movie + sport")
@functools.lru_cache(maxsize=None)
def view_for(country, package):
    return by_country[country].view(include=packages[package])

# Call, per user: the user's own rules
ids, scores = view_for(country, package).recommend(history, k=20, exclude=already_shown)
```

- To widen, start from the base model: `model.view(...)`; `view()` with no
  rules keeps the parent's.
- To update availability in a running service, build the new views and swap
  the references (here: rebuild `by_country`, then `view_for.cache_clear()`);
  requests in flight finish on the model they started with.
- Each view holds one bit per item (117 KB for 934k items): cache the views
  you use instead of building one per request.
- For items that are never available and S can be recomputed, prefer
  `target_cols` when computing S (see [Similarities](similarity.md#common-parameters)):
  each item then keeps `k` neighbours that can all be recommended, while
  `candidate_items` and the rules only remove neighbours from its top-k.

## Which orientation of S

`sim.cosine(URM.T, k=100)` keeps the top 100 items per row: row `i` holds item
`i`'s most similar items.

- `ItemKNN(S)` scores a candidate from the neighbours of each history item.
  Steadier latency: each history item adds at most `k` entries.
- `ItemKNN(S.T)` scores a candidate from its own neighbours in the history
  (the [Quickstart](guide.md#example)'s `dot_product(URM, S.T, filter_cols=URM)`).
  A row of `S.T` can be long: a popular item is in many top-k lists.

Which one scores better depends on the dataset and the similarity: evaluate
both. By similarity family:

| Family | Orientation |
|--------|-------------|
| Symmetric (`cosine`, `jaccard`, `dice`, `tversky` with `alpha = beta`, `dot_product`) | Both are valid models. In our tests `S.T` scored better, more so at larger `k`. |
| Graph (`p3alpha`, `rp3beta`) | Use `S`: `S.T` fixes the popularity penalty at `beta = 1` whatever `beta` you pass, so tune `beta` with `S`. |
| Asymmetric (`asymmetric_cosine`, `s_plus` and `tversky` with unequal weights) | Rows and columns differ by design: evaluate both. |

## In a threaded service

- Share one model across threads: `recommend` releases the GIL, so threads run
  requests in parallel (free-threaded Python too).
- `recommend_batch` uses `num_threads` as the similarity functions do.
- Long histories dominate the tail latency: cap them (e.g. the 1000 most
  recent items) if your p99 matters.

`recommend` latency by history cap, Python call, µs (p50 / p99, warm; S =
`cosine(k=100)`, `k=100`, history excluded; indicative, one Apple M2 Pro):

| History cap | MovieLens 32M (84k items) | Yambda 50M (934k items) |
|-------------|---------------------------|-------------------------|
| 50 | 10 / 17 | 26 / 37 |
| 200 | 13 / 30 | 73 / 91 |
| 1000 | 13 / 96 | 279 / 369 |
| none | 13 / 124 | 309 / 1904 |

## Saving and loading

`save(path)` writes a directory; `ItemKNN.load(path)` reads it back in this or
any later 1.x version, and checks it before use. The view's rules and
`block_size` are saved too.

| Option | Description |
|--------|-------------|
| `load(path, mmap=True)` | Maps the files read-only: one copy in the OS page cache, shared by every worker process, and fast to load. *(default)* |
| `load(path, mmap=False)` | Reads the arrays into memory, e.g. on a network file system (NFS). |
| `save(path, overwrite=False)` | Refuses a non-empty directory. *(default)* |
| `save(path, overwrite=True)` | Replaces a saved model: writes a new directory and renames it into place; processes that mapped the old files keep them. |

- **Never change a saved model's files in place**: a process mapping them
  would crash. For a live service, save each version to a new directory and
  switch what the workers load (a symlink or a config you flip).
- With `mmap=True`, per-process RSS counts the shared pages in every worker:
  the sum over-counts (on Linux, PSS in `/proc/<pid>/smaps_rollup` splits them).
- `overwrite=True` replaces only a directory holding a saved model's files
  (not a symlink). Between the two renames the directory is briefly missing
  (a `load` then fails); only one process may save to a path at a time. On
  Windows it fails with `PermissionError` while any process has the model
  mapped; the old model stays.
- Pickling works (same similaripy version only); pickling a mapped model
  copies its arrays into the pickle.
- The load check turns a corrupted file into an error, never an
  out-of-bounds read, but it doesn't guard against resource exhaustion (e.g.
  a `meta.json` asking for billions of items): load files you trust.

### Model file format

A saved model is a directory of 1-d little-endian `.npy` files (numpy's header
layout, as `np.save` writes it: `np.load` opens each one) and `meta.json`,
written last:

| File | Content |
|------|---------|
| `indptr.npy`, `indices.npy`, `data.npy` | S in internal ids (rows sorted), `data` float32 |
| `new_to_old.npy` | internal id → column of S |
| `item_ids.npy` | optional: your id per column of S |
| `allowed.npy` | optional: the view's rules, a uint64 bitset over internal ids |
| `meta.json` | format version and sizes |

```json
{
  "allowed": true,
  "block_size": "auto",
  "format": 1,
  "item_ids": true,
  "kind": "ItemKNN",
  "n_cols": 500,
  "n_items": 500,
  "nnz": 10000,
  "value_dtype": "float32",
  "written_by": "similaripy 1.0.0"
}
```

- `format`: a version loads every format up to its own; a newer format is a
  clear "upgrade similaripy" error.
- `value_dtype`: the type of `data` (`float32` today). It leaves room for
  quantized models (e.g. bfloat16) in a later format, with the same error on
  versions that don't know it.
- Internal ids order items by popularity (entries in their column of S); items
  with neither a row nor a column in S are not stored.

## In an evaluation framework

There is no `fit`: computing S (any similarity, any parameters) and serving it
are separate steps. A framework that wants `fit` / `recommend` needs a few
lines:

```python
class ItemKNNRecommender:
    def __init__(self, similarity=sim.cosine, **params):
        self.similarity, self.params = similarity, params

    def fit(self, URM):
        self.model = ItemKNN(self.similarity(URM.T, **self.params).T)
        return self

    def recommend(self, URM, k=10):
        histories = np.split(URM.indices, URM.indptr[1:-1])
        weights = np.split(URM.data, URM.indptr[1:-1])
        return self.model.recommend_batch(histories, weights=weights, k=k)   # (ids, scores) per user


rec = ItemKNNRecommender(sim.cosine, k=100, shrink=10).fit(URM_train)
ids, scores = rec.recommend(URM_train, k=20)
```

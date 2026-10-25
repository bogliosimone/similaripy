# 🛎️ Serving

`ItemKNN` serves an item-item model online: you compute the similarity matrix
S offline with any similarity function, build the model once, then ask for the
top-k items of one history (`recommend`) or of many at once
(`recommend_batch`). Everything is in your item ids. Requests run in Rust
without the GIL, on the calling thread.

```python
from similaripy.serving import ItemKNN
```

## Offline → Save → Online

```python
import similaripy as sim
from similaripy.serving import ItemKNN

# offline: compute S, build the model, save it
S = sim.rp3beta(URM.T, k=100)
ItemKNN(S, item_ids=item_ids).save("models/v3")

# online, in each worker: load (mmap: one copy of S shared by every process)
model = ItemKNN.load("models/v3")
ids, scores = model.recommend(history, k=20)       # history: item ids
ids, scores = model.recommend(history, weights=w, k=20)
```

- `item_ids` gives each column of S your own int64 id; histories, results
  and rules use them (`model.item_ids` returns them). Without it the ids are
  S's column indices.
- `score[j] = sum_i w_i * S[i, j]` over the history items `i` (weights 1.0 by
  default): the best `k` items with a positive score, best first
  (`sorted=False` skips the final sort). History items are excluded
  (`exclude_seen=True`). The history's order doesn't change the result (a
  duplicate id adds its weight); ids the model doesn't have are ignored.
- The model is immutable: it never changes after the build, so one instance
  can be shared by any number of threads.

### Many histories at once

`recommend_batch` takes a list of histories and returns two lists, in
parallel: element `u` equals `recommend(histories[u], ...)` bit for bit.
`weights`, `include` and `exclude` take one entry per user (an entry may be
`None`); the other options apply to every user.

```python
ids, scores = model.recommend_batch(histories, weights=weights, k=20)
for user, (i, s) in zip(users, zip(ids, scores)):
    cache[user] = (i, s)                       # e.g. precomputed recommendations
```

From a user-item matrix (rows = users, columns = S's columns), and in chunks
when there are millions of users:

```python
histories = np.split(item_ids[URM.indices], URM.indptr[1:-1])   # item_ids[...]: your ids
weights = np.split(URM.data, URM.indptr[1:-1])
for a in range(0, len(histories), 100_000):
    ids, scores = model.recommend_batch(histories[a:a + 100_000], weights=weights[a:a + 100_000], k=20)
```

The results equal `sim.dot_product(URM, S, k=k, filter_cols=URM)` up to float
rounding.

### Which orientation of S

`sim.cosine(URM.T, k=100)` keeps the top 100 items per row: row `i` holds item
`i`'s most similar items. `ItemKNN(S)` scores a candidate from the neighbours
of each history item; `ItemKNN(S.T)` from the candidate's own neighbours in
the history (the [Quickstart](guide.md#example)'s
`dot_product(URM, S.T, filter_cols=URM)`). Which one scores better depends on
the dataset and the similarity: evaluate both. `ItemKNN(S)` has steadier
latency: each history item adds at most `k` entries, while a row of `S.T` can
be long (a popular item is in many top-k lists).

### In a fit / recommend evaluation loop

There is no `fit`: computing S (any similarity, any parameters) and serving it
are separate steps. A framework that wants `fit` / `recommend` needs a few
lines:

```python
class ItemKNNRecommender:
    def __init__(self, similarity=sim.cosine, **params):
        self.similarity, self.params = similarity, params

    def fit(self, URM):
        self.model = ItemKNN(self.similarity(URM.T, **self.params))
        return self

    def recommend(self, URM, k=10):
        histories = np.split(URM.indices, URM.indptr[1:-1])
        weights = np.split(URM.data, URM.indptr[1:-1])
        return self.model.recommend_batch(histories, weights=weights, k=k)   # (ids, scores) per user


rec = ItemKNNRecommender(sim.rp3beta, k=100, alpha=0.8, beta=0.3).fit(URM_train)
ids, scores = rec.recommend(URM_train, k=20)
```

## In a threaded service

- Share one model across threads. `recommend` releases the GIL, so threads
  run requests in parallel (free-threaded Python too); each thread keeps its
  own scratch buffers between calls.
- `recommend_batch` uses `num_threads` as the similarity functions do: `0`
  (default) = one shared pool, `n` = a pool of `n` threads per calling thread.
- `block_size` (constructor, default `'auto'`) gives the same results at any
  value: blocking is faster when many requests run at once, `None` (no
  blocking) for a lone request on an idle machine.
- Long histories dominate the tail latency: cap them (e.g. the 1000 most
  recent items) if your p99 matters.

`recommend` latency by history cap, Python call, µs (p50 / p99, warm; S =
`cosine(k=100)`, `k=100`, history excluded; indicative, one Apple M2 Pro):

| History cap | MovieLens 32M (84k items) | Yambda 50M (934k items) |
|---|---|---|
| 50 | 10 / 17 | 26 / 37 |
| 200 | 13 / 30 | 73 / 91 |
| 1000 | 13 / 96 | 279 / 369 |
| none | 13 / 124 | 309 / 1904 |

## Availability

Three levels, from permanent to per request. They combine with AND.

| Level | How | Cost | Use for |
|---|---|---|---|
| Build | `ItemKNN(S, candidate_items=ids)` | rebuild to change | items that are never recommended (e.g. a huge catalogue with many dead items): they are dropped from the model, histories still use them |
| View | `model.view(include=ids, exclude=ids)` | one bit per item, S shared | rules shared by every request that change during the day (stock, region) |
| Call | `model.recommend(h, include=ids, exclude=ids)` | proportional to the ids | per user or per request rules (e.g. items already shown) |

`include` = only these items can be recommended, `exclude` = never these.

```python
it = model.view(include=available["IT"])                  # shares S with model
ids, scores = it.recommend(history, k=20, exclude=already_shown)
```

- A view of a view replaces both rules (`view()` = no rules).
- To update availability in a running service, build the new view and swap
  the reference; requests in flight finish on the model they started with:

    ```python
    state["model"] = base.view(include=available_ids)
    ```

- For items that are never available and S can be recomputed, prefer
  `target_cols` when computing S (see [Similarities](similarity.md#common-parameters)):
  each item then keeps `k` neighbours that can all be recommended, while
  `candidate_items` and the rules only remove neighbours from the top-k it
  already has.

## Saving and loading

`save(path)` writes a directory; `ItemKNN.load(path)` reads it back in this or
any later 1.x version, and checks it before use.

- **`mmap=True` (default)** maps the files read-only: they stay in the OS page
  cache, shared by every process that loads them, so N workers hold one copy
  of S, and loading is fast. Per-process RSS counts the shared pages in every
  worker: the sum over-counts (on Linux, PSS in `/proc/<pid>/smaps_rollup`
  splits them).
- **`mmap=False`** reads the arrays into memory: use it on a network file
  system (NFS).
- **Never change a saved model's files in place**: a process mapping them
  would crash. For a live service, save each version to a new directory and
  switch what the workers load (a symlink or a config you flip).
- `save` refuses a non-empty directory. `overwrite=True` replaces a saved
  model (a directory holding only a saved model's files, not a symlink) by
  writing a new directory and renaming it into place: processes that mapped
  the old files keep them. Between the two renames the directory is briefly
  missing (a `load` then fails), only one process may save to a path at a
  time, and nothing else may write into the directory meanwhile. On Windows
  the replace fails with `PermissionError` while any process has the model
  loaded with `mmap=True`; the old model stays.
- What is saved: the view's rules and `block_size` as given.
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
|---|---|
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

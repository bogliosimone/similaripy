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

Everything is on the `ItemKNN` class: build a `model`, then call its methods.

| Method | Description |
|--------|-------------|
| `model = ItemKNN(S)` | Constructor: builds the model from a square item-item matrix S. |
| `model.recommend(history)` | The top-k items for one history: `(ids, scores)`. |
| `model.recommend_batch(histories)` | `recommend` for a list of histories, in parallel: two lists, one array per user. |
| `model.view(include=, exclude=)` | The same model (S shared, no copy) with availability rules for every request. |
| `model.save(path)` | Writes the model to a directory. |
| `model = ItemKNN.load(path)` | Class method: a saved model, memory-mapped by default. |
| `model.item_ids` | Property: the id of each item (column of S). |

## Parameters

**`ItemKNN(S, ...)`**

| Parameter | Description |
|-----------|-------------|
| `S` | Square sparse item-item matrix, e.g. `sim.cosine(URM.T, k=100)` or its transpose (see [Orientation of S](#orientation-of-s)). |
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

## Single and batch recommendation

`recommend` serves one user; `recommend_batch` serves a list of users in parallel.

```python
ids, scores = model.recommend(history, weights=w, k=20)              # one user (online)
ids, scores = model.recommend_batch(histories, weights=ws, k=20)     # many users (offline)
```

Example: batch recommendations from an interaction log, in chunks for millions of users:

```python
from similaripy.preprocessing import from_interactions, to_histories

URM, user_ids, item_ids = from_interactions(log.user, log.item)   # repeats summed
model = ItemKNN(sim.cosine(URM.T, k=100).T, item_ids=item_ids)

histories, weights = to_histories(URM, item_ids)   # your ids per user
chunk = 100_000
for a in range(0, len(histories), chunk):
    ids, scores = model.recommend_batch(histories[a:a + chunk], weights=weights[a:a + chunk], k=20)
    pd.DataFrame({
        "user": np.repeat(user_ids[a:a + chunk], [len(i) for i in ids]),
        "item": np.concatenate(ids),
        "score": np.concatenate(scores),
    }).to_parquet(f"recs/part-{a // chunk}.parquet")   # one file per chunk
```

## Availability

Three levels, from permanent to per request, combined with **AND**:

| Level | How | Cost | Use for |
|-------|-----|------|---------|
| Build | `ItemKNN(S, candidate_items=ids)` | rebuild to change | items that are never recommended in this model version (e.g. inactive items in a huge catalogue) |
| View | `model.view(include=ids, exclude=ids)` | one bit per item, S shared | rules shared by many requests that change during the day (region, stock, package) |
| Call | `model.recommend(h, include=ids, exclude=ids)` | proportional to the ids | per user or per request rules (e.g. items shown to the user many times) |

`include` = only these items can be recommended, `exclude` = never these.

Example:

```python
# Build, per model version: only the active items can be recommended
model = ItemKNN(S.T, item_ids=item_ids, candidate_items=active_items)

# View, per country: the items available in the region, minus those out of stock
by_country = {c: model.view(include=catalogue[c], exclude=out_of_stock) for c in countries}

# View of a view, per package: the country's items AND the package's (e.g. "movie + sport")
@functools.lru_cache(maxsize=None)
def view_for(country, package):
    return by_country[country].view(include=packages[package])

# Call, per user: e.g. items shown many times and never clicked (impressions)
ids, scores = view_for(country, package).recommend(history, k=20, exclude=ignored_impressions)
```

- A view of a view narrows: it keeps its parent's rules and adds its own.
  To widen, start from the base model.
- To update availability in a running service, build the new views and swap
  the references. Requests in flight finish on the model they started with.
- Each view holds one bit per item (e.g. 117 KB for 934k items): cheap, you can
  cache the views you use instead of building one per request.
- **Tip:** for items that are never available, if S can be recomputed, pass
  `target_cols` when computing S: each item then keeps `k` neighbours that can
  all be recommended, while `candidate_items` and the rules only remove
  neighbours from its top-k.

## Saving and loading

`save(path)` writes a directory; `ItemKNN.load(path)` reads it back and checks
it before use. A view can't be saved: save the base
model and build views after `load`.

| Option | Description |
|--------|-------------|
| `load(path, mmap=True)` | Maps the files read-only: one copy shared by every worker process. *(default)* |
| `load(path, mmap=False)` | Reads the files into memory, e.g. on a network file system (NFS). |
| `save(path, overwrite=False)` | Refuses a non-empty directory. *(default)* |
| `save(path, overwrite=True)` | Replaces a saved model; processes that loaded the old one keep it. |

- **Never change a saved model's files in place**: save each version to a new
  directory and switch what the workers load.
- **Load files you trust**: a corrupted file is an error, but sizes aren't capped.

A saved model is a plain directory of `.npy` files plus `meta.json`, readable
with `np.load`. A model saved by a newer format is a clear "upgrade
similaripy" error.

## Orientation of S

A similarity matrix such as `sim.cosine(URM.T, k=100)` keeps the top 100 items per row: row `i` holds item `i`'s most similar items. It can be served in two ways:

- `ItemKNN(S)` scores a candidate from the neighbours of each history item.
  Steady latency: each history item adds at most `k` entries.
- `ItemKNN(S.T)` scores a candidate from its own neighbours in the history.
  A row of `S.T` can be long: a popular item is in many top-k lists.

Which one scores better depends on the dataset and the similarity: evaluate
both.

A few tips based on the similarity family:

| Family | Orientation |
|--------|-------------|
| Symmetric (`cosine`, `jaccard`, `dice`, `dot_product`) | Both are valid models. In our tests `S.T` scored better, more so at larger `k`. |
| Asymmetric (`asymmetric_cosine`, `tversky`, `s_plus`) | Transposing swaps the weights: `asymmetric_cosine(alpha).T` scores as `asymmetric_cosine(1 - alpha)`, `tversky(alpha, beta).T` as `tversky(beta, alpha)`, `s_plus` swaps `t1, t2` and `c1, c2`: both valid, tune the weights in the one you pick. |
| Graph (`p3alpha`, `rp3beta`) | Prefer `S`: with `alpha = 1`, `S.T` scores candidates like `rp3beta` with `beta = 1` whatever `beta` you pass (only the history weights change). Tune `beta` with `S`. |

## Use cases

### Threaded service

When serving the model in a threaded service:

- Share one model across threads: `recommend` releases the GIL, so threads
  run requests in parallel (free-threaded Python too). Across worker
  processes, `load` with `mmap=True` shares one copy of S.
- Long histories dominate the tail latency: cap them if your p99 matters
  (e.g. the 1000 most recent items).

Example of `recommend` latency by history cap:

| History cap | MovieLens 32M (84k items) | Yambda 50M (934k items) |
|-------------|---------------------------|-------------------------|
| 50 | 10 µs / 17 µs | 26 µs / 37 µs |
| 200 | 13 µs / 30 µs | 73 µs / 91 µs |
| 1000 | 13 µs / 96 µs | 279 µs / 369 µs |
| none | 13 µs / 124 µs | 309 µs / 1.9 ms |

p50 / p99 of the Python call, warm; S = `cosine(k=100)`, `k=100`, history
excluded; indicative, one Apple M2 Pro.

### Evaluation framework

A framework that wants `fit` / `recommend` needs a few lines:

```python
class ItemKNNRecommender:
    def __init__(self, similarity=sim.cosine, transpose=True, **params):
        # transpose: serve S.T; (see Orientation of S)
        self.similarity, self.transpose, self.params = similarity, transpose, params

    def fit(self, URM):
        S = self.similarity(URM.T, **self.params)
        self.model = ItemKNN(S.T if self.transpose else S)
        return self

    def recommend(self, URM, k=10):
        histories = np.split(URM.indices, URM.indptr[1:-1])
        weights = np.split(URM.data, URM.indptr[1:-1])
        return self.model.recommend_batch(histories, weights=weights, k=k)   # (ids, scores) per user


rec = ItemKNNRecommender(sim.cosine, k=100, shrink=10).fit(URM_train)
ids, scores = rec.recommend(URM_train, k=20)
```

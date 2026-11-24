# 🛎️ Serving

`SparseKNN` is the item-KNN serving engine: it serves a similarity matrix S
computed by any similarity function.

```python
from similaripy.serving import SparseKNN
```

## Offline → Save → Online

```python
import similaripy as sim
from similaripy.serving import SparseKNN

# offline: compute S, build the model, save it
S = sim.cosine(URM.T, k=100)
SparseKNN(S.T, item_ids=item_ids).save("models/v3")

# online, in each worker: load (mmap: one copy of S shared by every process)
model = SparseKNN.load("models/v3")
ids, scores = model.recommend(history, k=20)       # history: your item ids; any order (sorted=True: best first)
```

`score[j] = sum_i w_i * S[i, j]` over the history items `i` (weights 1.0 by
default): the best `k` items with a positive score, in no particular order
(`sorted=True`: best first; candidate generation doesn't need the sort, a
ranker re-scores them). The model is
immutable: one instance can be shared by any number of threads.

## Methods

Everything is on the `SparseKNN` class: build a `model`, then call its methods.

| Method | Description |
|--------|-------------|
| `model = SparseKNN(S)` | Constructor: builds the model from a square item-item matrix S. |
| `model.recommend(history)` | The top-k items for one history: `(ids, scores)`. |
| `model.recommend_batch(histories)` | `recommend` for a list of histories, in parallel: two lists, one array per user. |
| `model.score(history, items)` | The scores of the given items for one history, in your order: e.g. ranking features. |
| `model.score_batch(histories, items)` | `score` for a list of histories, in parallel: one array per user. |
| `model.view(include=, exclude=)` | The same model (S shared, no copy) with availability rules for every request. |
| `model.save(path)` | Writes the model to a directory. |
| `model = SparseKNN.load(path)` | Class method: a saved model, memory-mapped by default. |
| `model.item_ids` | Property: the id of each item (column of S). |

## Parameters

**`SparseKNN(S, ...)`**

| Parameter | Description |
|-----------|-------------|
| `S` | Square sparse item-item matrix, e.g. `sim.cosine(URM.T, k=100)` or its transpose (see [Orientation of S](#orientation-of-s)). |
| `item_ids` | One unique int64 id per item (column of S); histories, results and rules use them. `None` = S's column indices. String ids: [Ids](preprocessing.md#ids). *(default: `None`)* |
| `candidate_items` | Only these items can ever be recommended; the others are dropped from the model (histories still use them). The model is laid out for the candidates only: its score array covers just them. Changing it means a rebuild. *(default: `None` = all)* |
| `item_order` | How items are laid out in memory: speed only, the results are the same up to the order of exactly tied scores and float rounding in the last digits. `'auto'` = the library's choice for recommender catalogues (today `'clustered'`; it may change between versions). `'clustered'`: the most popular items first, the rest grouped by their most popular neighbour (fewer cache misses on large catalogues: −8..−17% latency on a 3M-item catalogue, neutral at 85k). `'popularity'`: the items with the most entries in their column of S first. `'original'`: S's own column order (usually slower; can help when S's ids already group similar items). Exactly tied scores follow the layout: under `'clustered'` the more popular item wins among the most popular ones, otherwise the order is unspecified. Fixed at build, saved with the model. *(default: `'auto'`)* |
| `block_size` | As in the similarity functions: `'auto'`, `None` (no blocking) or an int > 0. Same results at any value. `'auto'` sizes a single request for a busy server (every core sharing its L2 cache at work): blocking is faster under concurrent requests, `None` for a lone request on an idle machine (e.g. one request at a time per process). A setting of this process, not of the model: pickled, not saved (`load` takes its own). *(default: `'auto'`)* |
| `num_threads` | Threads for the build. `0` = all cores. *(default: `0`)* |

**`recommend(history, ...)`, `recommend_batch(histories, ...)`**

| Parameter | Description |
|-----------|-------------|
| `history` | Item ids. Order doesn't matter; a duplicate adds its weight; ids the model doesn't have are ignored. In batch: a list with one array per user. |
| `weights` | One finite weight per history id. In batch: one entry per user (an entry may be `None`). *(default: `None` = all 1.0)* |
| `k` | Number of items wanted; fewer when fewer items score > 0. *(default: `10`)* |
| `exclude_history` | Never recommend a history item. *(default: `True`)* |
| `include` | Only these items can be recommended. In batch: one entry per user. *(default: `None` = all)* |
| `exclude` | Never recommend these items. In batch: one entry per user. *(default: `None`)* |
| `views` | More views of the same base model (`model.view(...)`), at most 8: an item must be allowed by every one too (AND). In batch: for every user. See [Availability](#availability). *(default: `None`)* |
| `sorted` | `True`: best first, e.g. to show the first N. `False` skips the final sort (a few % faster at k in the thousands): the same items in an unspecified order, which can differ between calls. *(default: `False`, as the server's: retrieval)* |
| `num_threads` | Batch only. `0` = all cores (one shared pool), `n` = `n` threads. *(default: `0`)* |

**`score(history, items, ...)`, `score_batch(histories, items, ...)`**

| Parameter | Description |
|-----------|-------------|
| `history`, `weights`, `views`, `num_threads` | As in `recommend` / `recommend_batch`; an item a view excludes scores 0.0. |
| `items` | The item ids to score; duplicates are scored again. In batch: a list with one array per user. |

## Single and batch recommendation

`recommend` serves one user; `recommend_batch` serves a list of users in parallel.

```python
ids, scores = model.recommend(history, weights=w, k=20)              # one user (online)
ids, scores = model.recommend_batch(histories, weights=ws, k=20)     # many users (offline)
```

A negative weight is a dislike: each item's score is the weighted sum of its
similarities to the history, so a dislike pushes away what is similar to that
item, and items whose score ends at 0 or below are not returned. The disliked
item stays in the history (`exclude_history` keeps it out).

```python
ids, scores = model.recommend([10, 20], weights=[1, -1], k=3)   # likes 10, dislikes 20
```

Example: batch recommendations from an interaction log, in chunks for millions of users:

```python
from similaripy.preprocessing import interactions_to_urm, urm_to_histories

URM, user_ids, item_ids = interactions_to_urm(log.user, log.item)   # repeats summed
model = SparseKNN(sim.cosine(URM.T, k=100).T, item_ids=item_ids)

histories, weights = urm_to_histories(URM, item_ids=item_ids)   # your ids per user
chunk = 100_000
for a in range(0, len(histories), chunk):
    ids, scores = model.recommend_batch(histories[a:a + chunk], weights=weights[a:a + chunk], k=20)
    pd.DataFrame({
        "user": np.repeat(user_ids[a:a + chunk], [len(i) for i in ids]),
        "item": np.concatenate(ids),
        "score": np.concatenate(scores),
    }).to_parquet(f"recs/part-{a // chunk}.parquet")   # one file per chunk
```

## Scoring given items (ranking features)

`score` returns `sum_i w_i * S[i, j]` for each given item `j`, aligned with
`items`: no top-k, no threshold, history items included, zero and negative
scores as they are. The values are the same bits as `recommend`'s scores. Use
it when another stage picks the candidates, e.g. as a feature for a ranker or to
re-score the candidates of other retrievers.

```python
feat = model.score(history, candidates, weights=w)                      # one user (online)
feats = model.score_batch(histories, candidates_per_user, weights=ws)   # many users (offline)

# 0.0 means no evidence, an unknown item or one outside the model's availability;
# to tell unknown items apart (e.g. a missing value for a tree ranker):
feat = np.where(np.isin(candidates, model.item_ids), feat, np.nan)
```

Availability at the model level holds: items outside `candidate_items` or
outside a view's rules score 0.0. There is no per-call `include` / `exclude`.
Compute the training features on the same view you serve. Cost: about a
`recommend` call without the top-k (the accumulation over the history
dominates).

## Availability

Three levels, from permanent to per request, combined with **AND**:

| Level | How | Cost | Use for |
|-------|-----|------|---------|
| Build | `SparseKNN(S, candidate_items=ids)` | rebuild to change | items that are never recommended in this model version (e.g. inactive items in a huge catalogue) |
| View | `model.view(include=ids, exclude=ids)` | one bit per item, S shared | rules shared by many requests that change during the day (region, stock, package) |
| Call | `model.recommend(h, include=ids, exclude=ids)` | proportional to the ids | per user or per request rules (e.g. items shown to the user many times) |

`include` = only these items can be recommended, `exclude` = never these.

Example:

```python
# Build, per model version: only the active items can be recommended
model = SparseKNN(S.T, item_ids=item_ids, candidate_items=active_items)

# View, per country: the items available in the region, minus those out of stock
by_country = {c: model.view(include=catalogue[c], exclude=out_of_stock) for c in countries}

# View of a view, per package: the country's items AND the package's (e.g. "movie + sport")
@functools.lru_cache(maxsize=None)
def view_for(country, package):
    return by_country[country].view(include=packages[package])

# Call, per user: e.g. items shown many times and never clicked (impressions)
ids, scores = view_for(country, package).recommend(history, k=20, exclude=ignored_impressions)

# Or combine views per call, with no view per combination: the same items
by_package = {p: model.view(include=packages[p]) for p in packages}
ids, scores = by_country[country].recommend(history, k=20, views=[by_package[package]])
```

- A view of a view narrows: it keeps its parent's rules and adds its own.
  To widen, start from the base model.
- `views=` combines views per call (at most 8, all of the same base model).
  Two masks in all (the served view's and one in `views`, or two in `views`
  on the base model) cost what one view costs; three or more cost +15..+50%
  per request. For a combination served often, build it once as a view of a view.
- To update availability in a running service, build the new views and swap
  the references. Requests in flight finish on the model they started with.
- Each view holds one bit per item (e.g. 117 KB for 934k items): cheap, you can
  cache the views you use instead of building one per request.
- A trimmed model and a view with the same items give the same results up to
  the order of exactly tied scores and float rounding: the trimmed model has
  its own layout.
- **Tip:** for items that are never available, if S can be recomputed, pass
  `target_cols` when computing S: each item then keeps `k` neighbours that can
  all be recommended, while `candidate_items` and the rules only remove
  neighbours from its top-k.

## Saving and loading

`save(path)` writes a directory; `SparseKNN.load(path)` reads it back and checks
it before use. A view can't be saved: save the base
model and build views after `load`.

| Option | Description |
|--------|-------------|
| `load(path, mmap=True)` | Maps the files read-only: one copy shared by every worker process. *(default)* |
| `load(path, mmap=False)` | Reads the files into memory, e.g. on a network file system (NFS). |
| `load(path, block_size=...)` | As `SparseKNN`'s `block_size`: how this process runs requests, so not saved. *(default: `'auto'`)* |
| `save(path)` | Writes into a new (or empty) directory; refuses a non-empty one. |

- **A saved model is never changed**: save each version to its own directory
  (e.g. `models/v4`) and switch what the workers load (a config value, or a
  `current` symlink repointed by your deploy). With `mmap=True`, changing or
  truncating a file that a process has loaded is undefined behaviour (a crash
  or wrong results), not an error; `mmap=False` reads the files once.
- **Load files you trust**: a corrupted file is an error, but sizes aren't capped.
- **Pickle vs save/load**: a `SparseKNN` pickles (multiprocessing, joblib, Ray),
  but only to the same similaripy version, and every pickle is a full copy of S.
  For storage, deploys and big models, pass the path and `SparseKNN.load` it in
  each worker: one memory-mapped copy shared by all of them.

A saved model is a plain directory of `.npy` files plus `meta.json`, readable
with `np.load` ([format spec](saved-format.md)). The model is saved as built: its `item_order` and
`candidate_items` come with it (`load` has no such options). A model saved by a newer format is a clear "upgrade
similaripy" error.

## Orientation of S

A similarity matrix such as `sim.cosine(URM.T, k=100)` keeps the top 100 items per row: row `i` holds item `i`'s most similar items. It can be served in two ways:

- `SparseKNN(S)` scores a candidate from the neighbours of each history item.
  Steady latency: each history item adds at most `k` entries.
- `SparseKNN(S.T)` scores a candidate from its own neighbours in the history.
  A row of `S.T` can be long: a popular item is in many top-k lists.

Which one scores better depends on the dataset and the similarity: evaluate
both.

A few tips based on the similarity family:

| Family | Orientation |
|--------|-------------|
| Symmetric (`cosine`, `jaccard`, `dice`, `dot_product`) | Both are valid models. In our tests `S.T` scored better, more so at larger `k`. |
| Asymmetric (`asymmetric_cosine`, `tversky`, `s_plus`) | Transposing swaps the weights: `asymmetric_cosine(alpha).T` scores as `asymmetric_cosine(1 - alpha)`, `tversky(alpha, beta).T` as `tversky(beta, alpha)`, `s_plus` swaps `t1, t2` and `c1, c2`: both valid, tune the weights in the one you pick. |
| Graph (`p3alpha`, `rp3beta`) | Prefer `S`: with `alpha = 1`, `S.T` scores candidates like `rp3beta` with `beta = 1` whatever `beta` you pass (only the history weights change). Tune `beta` with `S`. |

## Beyond items

Nothing in `SparseKNN` is specific to items: it serves any square sparse KNN
graph over one set of ids. The "items" are the graph's nodes, a history is a
set of nodes, and `recommend` returns the nodes closest to it.

- **Lookalike users**: `SparseKNN(sim.cosine(URM, k=100), item_ids=user_ids)`
  (user-user); the history is a seed audience, the answer its lookalikes.
- **More like this**: `SparseKNN(sim.cosine(sim.normalization.bm25(DTM), k=50), item_ids=doc_ids)`
  over a document-term matrix (one row per document); the history is one
  document, or the ones a user read.

`item_ids` and `candidate_items` then name the nodes (users, documents).

## Use cases

### In production

A server that answers many users, one `recommend` per request:

- **Share one model across threads.** `recommend` releases the GIL, so threads
  run requests in parallel (free-threaded Python too). Across worker
  processes, `load` with `mmap=True` shares one copy of S.
- **Forked workers are fine** (gunicorn `--preload`, Celery prefork,
  `multiprocessing` with `fork`): load or compute in the parent, then fork.
  A forked child gets new threads for parallel calls (thread pools don't
  survive `fork()`). Fork while no similaripy call is running in another
  thread: a call in flight at the fork can't finish in the child.
- **Keep `block_size='auto'`** for a server with concurrent requests: a single
  request sizes its work for every core being busy. A process that runs one
  request at a time on an idle machine can use `block_size=None`.
- **Short histories are cheap.** A request costs a few µs of fixed work plus a
  little per history item (table below): cap long histories if your p99
  matters (e.g. the 1000 most recent items), and drop the events a decay has
  faded to almost nothing ([Preprocessing](preprocessing.md#shapes)).
- **Memory per thread.** Each thread that serves requests keeps a score buffer
  of up to 4 bytes per item (with `block_size=None`; a block's worth with
  `'auto'`), reused by its later requests: count it per thread of your
  server's pool on a large catalogue.
- **Batch when requests can wait.** `recommend_batch(histories, num_threads=n)`
  scores many users in one call (offline scoring, or requests collected for a
  millisecond or two): the per-call work is paid once.
- **Deploy a new model version** by saving it to a new directory and switching
  what the workers load ([Saving and loading](#saving-and-loading)).
- **No Python service to write?** [`similaripy-server`](server.md) serves a
  saved model over HTTP/JSON and gRPC (experimental).

A minimal FastAPI service (`fetch_history` is your store):

```python
from fastapi import FastAPI
from similaripy.preprocessing import rank_decay
from similaripy.serving import SparseKNN

app = FastAPI()
model = SparseKNN.load("models/v3")    # mmap: one copy of S for every worker process


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 20):   # `def`: FastAPI runs it in its thread pool, in parallel
    history, ts = fetch_history(user_id)    # item ids + their timestamps
    w = rank_decay(timestamps=ts, scale=50) # half-life 50 events: the latest weighs 1
    ids, scores = model.recommend(history, weights=w, k=k)
    return {"items": ids.tolist(), "scores": scores.tolist()}
```

An `async def` endpoint would run every request on the event loop thread, one
at a time.

Example of `recommend` latency by history cap:

| History cap | MovieLens 32M (84k items) | Yambda 500M (3.0M items) |
|-------------|---------------------------|--------------------------|
| 10 | 5 µs / 7 µs | 9 µs / 13 µs |
| 50 | 8 µs / 11 µs | 18 µs / 24 µs |
| 200 | 10 µs / 24 µs | 49 µs / 60 µs |
| 1000 | 10 µs / 80 µs | 185 µs / 250 µs |

p50 / p99 of the Python call, warm; S = `cosine(k=100)`, `k=100`, history
excluded; indicative, one Apple M2 Pro. MovieLens histories are mostly
shorter than 200 items, hence the flat p50.

### Evaluation framework

A framework that wants `fit` / `recommend` needs a few lines:

```python
class ItemKNNRecommender:
    def __init__(self, similarity=sim.cosine, transpose=True, **params):
        # transpose: serve S.T; (see Orientation of S)
        self.similarity, self.transpose, self.params = similarity, transpose, params

    def fit(self, URM):
        S = self.similarity(URM.T, **self.params)
        self.model = SparseKNN(S.T if self.transpose else S)
        return self

    def recommend(self, URM, k=10):
        histories = np.split(URM.indices, URM.indptr[1:-1])
        weights = np.split(URM.data, URM.indptr[1:-1])
        return self.model.recommend_batch(histories, weights=weights, k=k, sorted=True)   # best first: ranking metrics


rec = ItemKNNRecommender(sim.cosine, k=100, shrink=10).fit(URM_train)
ids, scores = rec.recommend(URM_train, k=20)
```

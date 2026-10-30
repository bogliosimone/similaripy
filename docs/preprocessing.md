# 🧰 Preprocessing

Utilities to prepare your interaction data:

- **Build**: the URM and the user histories from your interactions, in your own ids ([Utilities](#utilities)).
- **Weight**: each interaction by its age or recency, offline and for a live history ([Weighting](#weighting)).

```python
from similaripy.preprocessing import interactions_to_urm, time_decay   # ...
```

## Utilities

| Function | Description |
|----------|-------------|
| `interactions_to_urm(users, items, values)` | The user-item matrix (URM) and the id maps from one entry per event: `(URM, user_ids, item_ids)`. |
| `interactions_to_histories(users, items, weights)` | Each user's events as a history for `recommend_batch`: `(user_ids, histories, weights)`. |
| `urm_to_histories(URM, item_ids)` | Each row of a URM as a history for `recommend_batch`: `(histories, weights)`. |

### From an interaction log

```python
URM, user_ids, item_ids = interactions_to_urm(log.user, log.item, values)
```

- **Ids**: integers, as in `ItemKNN`; gaps and unaligned ids are fine. Row `u` is `user_ids[u]`, column `j` is `item_ids[j]` (sorted).
- **Values**: one per event (`None` = 1.0 each), e.g. an event weight: `log.event.map({"view": 1.0, "buy": 5.0})`.
- **Repeated (user, item) pairs**: `duplicates="sum"` (default) adds the values (counts); `"first"` / `"last"` keep the earliest / latest event's value, by `timestamps=` (ties: input order) or by input order without them. Explicit ratings: `"last"`.
- **Fixed vocabulary**: `item_ids=` (e.g. `model.item_ids`) builds a matrix with exactly those columns, e.g. one per country with the same columns; events with other ids are dropped.
- **Filters** (rare items, a country, an event type): on your dataframe, before the build, e.g. `log = log[log.item.map(log.item.value_counts()) >= 5]`.
- **Limits**: at most 2^31 - 1 distinct (user, item) pairs. Built in the Rust core: Yambda 500M (480M events) in ~20 s.

### Histories

From a log, e.g. the users of a holdout set: no matrix and no id mapping, histories in your item ids (ids the model doesn't have are ignored).

```python
user_ids, histories, weights = interactions_to_histories(test.user, test.item)
ids, scores = model.recommend_batch(histories, weights=weights, k=20)   # element u is user_ids[u]
```

Events keep the input order and repeats stay (`recommend` sums their weights: the same scores as through a URM). Events with weight 0 are dropped, so `weights=rank_decay(..., shape="step", scale=200)` also caps each history at 200 events.

From a URM (e.g. a benchmark dataset, or every training user):

```python
histories, weights = urm_to_histories(URM, item_ids)
ids, scores = model.recommend_batch(histories, weights=weights, k=20)   # element u is user_ids[u]
```

For millions of users, call `recommend_batch` in chunks (see [Serving](serving.md#single-and-batch-recommendation)).

## Weighting

| Function | Description |
|----------|-------------|
| `time_decay(timestamps, reference=..., scale=...)` | One weight per event from its age in time. |
| `rank_decay(users, scale=...)` | One weight per event from its recency rank in its user's history (0 = the latest). |

The decays weight single events, before the URM: multiply them into `values` (repeats are summed, so old repeats fade), or pass them to `recommend` for a live history, so training and serving use one formula. The [normalizations](normalization.md) (BM25, TF-IDF, ...) need row and column statistics, so they come after.

```text
events --time_decay / rank_decay--> weighted events --interactions_to_urm--> URM --bm25 / tfidf--> similarity
```

### Time decay

```python
w = time_decay(log.ts, reference=cutoff, scale=30 * 86400)      # half-life 30 days
URM, user_ids, item_ids = interactions_to_urm(log.user, log.item, values * w)

# online, the same formula on the live history
ids, scores = model.recommend(history, weights=time_decay(history_ts, reference=now, scale=30 * 86400))
```

The age is `reference - timestamp`. Choose where age 0 is:

| `reference` | Age from | Effect | Online |
|-------------|----------|--------|--------|
| a time (e.g. the training cutoff) | the same moment for every user | inactive users fade as a whole; recent trends weigh more | `reference=now` |
| `"user"` (with `users=`) | each user's latest event | every user's latest event weighs 1; only the order inside a profile matters | `reference=history_ts.max()` |

A time is a number in the timestamps' units, or a datetime for datetime timestamps.

```python
w = time_decay(log.ts, reference="user", users=log.user, scale=np.timedelta64(30, "D"))
```

### Rank decay

The age is the event's position in its user's history: 0 = the latest, 1 = the one before, ...
Without `timestamps` the log's own order (oldest first) is the time order.

```python
w = rank_decay(log.user, shape="step", scale=200)                    # each user's 200 latest events
w = rank_decay(log.user, timestamps=log.ts, shape="linear", scale=200)
URM, user_ids, item_ids = interactions_to_urm(log.user, log.item, w)
```

Ranks count events, not distinct items. For the latest distinct items, drop the repeats first: `log.sort_values("ts").drop_duplicates(["user", "item"], keep="last")`.

Online, a history kept in arrival order has the same weights on every request: compute them once, slice per request.

```python
table = rank_decay(np.zeros(1000, dtype=int), shape="step", scale=200)
history = history[-1000:]                                  # at most the table's length
ids, scores = model.recommend(history, weights=table[-len(history):])
```

### Shapes

Both decays share the shapes; the weight is 1 at age 0.

| `shape` | Weight | `scale` means |
|---------|--------|---------------|
| `"exponential"` (default) | 0.5^(age / scale) | half-life |
| `"power"` | (1 + (2^(1/p) - 1) age / scale)^-p, p = `exponent` (default 1) | half-life, with a longer tail for a smaller p |
| `"linear"` | max(0, 1 - age / scale) | horizon: 0 from there on |
| `"step"` | 1 if age < scale, else 0 | window |

- `scale` is required: in the timestamps' units for `time_decay` (durations like `np.timedelta64(30, "D")` work), in events for `rank_decay`.
- `scale=np.inf`: no decay (every weight 1).
- An event after the reference (a negative age, e.g. clock skew online) gets weight 1.

## End to end

A log with `user`, `item`, `ts` (seconds) and `event` columns:

```python
import numpy as np
import pandas as pd
import similaripy as sim
from similaripy.preprocessing import interactions_to_urm, time_decay, urm_to_histories
from similaripy.serving import ItemKNN

log = pd.read_parquet("events.parquet")
cutoff = log.ts.max()
values = log.event.map({"view": 1.0, "buy": 5.0}).to_numpy()

# 1. the URM, each event weighted by its age
w = time_decay(log.ts, reference=cutoff, scale=30 * 86400)
URM, user_ids, item_ids = interactions_to_urm(log.user, log.item, values * w)

# 2. similarity and model, in your item ids
S = sim.cosine(sim.bm25(URM).T, k=100)
model = ItemKNN(S.T, item_ids=item_ids)
model.save("model/")

# 3. offline: every user's top 20, back to a dataframe
histories, weights = urm_to_histories(URM, item_ids)
ids, scores = model.recommend_batch(histories, weights=weights, k=20)
recs = pd.DataFrame({
    "user": np.repeat(user_ids, [len(i) for i in ids]),
    "item": np.concatenate(ids),
    "score": np.concatenate(scores),
})

# 4. online, in a worker: one live history, the same decay
model = ItemKNN.load("model/")
ids, scores = model.recommend(history, weights=time_decay(history_ts, reference=now, scale=30 * 86400), k=20)
```

Without timestamps: drop the decay (`interactions_to_urm(log.user, log.item, values)`), or use `rank_decay` on the log's order.

## Tips

- **Tuning the decay**: `scale` is a parameter like `k`; `scale=np.inf` is the no-decay baseline. Each trial rebuilds the URM (~1 s on 50M events, ~20 s on 500M):

    ```python
    for days in [np.inf, 7, 30, 90]:
        w = time_decay(log.ts, reference=cutoff, scale=days * 86400)
        URM, user_ids, item_ids = interactions_to_urm(log.user, log.item, values * w)
        score = evaluate(ItemKNN(sim.cosine(URM.T, k=100).T, item_ids=item_ids))   # your metric
    ```

- **String ids**: `ItemKNN` serves integer ids. Map strings at the edge with `pd.factorize` and keep the names next to the model:

    ```python
    item_codes, item_names = pd.factorize(log.item)   # strings -> 0..n-1, back with item_names[ids]
    URM, user_ids, item_ids = interactions_to_urm(pd.factorize(log.user)[0], item_codes, values)
    np.save("model/item_names.npy", item_names.to_numpy(dtype=str))
    ```

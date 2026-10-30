# 🧰 Preprocessing

From an interaction log to the matrices the similarities and the serving engine take, in your own ids.

```python
from similaripy.preprocessing import from_interactions, rank_decay, time_decay, to_histories
```

| Function | Description |
|----------|-------------|
| `from_interactions(users, items, values)` | The user-item matrix (URM) and the id maps from one entry per event: `(URM, user_ids, item_ids)`. |
| `time_decay(timestamps, reference=..., scale=...)` | One weight per event from its age in time, to multiply into `values` (offline) or pass to `recommend` (online). |
| `rank_decay(users, scale=...)` | One weight per event from its recency rank in its user's history (0 = the latest). |
| `to_histories(URM, item_ids)` | Each row of the URM as a history for `recommend_batch`: `(histories, weights)`. |

Ids are integers everywhere, as in `ItemKNN`. String ids: map them at your edge (see [String ids](#string-ids)).

The order of the steps: the decays weight single events, before aggregation; the [normalizations](normalization.md) (BM25, TF-IDF, ...) need row and column statistics, so they come after.

```text
events --time_decay / rank_decay--> weighted events --from_interactions--> URM --bm25 / tfidf / normalize--> similarity
          (per event)                                                            (per matrix)
```

## From an interaction log

```python
URM, user_ids, item_ids = from_interactions(log.user, log.item, values)
```

- **Ids**: gaps and unaligned ids are fine; row `u` is `user_ids[u]`, column `j` is `item_ids[j]` (sorted).
- **Values**: one per event (`None` = 1.0 each), e.g. an event weight: `log.event.map({"view": 1.0, "buy": 5.0})`.
- **Repeated (user, item) pairs**: `duplicates="sum"` (default) adds the values (counts); `"first"` / `"last"` keep the earliest / latest event's value, by `timestamps=` (ties: input order) or by input order without them. Explicit ratings: `"last"`.
- **Fixed vocabulary**: `item_ids=` (e.g. `model.item_ids`) builds a matrix with exactly those columns, e.g. one per country with the same columns; events with other ids are dropped.
- **Filters** (rare items, a country, an event type): on your dataframe, before the build, e.g. `log = log[log.item.map(log.item.value_counts()) >= 5]`.
- **Limits**: at most 2^31 - 1 distinct (user, item) pairs. Built in the Rust core (a counting sort by user, then each user in parallel): Yambda 500M (480M events) in ~20 s.

## Time decay

Weight each event by its age, then build: repeats are summed, so old repeats fade.
The same call weights a live history, so training and serving use one formula.

```python
w = time_decay(log.ts, reference=cutoff, scale=30 * 86400)      # half-life 30 days
URM, user_ids, item_ids = from_interactions(log.user, log.item, values * w)

# online, the same formula on the live history
ids, scores = model.recommend(history, weights=time_decay(history_ts, reference=now, scale=30 * 86400))
```

The age is `reference - timestamp`. Choose where age 0 is:

| `reference` | Age from | Effect | Online |
|-------------|----------|--------|--------|
| a time (e.g. the training cutoff: a number in the timestamps' units, or a datetime for datetime timestamps) | the same moment for every user | inactive users fade as a whole; recent trends weigh more | `reference=now` |
| `"user"` (with `users=`) | each user's latest event | every user's latest event weighs 1; only the order inside a profile matters | `reference=history_ts.max()` |

```python
w = time_decay(log.ts, reference="user", users=log.user, scale=np.timedelta64(30, "D"))
```

## Rank decay

Weight each event by its position in its user's history: 0 = the latest, 1 = the one before, ...
Without `timestamps` the log's own order (oldest first) is the time order.

```python
w = rank_decay(log.user, shape="step", scale=200)                    # each user's 200 latest events
w = rank_decay(log.user, timestamps=log.ts, shape="linear", scale=200)
URM, user_ids, item_ids = from_interactions(log.user, log.item, w)
```

Ranks count events, not distinct items. For the latest distinct items, drop the repeats first: `log.sort_values("ts").drop_duplicates(["user", "item"], keep="last")`.

Online, a history kept in arrival order has the same weights on every request: compute them once, slice per request.

```python
table = rank_decay(np.zeros(1000, dtype=int), shape="step", scale=200)
history = history[-1000:]                                  # at most the table's length
ids, scores = model.recommend(history, weights=table[-len(history):])
```

## Decay shapes

Both decays share the shapes; the weight is 1 at age 0.

| `shape` | Weight | `scale` means |
|---------|--------|---------------|
| `"exponential"` (default) | 0.5^(age / scale) | half-life |
| `"power"` | (1 + (2^(1/p) - 1) age / scale)^-p, p = `exponent` (default 1) | half-life, with a longer tail for a smaller p |
| `"linear"` | max(0, 1 - age / scale) | horizon: 0 from there on |
| `"step"` | 1 if age < scale, else 0 | window |

- `scale` is required: in the timestamps' units for `time_decay` (seconds for datetimes; durations like `np.timedelta64(30, "D")` work), in events for `rank_decay`.
- `scale=np.inf`: no decay (every weight 1), the baseline of a parameter search.
- An event after the reference (a negative age, e.g. clock skew online) gets weight 1.

## Histories for batch recommendation

```python
histories, weights = to_histories(URM, item_ids)
ids, scores = model.recommend_batch(histories, weights=weights, k=20)   # element u is user_ids[u]
```

The histories carry the URM's values, so a decayed URM gives decayed histories, capped where the weight is 0.
For millions of users, call `recommend_batch` in chunks (see [Serving](serving.md#single-and-batch-recommendation)).

## End to end

A log with `user`, `item`, `ts` (seconds) and `event` columns:

```python
import numpy as np
import pandas as pd
import similaripy as sim
from similaripy.preprocessing import from_interactions, time_decay, to_histories
from similaripy.serving import ItemKNN

log = pd.read_parquet("events.parquet")
cutoff = log.ts.max()
values = log.event.map({"view": 1.0, "buy": 5.0}).to_numpy()

# 1. the URM, each event weighted by its age
w = time_decay(log.ts, reference=cutoff, scale=30 * 86400)
URM, user_ids, item_ids = from_interactions(log.user, log.item, values * w)

# 2. similarity and model, in your item ids
S = sim.cosine(sim.bm25(URM).T, k=100)
model = ItemKNN(S.T, item_ids=item_ids)
model.save("model/")

# 3. offline: every user's top 20, back to a dataframe
histories, weights = to_histories(URM, item_ids)
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

Without timestamps: drop the decay (`from_interactions(log.user, log.item, values)`), or use `rank_decay` on the log's order.

## Tuning the decay

Each trial rebuilds the URM and the similarity (the build is ~1 s on 50M events, ~20 s on 500M):

```python
import optuna

def objective(trial):
    kind = trial.suggest_categorical("decay", ["none", "time", "rank"])
    if kind == "time":
        w = time_decay(log.ts, reference=cutoff, scale=trial.suggest_float("days", 1, 365, log=True) * 86400)
    elif kind == "rank":
        w = rank_decay(log.user, timestamps=log.ts, shape="linear", scale=trial.suggest_int("events", 10, 2000, log=True))
    else:
        w = 1.0
    URM, user_ids, item_ids = from_interactions(log.user, log.item, values * w)
    S = sim.cosine(sim.bm25(URM).T, k=trial.suggest_int("k", 20, 500, log=True))
    return evaluate(ItemKNN(S.T, item_ids=item_ids))   # your metric on held-out events

optuna.create_study(direction="maximize").optimize(objective, n_trials=50)
```

## String ids

`ItemKNN` serves integer ids. Map strings at the edge with `pd.factorize`, and keep the names next to the model:

```python
user_codes, user_names = pd.factorize(log.user)
item_codes, item_names = pd.factorize(log.item)
URM, user_ids, item_ids = from_interactions(user_codes, item_codes, values)
model = ItemKNN(S.T, item_ids=item_ids)
model.save("model/")
np.save("model/item_names.npy", item_names.to_numpy(dtype=str))

# online: strings -> codes -> strings
item_names = pd.Index(np.load("model/item_names.npy"))
codes = item_names.get_indexer(history_names)         # unknown names -> -1, ignored by recommend
ids, scores = model.recommend(codes, k=20)
names = item_names[ids]
```

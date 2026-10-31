# 🧰 Preprocessing

Preprocessing prepares your interaction data in two parts:

- **Utilities**: the URM and the user histories from your interactions ([Utilities](#utilities)).
- **Weighting**: each interaction by its age or recency, offline and for live serving ([Weighting](#weighting)).

```python
from similaripy.preprocessing import (
    interactions_to_urm, interactions_to_histories, urm_to_histories,   # Utilities
    time_decay, rank_decay,                                              # Weighting
)
```

## Utilities

### Functions

| Function | Description |
|----------|-------------|
| `interactions_to_urm(users, items, values=None, *, timestamps=None, duplicates='sum')` | The user-item matrix (URM) from one entry per event, with the id maps: `(URM, user_ids, item_ids)`. |
| `interactions_to_histories(users, items, weights=None)` | Each user's events as a history for `recommend_batch`, e.g. for an evaluation test set: `(user_ids, histories, weights)`. |
| `urm_to_histories(URM, item_ids=None)` | Each row of a URM as a history for `recommend_batch`: `(histories, weights)`. |

- Ids mapping: in the URM, row `u` is `user_ids[u]` and column `j` is `item_ids[j]`.
- Histories use your item ids.
- Ids on your side must be integers; for strings or other types, see [Ids](#ids).

### Parameters

**`interactions_to_urm`**

| Parameter | Description |
|-----------|-------------|
| `users`, `items` | One integer id per event; gaps are fine. |
| `values` | One value per event, e.g. an event weight or a decay. *(default: `None` = 1.0 each)* |
| `timestamps` | One per event: numbers, or datetimes; only to order `'first'` / `'last'` (ties: input order). *(default: `None` = input order)* |
| `duplicates` | What a repeated (user, item) pair keeps: `'sum'` (counts), `'first'` or `'last'` (e.g. ratings). *(default: `'sum'`)* |

**`interactions_to_histories`**

| Parameter | Description |
|-----------|-------------|
| `users`, `items` | One integer id per event. |
| `weights` | One weight per event, e.g. a decay; weight-0 events are dropped. *(default: `None` = 1.0 each)* |

**`urm_to_histories`**

| Parameter | Description |
|-----------|-------------|
| `URM` | A user-item matrix. |
| `item_ids` | Your id of each column. *(default: `None` = column indices)* |

## Weighting

### Functions

| Function | Description |
|----------|-------------|
| `time_decay(timestamps, *, reference, scale, shape='exponential', exponent=None, users=None)` | One weight per event from its age in time: `reference - timestamp`. |
| `rank_decay(users=None, *, scale, timestamps=None, shape='exponential', exponent=None)` | One weight per event from its recency rank in its user's history: 0 = the latest. |

- Weighting and normalization order: the decays weight single events, before the URM (repeats are summed, so old repeats fade); the [normalizations](normalization.md) need the whole matrix, so they come after.

### Parameters

**`time_decay`**

| Parameter | Description |
|-----------|-------------|
| `timestamps` | One per event: numbers, or datetimes (read as seconds). |
| `reference` | Where age 0 is: a time (e.g. the training cutoff; online, `now` or the history's latest event) or `'user'` (each user's latest event). Required. |
| `scale` | In the timestamps' units, or a duration (e.g. `np.timedelta64(30, "D")`); `np.inf` = no decay. Required. |
| `shape` | The decay curve (see [Shapes](#shapes)). *(default: `'exponential'`)* |
| `exponent` | The power-law `p`, for `shape='power'` only. *(default: `1`)* |
| `users` | One id per event, for `reference='user'` only. |

**`rank_decay`**

| Parameter | Description |
|-----------|-------------|
| `users` | One id per event. *(default: `None` = one history, with `timestamps`)* |
| `scale` | In events; `np.inf` = no decay. Required. |
| `timestamps` | Orders each user's events. *(default: `None` = input order, oldest first)* |
| `shape` | The decay curve (see [Shapes](#shapes)). *(default: `'exponential'`)* |
| `exponent` | The power-law `p`, for `shape='power'` only. *(default: `1`)* |

### Shapes

| Shape | Weight | `scale` is |
|-------|--------|------------|
| `'exponential'` | `0.5^(age / scale)` | the half-life |
| `'power'` | `(1 + (2^(1/p) - 1) · age / scale)^(-p)` | the half-life, with a longer tail for a smaller `p` |
| `'linear'` | `max(0, 1 - age / scale)` | the horizon: 0 from there on |
| `'step'` | `1 if age < scale, else 0` | a window |

## Example

`train` and `test` are your event logs (`user`, `item`, `ts` in epoch seconds), split by time.

```python
import similaripy as sim
from similaripy.preprocessing import interactions_to_histories, interactions_to_urm, time_decay
from similaripy.serving import ItemKNN

# train: each event weighted by its age from the same time for every user (half-life 30 days)
cutoff = train.ts.max()                     # the end of the training data
w = time_decay(train.ts, reference=cutoff, scale=30 * 86400)
URM, user_ids, item_ids = interactions_to_urm(train.user, train.item, w)
model = ItemKNN(sim.cosine(sim.bm25(URM).T, k=100).T, item_ids=item_ids)

# evaluate: the holdout users' histories, straight from the log
test_users, histories, weights = interactions_to_histories(test.user, test.item)
ids, scores = model.recommend_batch(histories, weights=weights, k=20)

# online: one live history (item ids + epoch seconds), weighted from the user's latest event
w = time_decay(history_ts, reference=history_ts.max(), scale=30 * 86400)
ids, scores = model.recommend(history, weights=w)
```

## Ids

SimilariPy takes **integer ids** (any integer dtype, gaps fine): the URM numbers its rows and columns for you, and `ItemKNN(S, item_ids=item_ids)` takes and returns your ids. So `pd.factorize` is needed only for ids that aren't integers.

| Your ids | What to do |
|----------|------------|
| Integers, e.g. `905`, `12`, `7_000_000_000` | Nothing: pass them as they are. |
| Integers stored as text, e.g. `"905"` | Convert: `log.item.astype("int64")`. |
| Strings, UUIDs, other types (or unsigned ids beyond int64) | `pd.factorize` once, and keep the names next to the model. |

With strings, the codes are your ids from then on: build, serve and evaluate in codes, and map names in and out at the edges.

```python
item_codes, item_names = pd.factorize(log.item)   # once, on the whole log
user_codes, user_names = pd.factorize(log.user)
URM, user_ids, item_ids = interactions_to_urm(user_codes, item_codes)   # row u: user_names[user_ids[u]]
model = ItemKNN(sim.cosine(URM.T, k=100).T, item_ids=item_ids)
# the model knows only codes: keep item_names with it (e.g. save both side by side)

# serving: names -> codes -> names
name_to_code = pd.Index(item_names)                  # once, at startup
history = name_to_code.get_indexer(history_names)    # unknown names -> -1, ignored by recommend
ids, scores = model.recommend(history, k=20)
item_names[ids]
```

Factorizing train and test separately gives the same item two different codes: factorize the whole log once, or map later data with `name_to_code.get_indexer`.


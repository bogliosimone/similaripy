# 💾 Saved model format

`SparseKNN.save(path)` writes a plain directory: a few `.npy` files plus
`meta.json`. Any tool that reads numpy files and JSON can read a saved model,
and this page is the contract. It describes format `1`.

**The promise:** every 1.x version of similaripy loads every format up to its
own. A model saved by a newer format is a clear "upgrade similaripy" error,
never a silent misread.

## Directory layout

```text
model/
├── meta.json          what the files are (written last)
├── indptr.npy         row starts of S
├── pairs.npy          S's entries: (column, value) side by side
├── item_columns.npy   internal id -> original item (column of S)
└── item_ids.npy       your ids per original item (only if "item_ids": true)
```

- `meta.json` is written last: a directory without it is an incomplete save
  (`load` refuses it).
- Files and `meta.json` keys a reader doesn't know are ignored (see
  [Compatibility rules](#compatibility-rules)).

## Ids: original and internal

A model has two kinds of item ids:

- **Original ids** `0 .. n_cols - 1`: the columns of the `S` the model was
  built from. With `item_ids`, item `c` is called `item_ids[c]` in requests and
  answers; without, it is called `c`.
- **Internal ids** `0 .. n_items - 1`: the model's own layout (its
  `item_order`). `item_columns[i]` is the original id of internal item `i`.

Original items that `item_columns` doesn't list were dropped at build: the
model doesn't know them (ignored in a history, never recommended). Internal
items that never appear as a neighbour (`col`) can be in a history but are
never recommended (items outside `candidate_items`). The order of
internal ids also breaks ties: of two items with exactly the same score, the
lower internal id comes first.

## The `.npy` files

All are numpy `.npy` files (version 1.0 written; 1.0 to 3.0 read),
little-endian, 1-d, C order. The data starts at a 64-byte boundary, so the
files can be memory-mapped. The header must be exactly as `numpy.save` writes
it (`{'descr': '<i8', 'fortran_order': False, 'shape': (7,), }` plus padding):
another writer should use numpy, or copy that byte layout.

| File | dtype | Shape | Content |
|---|---|---|---|
| `indptr.npy` | `<i8` | `(n_items + 1,)` | Row `i` of S (internal item `i`'s neighbours) is `pairs[indptr[i]:indptr[i + 1]]`. `indptr[0] = 0`, non-decreasing, last = `nnz`. |
| `pairs.npy` | `[('col', '<i4'), ('val', '<f4')]` | `(nnz,)` | One entry of S: `col` = the neighbour's internal id (`< n_items`), strictly increasing within a row; `val` = the similarity, finite. |
| `item_columns.npy` | `<i4` | `(n_items,)` | Original id of each internal id: distinct, in `[0, n_cols)`. |
| `item_ids.npy` | `<i8` | `(n_cols,)` | Your id for each original item: unique. |

## `meta.json`

```json
{
  "format": 1,
  "item_ids": true,
  "kind": "SparseKNN",
  "n_cols": 7,
  "n_items": 6,
  "nnz": 12,
  "written_by": "similaripy 1.0.0"
}
```

| Key | Type | Meaning |
|---|---|---|
| `format` | int | The format version: `1`. A reader refuses a format newer than its own. |
| `kind` | string | `"SparseKNN"`. |
| `n_items` | int | Internal items: rows of S, length of `item_columns.npy`. |
| `n_cols` | int | Original items: columns of the S the model was built from, including dropped ones. |
| `nnz` | int | Entries of S: length of `pairs.npy`. |
| `item_ids` | bool | Whether `item_ids.npy` exists. |
| `written_by` | string | Which program saved it, for people and logs (e.g. `"similaripy 1.0.0"`). Free text: readers must not parse or require it. |

`n_items` and `nnz` repeat the array lengths so a reader can check that the
files belong together (and size memory before reading them).

## Reading it without similaripy

S in internal ids is a CSR matrix, and a recommendation is a sum of rows. This
plain numpy version returns what `recommend(history, k=k, sorted=True)` returns with the
other options' defaults (unit weights, history items excluded):

```python
import json
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from similaripy.serving import SparseKNN

# A small model to read back (any saved model works the same)
S = sp.random(50, 50, density=0.1, random_state=0, format="csr", dtype=np.float32)
path = Path(tempfile.mkdtemp()) / "model"
SparseKNN(S, item_ids=np.arange(50) * 10).save(path)

meta = json.loads((path / "meta.json").read_text())
assert meta["kind"] == "SparseKNN" and meta["format"] <= 1
indptr = np.load(path / "indptr.npy")
pairs = np.load(path / "pairs.npy")
item_columns = np.load(path / "item_columns.npy")
item_ids = np.load(path / "item_ids.npy") if meta["item_ids"] else np.arange(meta["n_cols"])

n = meta["n_items"]
S_internal = sp.csr_array((pairs["val"], pairs["col"], indptr), shape=(n, n))
id_of = item_ids[item_columns]  # internal id -> your id
internal_of = {int(e): i for i, e in enumerate(id_of)}


def recommend(history, k=10):
    rows = sorted(internal_of[h] for h in history if h in internal_of)
    scores = np.zeros(n, dtype=np.float32)
    for i in rows:  # summed in internal order, as similaripy does
        scores[S_internal.indices[S_internal.indptr[i] : S_internal.indptr[i + 1]]] += (
            S_internal.data[S_internal.indptr[i] : S_internal.indptr[i + 1]]
        )
    scores[rows] = 0  # history items are never recommended
    best = np.lexsort((np.arange(n), -scores))[:k]  # score desc, then the lower internal id
    best = best[scores[best] > 0]  # only positive scores
    return id_of[best], scores[best]


ids, scores = recommend([30, 70, 120], k=5)
want_ids, want_scores = SparseKNN.load(path).recommend([30, 70, 120], k=5, sorted=True)
assert np.array_equal(ids, want_ids) and np.allclose(scores, want_scores)
```

## Compatibility rules

These hold for every 1.x version, so tools reading format 1 keep working:

1. **Readers ignore what they don't know:** unknown `meta.json` keys and
   unknown files in the directory.
2. **Additions that don't change results need no new format.** A later version
   may add a key or a file that a reader can ignore and still serve the same
   results (a label, a timestamp, user metadata). Anything else (a new value
   dtype, a different layout, a file that changes scores) bumps `format`.
3. **A newer `format` is an error**, never a best-effort read.
4. **Known keys with unknown values are errors** where they change meaning
   (e.g. a `kind` other than `"SparseKNN"`).

## Writing a model directory

similaripy only writes new files into a new (or empty) directory, and never
changes a saved model: with `mmap=True`, changing a file that a process has
mapped is undefined behaviour. Save each version to its own directory and
switch what the workers load (see [Saving and loading](serving.md#saving-and-loading)).
Another tool writing this format should do the same, write `meta.json` last,
and keep to the dtypes and invariants above: `load` checks all of them.

"""The public import paths: matrix functions at the top level (the 0.x API) and
in their modules, the same objects; the serving engine and the preprocessing utilities in their modules only."""

import inspect
import math

import similaripy as sim
from similaripy import normalization, preprocessing, serving, similarity


def test_top_level_functions_are_the_module_ones():
    for module in (similarity, normalization):
        for name in module.__all__:
            assert getattr(sim, name) is getattr(module, name), name
    assert set(sim.__all__) == {"__version__", "normalization", "preprocessing", "serving", "similarity"} | set(
        similarity.__all__
    ) | set(normalization.__all__)


def test_serving_has_one_path():
    assert serving.__all__ == ["ItemKNN"]
    assert not hasattr(sim, "ItemKNN")
    assert sim.serving.ItemKNN is serving.ItemKNN


def test_preprocessing_has_one_path():
    assert preprocessing.__all__ == [
        "interactions_to_histories",
        "interactions_to_urm",
        "rank_decay",
        "time_decay",
        "urm_to_histories",
    ]
    assert not any(hasattr(sim, name) for name in preprocessing.__all__)


# Every public signature, without annotations (their text varies with the Python
# and numpy versions). A change here is an API change: update the dict on purpose.
_SIM = "verbose=False, format_output='csr', num_threads=0, block_size='auto'"
_SIM = f"k=100, shrink=0.0, shrink_type='stabilized', threshold=0.0, binary=False, target_rows=None, target_cols=None, filter_cols=None, {_SIM}"
_E = repr(math.e)
SIGNATURES = {
    "similarity.asymmetric_cosine": f"(matrix1, matrix2=None, *, alpha=0.5, {_SIM})",
    "similarity.cosine": f"(matrix1, matrix2=None, *, {_SIM})",
    "similarity.dice": f"(matrix1, matrix2=None, *, {_SIM})",
    "similarity.dot_product": f"(matrix1, matrix2=None, *, {_SIM})",
    "similarity.jaccard": f"(matrix1, matrix2=None, *, {_SIM})",
    "similarity.p3alpha": f"(matrix1, matrix2=None, *, alpha=1.0, {_SIM})",
    "similarity.rp3beta": f"(matrix1, matrix2=None, *, alpha=0.75, beta=0.3, {_SIM})",
    "similarity.s_plus": "(matrix1, matrix2=None, *, l1=0.5, l2=0.5, l3=0.0, t1=1.0, t2=1.0, c1=0.5, c2=0.5, "
    f"pop1='none', pop2='none', alpha=1.0, beta1=0.0, beta2=0.0, {_SIM})",
    "similarity.tversky": f"(matrix1, matrix2=None, *, alpha=1.0, beta=1.0, {_SIM})",
    "normalization.binarize": "(X)",
    "normalization.bm25": f"(X, *, axis=1, k1=1.2, b=0.75, logbase={_E}, tf_mode='raw', idf_mode='bm25')",
    "normalization.bm25plus": f"(X, *, axis=1, k1=1.2, b=0.75, delta=1.0, logbase={_E}, tf_mode='raw', idf_mode='bm25')",
    "normalization.normalize": "(X, *, norm='l2', axis=1)",
    "normalization.tfidf": f"(X, *, axis=1, logbase={_E}, tf_mode='sqrt', idf_mode='smooth')",
    "preprocessing.interactions_to_histories": "(users, items, weights=None)",
    "preprocessing.interactions_to_urm": "(users, items, values=None, *, timestamps=None, duplicates='sum')",
    "preprocessing.rank_decay": "(users=None, *, scale, timestamps=None, shape='exponential', exponent=None)",
    "preprocessing.time_decay": "(timestamps, *, reference, scale, shape='exponential', exponent=None, users=None)",
    "preprocessing.urm_to_histories": "(URM, *, item_ids=None)",
    "ItemKNN.__init__": "(self, S, *, item_ids=None, candidate_items=None, block_size='auto', num_threads=0)",
    "ItemKNN.load": "(path, *, mmap=True)",
    "ItemKNN.recommend": "(self, history, *, weights=None, k=10, exclude_seen=True, include=None, exclude=None, sorted=True)",
    "ItemKNN.recommend_batch": "(self, histories, *, weights=None, k=10, exclude_seen=True, include=None, exclude=None, "
    "sorted=True, num_threads=0)",
    "ItemKNN.save": "(self, path, *, overwrite=False)",
    "ItemKNN.view": "(self, *, include=None, exclude=None)",
}


def _signature(f):
    sig = inspect.signature(f)
    params = [p.replace(annotation=p.empty) for p in sig.parameters.values()]
    return str(sig.replace(parameters=params, return_annotation=sig.empty))


def test_public_signatures():
    found = {
        f"{m.__name__.split('.')[-1]}.{n}": getattr(m, n)
        for m in (similarity, normalization, preprocessing)
        for n in m.__all__
    }
    K = serving.ItemKNN
    found |= {
        f"ItemKNN.{n}": getattr(K, n)
        for n in vars(K)
        if (n == "__init__" or not n.startswith("_")) and callable(getattr(K, n))
    }
    assert {k: _signature(f) for k, f in found.items()} == SIGNATURES
    assert [n for n, v in vars(K).items() if isinstance(v, property)] == ["item_ids"]

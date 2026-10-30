"""The public import paths: matrix functions at the top level (the 0.x API) and
in their modules, the same objects; the serving engine and the preprocessing utilities in their modules only."""

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
    assert preprocessing.__all__ == ["from_interactions", "rank_decay", "time_decay", "to_histories"]
    assert not any(hasattr(sim, name) for name in preprocessing.__all__)

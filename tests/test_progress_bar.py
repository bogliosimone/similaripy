import re

import numpy as np
import pytest
import scipy.sparse as sp

import similaripy as sim


def test_progress_bar_output(capfd):
    m = sp.random(300, 200, density=0.05, format="csr", dtype=np.float32, random_state=0)
    sim.cosine(m, k=10, verbose=True, format_output="csr")
    out, err = capfd.readouterr()
    assert out == ""
    # Last \r-redraw: the closing line, same format as similaripy 0.7.0
    final = err.split("\r")[-1]
    assert re.fullmatch(r"Done: 100%\|#{25}\| 300/300 \[\d\d:\d\d, \d+\.\d\dit/s\]\n", final), repr(final)
    assert "Computing:" in err and "Building csr matrix:" in err


@pytest.mark.parametrize("kw", [{}, {"verbose": False}])  # quiet by default
def test_progress_bar_silent(capfd, kw):
    m = sp.random(300, 200, density=0.05, format="csr", dtype=np.float32, random_state=0)
    sim.cosine(m, k=10, **kw)
    assert capfd.readouterr() == ("", "")


def test_progress_bar_closed_on_prep_error(capfd, monkeypatch):
    from similaripy import _s_plus

    def boom(cols, num_threads):
        raise RuntimeError("prep failed")

    monkeypatch.setattr(_s_plus, "_matrix_selector", boom)
    with pytest.raises(RuntimeError, match="prep failed"):
        sim.cosine(sp.random_array((60, 40), density=0.2, format="csr", rng=0), k=5, verbose=True)
    assert capfd.readouterr().err.split("\r")[-1].startswith("Failed:")

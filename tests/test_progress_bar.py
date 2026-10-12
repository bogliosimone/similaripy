import importlib.util
import re

import numpy as np
import pytest
import scipy.sparse as sp

import similaripy as sim

# The Rust module exists only after `make install-dev-rust` (Step 2a).
BACKENDS = ['cython'] + (['rust'] if importlib.util.find_spec('_similaripy_rs') else [])


@pytest.mark.parametrize('backend', BACKENDS)
def test_progress_bar_output(capfd, backend):
    m = sp.random(300, 200, density=0.05, format='csr', dtype=np.float32, random_state=0)
    sim.cosine(m, k=10, verbose=True, format_output='csr', _backend=backend)
    out, err = capfd.readouterr()
    assert out == ''
    # Last \r-redraw: the closing line, same format on both backends
    final = err.split('\r')[-1]
    assert re.fullmatch(r'Done: 100%\|#{25}\| 300/300 \[\d\d:\d\d, \d+\.\d\dit/s\]\n', final), repr(final)
    assert 'Computing:' in err and 'Building csr matrix:' in err


@pytest.mark.parametrize('backend', BACKENDS)
def test_progress_bar_silent(capfd, backend):
    m = sp.random(300, 200, density=0.05, format='csr', dtype=np.float32, random_state=0)
    sim.cosine(m, k=10, verbose=False, _backend=backend)
    assert capfd.readouterr() == ('', '')

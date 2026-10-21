import re

import numpy as np
import pytest
import scipy.sparse as sp

import similaripy as sim


def test_progress_bar_output(capfd):
    m = sp.random(300, 200, density=0.05, format='csr', dtype=np.float32, random_state=0)
    sim.cosine(m, k=10, verbose=True, format_output='csr')
    out, err = capfd.readouterr()
    assert out == ''
    # Last \r-redraw: the closing line, same format as similaripy 0.7.0
    final = err.split('\r')[-1]
    assert re.fullmatch(r'Done: 100%\|#{25}\| 300/300 \[\d\d:\d\d, \d+\.\d\dit/s\]\n', final), repr(final)
    assert 'Computing:' in err and 'Building csr matrix:' in err


@pytest.mark.parametrize('kw', [{}, {'verbose': False}])  # quiet by default
def test_progress_bar_silent(capfd, kw):
    m = sp.random(300, 200, density=0.05, format='csr', dtype=np.float32, random_state=0)
    sim.cosine(m, k=10, **kw)
    assert capfd.readouterr() == ('', '')

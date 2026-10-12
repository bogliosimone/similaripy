import re

import numpy as np
import scipy.sparse as sp

import similaripy as sim


def test_progress_bar_output(capfd):
    m = sp.random(300, 200, density=0.05, format='csr', dtype=np.float32, random_state=0)
    sim.cosine(m, k=10, verbose=True, format_output='csr')
    out, err = capfd.readouterr()
    assert out == ''
    # Last \r-redraw: the closing line, same format as 0.7.0's C++ bar
    final = err.split('\r')[-1]
    assert re.fullmatch(r'Done: 100%\|#{25}\| 300/300 \[\d\d:\d\d, \d+\.\d\dit/s\]\n', final), repr(final)
    assert 'Computing:' in err and 'Building csr matrix:' in err


def test_progress_bar_silent(capfd):
    m = sp.random(300, 200, density=0.05, format='csr', dtype=np.float32, random_state=0)
    sim.cosine(m, k=10, verbose=False)
    assert capfd.readouterr() == ('', '')

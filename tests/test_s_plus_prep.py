"""Prep regressions: 0.7.0 bugs fixed on the Rust path (v1 Step 2a slice 3).

The Cython path is kept as 0.7.0 (the gate reference), so these xfail there
until slice 5 deletes it.
"""
import importlib.util
import subprocess
import sys

import numpy as np
import pytest
import scipy.sparse as sp

import similaripy as sim

HAS_RUST = importlib.util.find_spec('_similaripy_rs') is not None
needs_rust = pytest.mark.skipif(not HAS_RUST, reason='needs `make install-dev-rust`')
BACKENDS = [
    pytest.param('cython', marks=pytest.mark.xfail(strict=True, reason='0.7.0 bug, fixed on the Rust path')),
    pytest.param('rust', marks=needs_rust),
]


def _m(rows=60, cols=40, seed=0):
    return sp.random_array((rows, cols), density=0.2, format='csr', dtype=np.float32, rng=seed)


def _same(a, b):
    np.testing.assert_allclose(a.toarray(), b.toarray(), rtol=1e-6)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('selector', ['filter_cols', 'target_cols'])
def test_int64_indices_with_array_selector(backend, selector):
    m = _m()
    m64 = m.copy()
    m64.indices, m64.indptr = m64.indices.astype(np.int64), m64.indptr.astype(np.int64)
    kw = {selector: [1, 2, 7], 'k': 10, 'verbose': False, '_backend': backend}
    _same(sim.cosine(m64, **kw), sim.cosine(m, **kw))


@pytest.mark.parametrize('backend', BACKENDS)
def test_unsorted_m2_blocked_identity_permutation(backend):
    # Column j is in the first 40 - j rows: popularity already descending, so
    # the popularity permutation is the identity and 0.7.0 skipped the sort.
    cnt = np.arange(40, 0, -1)
    rows = np.concatenate([np.arange(c) for c in cnt])
    cols = np.concatenate([np.full(c, j) for j, c in enumerate(cnt)])
    m2 = sp.csr_array((np.ones(len(rows), np.float32), (rows, cols)), shape=(50, 40))
    m2u = m2.copy()
    for r in range(m2u.shape[0]):
        s, e = m2u.indptr[r], m2u.indptr[r + 1]
        m2u.indices[s:e] = m2u.indices[s:e][::-1].copy()
    m2u.has_sorted_indices = False
    m1 = _m(30, 50, seed=1)
    ref = sim.dot_product(m1, m2, k=40, block_size=None, verbose=False, _backend=backend)
    out = sim.dot_product(m1, m2u, k=40, block_size=8, verbose=False, _backend=backend)
    _same(out, ref)


@pytest.mark.parametrize('backend', BACKENDS)
def test_inputs_not_mutated(backend):
    m = _m()
    m.data[::7] = 0  # explicit zeros
    f = sp.csr_array((np.array([1, 0, 1], np.float32), np.array([5, 3, 1]), np.array([0, 3] + [3] * 59)),
                     shape=(60, 60))
    m_before, f_before = (m.data.copy(), m.indices.copy()), (f.data.copy(), f.indices.copy())
    sim.cosine(m, k=5, binary=True, filter_cols=f, target_cols=f.copy(), block_size=8,
               verbose=False, _backend=backend)
    for arr, before in zip((m.data, m.indices, f.data, f.indices), (*m_before, *f_before)):
        np.testing.assert_array_equal(arr, before)


@pytest.mark.parametrize('backend', ['cython', pytest.param('rust', marks=needs_rust)])
def test_binary_ignores_explicit_zeros(backend):
    m = _m()
    clean = m.copy()
    m.data[::7] = 0
    clean.data[::7] = 0
    clean.eliminate_zeros()
    kw = {'k': 10, 'binary': True, 'verbose': False, '_backend': backend}
    _same(sim.jaccard(m, **kw), sim.jaccard(clean, **kw))


@pytest.mark.parametrize('backend', BACKENDS)
def test_trailing_empty_rows(backend):
    m = _m().tolil()
    m[-3:, :] = 0
    m = m.tocsr()
    m.eliminate_zeros()
    out = sim.cosine(m, k=5, verbose=False, _backend=backend)
    assert out.shape == (60, 60) and out.tocsr()[-3:].nnz == 0


@needs_rust
@pytest.mark.parametrize('format_output', ['csr', 'coo'])
@pytest.mark.parametrize('kw', [
    {},
    {'target_rows': [17, 3, 3, 40, 0]},
    {'filter_cols': [1, 5, 9]},
    {'target_cols': np.arange(0, 60, 2)},
    {'block_size': 8},
    {'block_size': 8, 'filter_cols': 'matrix'},
], ids=['plain', 'target_rows', 'filter_array', 'target_array', 'blocked', 'blocked_filter_matrix'])
def test_rust_matches_cython(kw, format_output):
    m = _m()
    if kw.get('filter_cols') == 'matrix':
        kw = {**kw, 'filter_cols': _m(60, 60, seed=3)}
    outs = [sim.cosine(m, k=10, verbose=False, format_output=format_output, _backend=b, **kw)
            for b in ('cython', 'rust')]
    assert all(type(o) is type(outs[0]) for o in outs)
    _same(*outs)
    if format_output == 'csr':
        c, r = (o.tocsr() for o in outs)
        assert c.indices.dtype == r.indices.dtype
        c.sort_indices()  # within-row order is the kernel's top-K drain order, which differs
        r.sort_indices()
        np.testing.assert_array_equal(c.indptr, r.indptr)
        np.testing.assert_array_equal(c.indices, r.indices)


@needs_rust
def test_progress_bar_closed_on_prep_error(capfd, monkeypatch):
    from similaripy import _s_plus

    def boom(cols):
        raise RuntimeError('prep failed')
    monkeypatch.setattr(_s_plus, '_column_selector', boom)
    with pytest.raises(RuntimeError, match='prep failed'):
        sim.cosine(_m(), k=5, verbose=True, _backend='rust')
    assert capfd.readouterr().err.split('\r')[-1].startswith('Failed:')


@needs_rust
def test_get_num_threads():
    from similaripy import _s_plus
    assert _s_plus.get_num_threads() >= 1


@needs_rust
def test_rust_path_never_imports_cython():
    code = (
        "import sys; sys.modules['similaripy.cython_code'] = None\n"
        "import numpy as np, scipy.sparse as sp, similaripy as sim\n"
        "m = sp.random_array((50, 40), density=0.2, format='csr', dtype=np.float32, rng=0)\n"
        "sim.cosine(m, k=5, verbose=True, filter_cols=[1], block_size=8, _backend='rust')\n"
        "sim.bm25(m, _backend='rust'); sim.normalize(m, _backend='rust')\n"
    )
    subprocess.run([sys.executable, '-c', code], check=True)


@needs_rust
def test_int32_output_guard():
    # 50000 rows * k=50000 > int32 max: must fail before allocating the output
    m = sp.csr_array((50000, 50000), dtype=np.float32)
    with pytest.raises(ValueError, match='output too large'):
        sim.dot_product(m, k=50000, verbose=False, _backend='rust')


@needs_rust
def test_negative_block_size():
    with pytest.raises(ValueError, match='block_size'):
        sim.cosine(_m(), k=5, block_size=-1, verbose=False, _backend='rust')


@needs_rust
def test_no_blocking_matches_blocked():
    m = _m(60, 300)
    kw = {'k': 20, 'verbose': False, '_backend': 'rust'}
    _same(sim.cosine(m.T, block_size=None, **kw), sim.cosine(m.T, block_size=16, **kw))

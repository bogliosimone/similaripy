import re
import sys
import threading

import numpy as np
import pytest
import scipy.sparse as sp

import similaripy as sim
from similaripy import _core, _s_plus

BAR = _core.ProgressBar
DONE = r"Done 100%\|█{25}\| 300/300 rows \[\d\d:\d\d = prep \d+\.\ds \+ compute \d+\.\ds \+ build \d+\.\ds\]\n"


def _m():
    return sp.random(300, 200, density=0.05, format="csr", dtype=np.float32, random_state=0)


def test_log_mode(capsys):
    # not a terminal (captured): one line per phase, the final line last
    sim.cosine(_m(), k=10, verbose=True, format_output="csr")
    out, err = capsys.readouterr()
    assert out == "" and "\r" not in err
    lines = err.splitlines(keepends=True)
    assert lines[0].startswith("Preprocessing [")
    assert any(x.startswith("Building csr matrix [") for x in lines)
    assert re.fullmatch(DONE, lines[-1]), lines[-1]


def test_live_mode(capsys, monkeypatch):
    monkeypatch.setattr(_s_plus, "_live_bar", lambda: True)
    sim.cosine(_m(), k=10, verbose=True, format_output="csr")
    err = capsys.readouterr().err
    # \r redraws of one line, padded over a longer previous one; one newline at the end
    assert err.count("\n") == 1 and err.endswith("\n")
    assert re.fullmatch(DONE, err.split("\r")[-1].rstrip(" ")), repr(err)


@pytest.mark.parametrize("columns", [80, 50])
def test_live_fits_narrow_terminal(capsys, monkeypatch, columns):
    # live lines stay within columns - 1 (no wrap); the bar shrinks, gone when no room
    monkeypatch.setattr(_s_plus, "_live_bar", lambda: True)
    monkeypatch.setattr(_s_plus, "_columns", lambda: columns)
    sim.cosine(_m(), k=10, verbose=True, format_output="csr")
    *live, final = capsys.readouterr().err.split("\r")
    assert all(len(x) <= columns - 1 for x in live), live
    if columns == 80:
        assert final.startswith("Done 100%|") and 0 < final.count("█") < 25, final
    else:
        assert final.startswith("Done 100% 300/300 rows ["), final


def test_live_when_jupyter(monkeypatch):
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
    monkeypatch.delitem(sys.modules, "ipykernel", raising=False)
    assert not _s_plus._live_bar()
    monkeypatch.setitem(sys.modules, "ipykernel", object())
    assert _s_plus._live_bar()


@pytest.mark.parametrize("kw", [{}, {"verbose": False}])  # quiet by default
def test_silent(capsys, kw):
    sim.cosine(_m(), k=10, **kw)
    assert capsys.readouterr() == ("", "")


def test_rows_are_unique_targets(capsys):
    sim.cosine(_m(), k=10, verbose=True, target_rows=[3, 1, 3, 2])
    assert re.search(r"Done 100%\|█{25}\| 3/3 rows", capsys.readouterr().err)


def test_done_counts_every_row(capsys):
    # m2 without columns: k = 0, the kernel skips every row; Done still reads 100%
    sim.dot_product(_m(), sp.csr_array((200, 0), dtype=np.float32), k=10, verbose=True)
    assert re.search(r"Done 100%\|█{25}\| 300/300 rows", capsys.readouterr().err)


def test_failed_on_prep_error(capsys):
    # the core's prep raises (a NaN value): the bar closes on "Failed", no rows yet
    m = sp.random_array((60, 40), density=0.2, format="csr", rng=0)
    m.data[0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        sim.cosine(m, k=5, verbose=True)
    lines = capsys.readouterr().err.splitlines()
    assert lines[0].startswith("Preprocessing [")
    assert re.fullmatch(r"Failed \[\d\d:\d\d = prep \d+\.\ds\]", lines[-1]), lines[-1]


def test_concurrent_bars(capsys, monkeypatch):
    # one live bar owns the line, the other prints log lines; both finish
    monkeypatch.setattr(_s_plus, "_live_bar", lambda: True)
    m = _m()
    ts = [threading.Thread(target=sim.cosine, args=(m,), kwargs={"k": 10, "verbose": True}) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    err = capsys.readouterr().err
    assert len(re.findall(r"Done 100%", err)) == 4
    assert "\n\n" not in err  # a log line breaks the live line once, no blank lines


def test_no_stderr(monkeypatch):
    monkeypatch.setattr(sys, "stderr", None)
    out = sim.cosine(_m(), k=10, verbose=True)
    assert out.nnz > 0


def test_dropped_without_exit():
    # never entered or exited: the drawer stops, nothing hangs
    bar = BAR(False)
    del bar


def test_formats():
    assert [BAR._fmt_count(n) for n in (0, 999, 999.6, 9_994, 9_995, 25_800, 999_500, 1.234e6, 5e9)] == [
        "0",
        "999",
        "1.00k",
        "9.99k",
        "10.0k",
        "25.8k",
        "1.00M",
        "1.23M",
        "5.00G",
    ]
    assert [BAR._fmt_clock(s) for s in (0, 59.9, 61, 3599, 3600, 3725)] == [
        "00:00",
        "00:59",
        "01:01",
        "59:59",
        "1:00:00",
        "1:02:05",
    ]
    # 25 cells in eighths: 1/8 of the rows = 25 eighths = 3 full + 1/8
    assert BAR._bar(0, 8) == " " * 25
    assert BAR._bar(1, 8) == "███▏" + " " * 21
    assert BAR._bar(8, 8) == BAR._bar(5, 0) == "█" * 25
    assert BAR._bar(1, 2, width=3) == "█▌ " and BAR._bar(1, 2, width=0) == ""

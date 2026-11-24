"""`ServerClient` against the real `similaripy-server` binary (SIMILARIPY_SERVER_BIN:
built by CI's client job, `cargo build -p similaripy-server` locally; unset =
those tests skip), and `write_view` / `read_view` (no server needed)."""

import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from similaripy.serving import ServerClient, ServerError, SparseKNN, read_view, write_view

MODEL = Path(__file__).parent / "data" / "sparseknn-format1"  # items 10, 20, ..., 70


def _same(a, b):
    return np.array_equal(a[0], b[0]) and np.array_equal(a[1].view(np.int32), b[1].view(np.int32))


def _wait(pred, what, timeout=10.0):
    end = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < end, f"timed out waiting for {what}"
        time.sleep(0.05)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """(url, views dir) of a server on MODEL, a free port, views rescanned every 0.1 s,
    idle keep-alive connections closed after 1 s."""
    binary = os.environ.get("SIMILARIPY_SERVER_BIN")
    if not binary:
        pytest.skip("SIMILARIPY_SERVER_BIN not set (cargo build -p similaripy-server)")
    d = tmp_path_factory.mktemp("server")
    views, log = d / "views", d / "stderr.log"
    views.mkdir()
    args = [binary, "--model", str(MODEL), "--model.name", "fmt1", "--addr", "127.0.0.1:0"]
    args += ["--views.dir", str(views), "--views.reload_interval", "0.1", "--timeouts.header_read", "1"]
    with open(log, "w") as err:
        proc = subprocess.Popen(args, stderr=err)
    try:

        def addr():
            assert proc.poll() is None, log.read_text()  # exited: its error
            m = re.search(r"listening on (\S+)", log.read_text())
            return m and m.group(1)

        _wait(addr, "the server's address")
        yield f"http://{addr()}", views
    finally:
        proc.terminate()
        proc.wait(10)


@pytest.fixture
def client(server):
    with ServerClient(server[0]) as c:
        yield c


@pytest.fixture(scope="module")
def model():
    return SparseKNN.load(MODEL)


def test_recommend_and_score_match_the_library(client, model):
    cases = [
        ([20, 40], {}),
        ([10, 50, 70], {"weights": [1.0, 0.5, 3.0]}),
        ([10, 20, 99], {"weights": [1.0, -1.0, 2.0], "exclude_history": False}),  # 99: unknown, a dislike
        ([20, 40], {"include": [10, 30, 60]}),
        ([20, 40], {"include": []}),  # [] = no item, not "every item"
        ([20, 40], {"exclude": [30]}),
        ([], {}),
    ]
    for h, opts in cases:
        for k in (1, 3, 10):
            want = model.recommend(h, k=k, sorted=True, **opts)
            got = client.recommend(np.array(h, dtype=np.int64), k=k, sorted=True, **opts)
            assert got[0].dtype == np.int64 and got[1].dtype == np.float32
            assert _same(got, want), (h, opts, k)
        w = opts.get("weights")
        items = [70, 10, 99, 10, 30]
        assert np.array_equal(client.score(h, items, weights=w), model.score(h, items, weights=w))
    # the server's default: unsorted, k 10, the same items as the library's default
    got, want = client.recommend([20, 40]), model.recommend([20, 40])
    assert sorted(got[0].tolist()) == sorted(want[0].tolist())


def test_return_info_and_request_id(client):
    ids, scores, info = client.recommend([20, 40], k=3, request_id="abc-1", return_info=True)
    assert set(ids.tolist()) == {30, 10, 50}  # unsorted: the server's default
    assert info["request_id"] == "abc-1" and info["model"]["name"] == "fmt1"
    assert info["k"] == 3 and info["sorted"] is False and info["view"] is None and info["views"] == []
    assert info["history"] == {"sent": 2, "truncated": 0, "unknown": 0}
    scores, info = client.score([20, 99], [30], return_info=True)
    assert scores.shape == (1,) and info["history"]["unknown"] == 1 and "scores" not in info


@pytest.mark.parametrize(
    "call, code",
    [
        (lambda c: c.recommend([20], k=5000), "k_over_max"),
        (lambda c: c.recommend([20], view="nope"), "unknown_view"),
        (lambda c: c.recommend([20], views=["a"] * 9), "too_many_views"),
        (lambda c: c.score([20], [30], weights=[1.0, 2.0]), "bad_weights"),
    ],
)
def test_server_errors(client, call, code):
    with pytest.raises(ServerError) as e:
        call(client)
    assert e.value.status == 400 and e.value.code == code and e.value.message and e.value.request_id
    assert client.recommend([20], k=1)[0].tolist() == [30]  # the connection still works


def test_request_id_of_an_error(client):
    with pytest.raises(ServerError, match="k_over_max") as e:
        client.recommend([20], k=5000, request_id="mine-7")
    assert e.value.request_id == "mine-7"


def test_bad_arguments_never_reach_the_server():
    c = ServerClient("http://127.0.0.1:9")  # nothing listens: a call would fail differently
    with pytest.raises(TypeError, match="integer ids"):
        c.recommend([1.5])
    with pytest.raises(ValueError, match="k must be"):
        c.recommend([1], k=-1)
    with pytest.raises(ValueError, match="finite"):  # JSON has no NaN: the library's error, not bad_json
        c.score([1], [2], weights=[np.nan])
    for bad in ("eu", ["eu", 3]):
        with pytest.raises(TypeError, match="list of view names"):
            c.recommend([1], views=bad)
    for bad in ("", "a b", "é-1", "x" * 129):  # the server would replace them: a silent log-join miss
        with pytest.raises(ValueError, match="request_id"):
            c.recommend([1], request_id=bad)
    for url in ("127.0.0.1:8080", "ftp://h", "http://", "http://h:abc", "http://u:p@h", "http://h/?x=1"):
        with pytest.raises(ValueError, match="url must be"):
            ServerClient(url)
    with pytest.raises(ValueError, match="max_retries"):
        ServerClient(max_retries=-1)


def test_views_round_trip(server, client, model):
    """write_view -> the server loads the folder -> views() lists it, requests use it,
    and read_view gives the library the same view."""
    _, views = server
    write_view(views, "in_stock", include=[10, 30, 50, 60])
    write_view(views, "no_30", exclude=np.array([30], dtype=np.uint32))
    _wait(lambda: {"in_stock", "no_30"} <= client.views()["views"].keys(), "the views")
    v = client.views()
    assert v["items"] == 6  # items with a column (70 has none)
    assert v["views"]["in_stock"]["allowed"] == 4 and v["views"]["in_stock"]["exclude"] is None
    assert v["views"]["in_stock"]["include"]["file"] == str(views / "in_stock.include.txt")
    assert v["views"]["in_stock"]["include"]["ids"] == 4 and v["views"]["no_30"]["allowed"] == 5
    local = model.view(**dict(zip(("include", "exclude"), read_view(views, "in_stock"))))
    no_30 = model.view(**dict(zip(("include", "exclude"), read_view(views, "no_30"))))
    for h in ([20, 40], [10, 50, 70]):
        assert _same(client.recommend(h, k=5, sorted=True, view="in_stock"), local.recommend(h, k=5, sorted=True))
        both = client.recommend(h, k=5, sorted=True, view="in_stock", views=["no_30"])
        assert _same(both, local.recommend(h, k=5, sorted=True, views=[no_30]))
        assert np.array_equal(client.score(h, [30, 40], view="in_stock"), local.score(h, [30, 40]))
    # a rewrite is served from the next scan
    write_view(views, "in_stock", include=[60])
    _wait(lambda: client.views()["views"]["in_stock"]["allowed"] == 1, "the rewrite")
    assert client.recommend([10, 50, 70], k=5, view="in_stock")[0].tolist() == [60]


def test_threads_share_a_client(client, model):
    hists = [[10, 20], [30], [40, 50, 60], [70, 10]] * 8
    with ThreadPoolExecutor(8) as pool:
        got = list(pool.map(lambda h: client.recommend(h, k=3, sorted=True), hists))
    assert all(_same(g, model.recommend(h, k=3, sorted=True)) for g, h in zip(got, hists))
    assert len(client._conns) <= 8
    client.close()
    assert len(client._conns) == 0 and client.recommend([20], k=1)[0].tolist() == [30]  # reopens
    assert len(client._conns) == 1  # tracked: the next close() closes it


def test_idle_connection_closed_by_the_server_is_replaced(client):
    """timeouts.header_read = 1 s: the server closes an idle kept-alive connection; the next
    call reconnects once instead of failing."""
    assert client.recommend([20], k=1)[0].tolist() == [30]
    time.sleep(1.5)
    assert client.recommend([20], k=1)[0].tolist() == [30]


def test_health_ready_and_connection_failures(server):
    with ServerClient(server[0]) as c:
        assert c.health() and c.ready()
    dead = ServerClient("http://127.0.0.1:9", timeout=1.0)  # discard port: refused
    assert not dead.health() and not dead.ready()
    with pytest.raises(ConnectionRefusedError):
        dead.recommend([1])
    retrying = ServerClient("http://127.0.0.1:9", timeout=1.0, max_retries=2)
    t = time.monotonic()
    with pytest.raises(ConnectionRefusedError):
        retrying.recommend([1])
    assert time.monotonic() - t >= 0.3  # 0.1 s + 0.2 s of backoff


def test_retries_a_draining_server_only(monkeypatch):
    """503 is tried again, other errors never are."""
    answers = []

    def fake_send(self, method, path, body, request_id):
        status = answers.pop(0)
        return status, {}, b'{"error": {"code": "x", "message": "m", "request_id": "r"}}'

    monkeypatch.setattr(ServerClient, "_send", fake_send)
    c = ServerClient(max_retries=2)
    answers[:] = [503, 503, 400]
    with pytest.raises(ServerError) as e:
        c.views()
    assert e.value.status == 400 and answers == []
    answers[:] = [400, 200]
    with pytest.raises(ServerError):
        c.views()
    assert answers == [200]
    answers[:] = [503, 503, 503]
    with pytest.raises(ServerError) as e:
        c.views()
    assert e.value.status == 503 and answers == []


def test_not_http_or_cut_answers_are_connection_errors(tmp_path):
    """A peer that isn't HTTP, or an answer cut short: ConnectionError (an OSError, retried),
    and health() is False."""
    import socket

    def serve_once(reply):
        s = socket.create_server(("127.0.0.1", 0))

        def run():
            for _ in range(2):
                conn, _ = s.accept()
                conn.recv(65536)
                conn.sendall(reply)
                conn.close()
            s.close()

        threading.Thread(target=run, daemon=True).start()
        return f"http://127.0.0.1:{s.getsockname()[1]}"

    for reply in (b"+OK redis\r\n", b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{"):
        url = serve_once(reply)
        with pytest.raises(ConnectionError):
            ServerClient(url).views()
        assert not ServerClient(url).health()


def test_non_json_ok_answer(monkeypatch):
    monkeypatch.setattr(ServerClient, "_send", lambda *a: (200, {}, b"<html>portal</html>"))
    with pytest.raises(ServerError) as e:
        ServerClient().views()
    assert (e.value.status, e.value.code, e.value.message) == (200, None, "<html>portal</html>")


def test_finished_threads_leave_no_open_connection(client):
    import gc

    for _ in range(5):
        t = threading.Thread(target=client.recommend, args=([20],))
        t.start()
        t.join()
    gc.collect()
    assert len(client._conns) == 0  # each went with its thread


def test_non_json_error_answer(monkeypatch):
    """An error from something in front of the server (a proxy): code None, the text as message."""
    monkeypatch.setattr(ServerClient, "_send", lambda *a: (502, {"x-request-id": "p-1"}, b"<html>Bad Gateway</html>"))
    with pytest.raises(ServerError) as e:
        ServerClient().recommend([1])
    assert (e.value.status, e.value.code, e.value.message, e.value.request_id) == (
        502,
        None,
        "<html>Bad Gateway</html>",
        "p-1",
    )


# write_view / read_view: the core's format, no server needed


def test_write_view_files(tmp_path):
    paths = write_view(tmp_path, "eu", include=[3, -1, 2**62], exclude=np.array([7], dtype=np.int32))
    assert paths == [tmp_path / "eu.include.txt", tmp_path / "eu.exclude.txt"]
    assert paths[0].read_text() == f"3\n-1\n{2**62}\n" and paths[1].read_text() == "7\n"
    inc, exc = read_view(tmp_path, "eu")
    assert inc.dtype == np.int64 and inc.tolist() == [3, -1, 2**62] and exc.tolist() == [7]
    # one rule rewritten, the other left as it is; an empty list is a rule too
    write_view(tmp_path, "eu", include=[])
    assert [x.tolist() for x in read_view(tmp_path, "eu")] == [[], [7]]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["eu.exclude.txt", "eu.include.txt"]  # no temp left
    (tmp_path / "eu.include.txt").unlink()
    assert read_view(tmp_path, "eu")[0] is None


def test_read_view_reads_what_the_server_reads(tmp_path):
    (tmp_path / "a.include.txt").write_text("﻿# from the stock job\n 5 \n\n  # x\n6\n")
    assert read_view(tmp_path, "a")[0].tolist() == [5, 6]
    (tmp_path / "b.exclude.txt").write_text("5\n6 # inline\n")
    with pytest.raises(ValueError, match=r"b.exclude.txt:2: .* is not an integer id"):
        read_view(tmp_path, "b")


def test_view_file_errors(tmp_path):
    for bad in ("", "Caps", "a/b", "../x", "x" * 65, "a.b"):
        with pytest.raises(ValueError, match="view name"):
            write_view(tmp_path, bad, include=[1])
        with pytest.raises(ValueError, match="view name"):
            read_view(tmp_path, bad)
    with pytest.raises(ValueError, match="include, exclude or both"):
        write_view(tmp_path, "a")
    with pytest.raises(TypeError, match="integer ids"):
        write_view(tmp_path, "a", include=[1.0])
    with pytest.raises(FileNotFoundError):
        write_view(tmp_path / "missing", "a", include=[1])
    with pytest.raises(FileNotFoundError, match="no view"):
        read_view(tmp_path, "a")
    write_view(tmp_path, "m", include=[1])
    (tmp_path / "m.meta.json").write_text("{}")
    with pytest.raises(ValueError, match="later version"):
        read_view(tmp_path, "m")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.include.txt", "m.meta.json"]


def test_write_view_from_threads(tmp_path):
    """Concurrent writers of one view: each file is whole, one of the writes."""

    def write(i):
        write_view(tmp_path, "v", include=np.full(1000, i))

    ts = [threading.Thread(target=write, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    ids = read_view(tmp_path, "v")[0]
    assert ids.size == 1000 and len(set(ids.tolist())) == 1
    assert [p.name for p in tmp_path.iterdir()] == ["v.include.txt"]

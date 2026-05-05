from __future__ import annotations

import pytest

from emtp_solver_v3 import NodeBook, NodeIndexer


def test_node_book_ground_aliases_and_string_allocation():
    book = NodeBook()
    assert book.get("GND") == 0
    assert book.get("ground") == 0
    assert book.get("0") == 0

    n_top = book.get("tower.top")
    assert n_top > 0
    assert book.get("tower.top") == n_top
    assert book.name_of(n_top) == "tower.top"

    n_reserved = book.reserve("bus.A", 42)
    assert n_reserved == 42
    assert book.get("bus.A") == 42
    assert book.alias("bus.A.alias", "bus.A") == 42


def test_node_indexer_compacts_sparse_external_ids_and_freezes():
    idx = NodeIndexer()
    assert idx.register(0) == NodeIndexer.COMPACT_GND
    assert idx.register(10) == 0
    assert idx.register(999) == 1
    assert idx.register(50000) == 2
    assert idx.n == 3
    assert idx.to_compact(999) == 1
    assert idx.to_external(2) == 50000

    idx.freeze()
    assert idx.register(999) == 1
    with pytest.raises(RuntimeError):
        idx.register(123456)

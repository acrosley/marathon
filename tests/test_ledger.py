import dataclasses

import pytest

from marathon.ledger import Ledger, LedgerError


def build(n=5):
    ledger = Ledger()
    for i in range(n):
        ledger.append({"turn": i, "content": f"state {i}"})
    return ledger


def test_append_links_chain():
    ledger = build()
    assert len(ledger) == 5
    assert ledger[0].parent is None
    for i in range(1, 5):
        assert ledger[i].parent == ledger[i - 1].chain_hash
    ledger.verify()


def test_head():
    ledger = Ledger()
    assert ledger.head is None
    snap = ledger.append({"a": 1})
    assert ledger.head == snap


def test_tamper_detection_state():
    ledger = build()
    snap = ledger[2]
    ledger._snapshots[2] = dataclasses.replace(snap, state={"turn": 2, "content": "FORGED"})
    with pytest.raises(LedgerError):
        ledger.verify()


def test_tamper_detection_chain():
    ledger = build()
    snap = ledger[3]
    ledger._snapshots[3] = dataclasses.replace(snap, parent="sha256:" + "0" * 64)
    with pytest.raises(LedgerError):
        ledger.verify()


def test_jsonl_round_trip(tmp_path):
    ledger = build(8)
    path = tmp_path / "ledger.jsonl"
    ledger.to_jsonl(path)
    loaded = Ledger.from_jsonl(path)
    assert len(loaded) == 8
    assert loaded.head.chain_hash == ledger.head.chain_hash


def test_jsonl_load_rejects_tampered_file(tmp_path):
    ledger = build(3)
    path = tmp_path / "ledger.jsonl"
    ledger.to_jsonl(path)
    text = path.read_text(encoding="utf-8").replace("state 1", "FORGED")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(LedgerError):
        Ledger.from_jsonl(path)

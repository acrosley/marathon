from marathon.canonical import canonical_bytes, digest, serialize_history, snapshot_hash


def test_key_order_invariance():
    assert canonical_bytes({"b": 1, "a": 2}) == canonical_bytes({"a": 2, "b": 1})


def test_no_whitespace_variance():
    assert canonical_bytes({"a": [1, 2]}) == b'{"a":[1,2]}'


def test_digest_format_and_stability():
    h = digest(b"marathon")
    assert h.startswith("sha256:") and len(h) == 7 + 64
    assert h == digest(b"marathon")
    assert h != digest(b"marathon!")


def test_snapshot_hash_matches_canonical_digest():
    obj = {"x": [1, {"y": "z"}]}
    assert snapshot_hash(obj) == digest(canonical_bytes(obj))


def test_history_serialization_is_append_only():
    history = [{"turn": i, "content": f"message {i}"} for i in range(10)]
    full = serialize_history(history)
    for k in range(len(history) + 1):
        prefix = serialize_history(history[:k])
        assert full.startswith(prefix)


def test_unicode_is_byte_stable():
    obj = {"text": "héllo — 日本語 🚀"}
    assert canonical_bytes(obj) == canonical_bytes(obj)
    assert "é" in canonical_bytes(obj).decode("utf-8")

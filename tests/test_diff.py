import random

import pytest

from marathon.diff import Copy, Delta, DeltaError, Insert, apply_delta, compute_delta


def round_trip(base: bytes, target: bytes, block_size: int = 64) -> Delta:
    delta = compute_delta(base, target, block_size)
    assert apply_delta(base, delta) == target
    # wire round-trip too
    assert apply_delta(base, Delta.from_dict(delta.to_dict())) == target
    return delta


def test_identical_content_is_mostly_copies():
    base = bytes(range(256)) * 40
    delta = round_trip(base, base)
    assert delta.insert_bytes == 0
    assert delta.copy_bytes == len(base)


def test_empty_cases():
    round_trip(b"", b"")
    round_trip(b"", b"hello world")
    round_trip(b"hello world", b"")


def test_target_shorter_than_block():
    round_trip(b"x" * 1000, b"tiny", block_size=64)


def test_append_only_growth_is_cheap():
    base = b"A" * 10_000
    target = base + b"B" * 100
    delta = round_trip(base, target)
    # everything old is copied; only the appended tail is inserted
    assert delta.insert_bytes <= 100 + 64
    assert len(delta.wire_bytes()) < len(target) / 20


def test_mid_edit_preserves_matching_after_edit_point():
    rng = random.Random(1)
    base = bytes(rng.randrange(256) for _ in range(20_000))
    # edit near the front — the case that kills prefix caching
    target = base[:500] + b"[EDIT]" + base[500:]
    delta = round_trip(base, target)
    assert delta.insert_bytes < 700  # unchanged tail must still byte-match
    assert len(delta.wire_bytes()) < len(target) / 10


@pytest.mark.parametrize("seed", range(8))
def test_randomized_mutations_round_trip(seed):
    rng = random.Random(seed)
    base = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 8000)))
    target = bytearray(base)
    for _ in range(rng.randrange(0, 12)):
        kind = rng.choice(["insert", "delete", "edit"])
        if not target and kind != "insert":
            continue
        pos = rng.randrange(0, len(target) + 1)
        if kind == "insert":
            target[pos:pos] = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 300)))
        elif kind == "delete":
            del target[pos : pos + rng.randrange(1, 300)]
        else:
            end = min(len(target), pos + rng.randrange(1, 100))
            target[pos:end] = bytes(rng.randrange(256) for _ in range(end - pos))
    round_trip(base, bytes(target), block_size=rng.choice([16, 32, 64, 128]))


def test_apply_rejects_out_of_range_copy():
    with pytest.raises(DeltaError):
        apply_delta(b"short", Delta((Copy(0, 100),)))


def test_wire_format_v0():
    delta = Delta((Copy(0, 5), Insert(b"hi")), block_size=64)
    d = delta.to_dict()
    assert d["v"] == 0 and d["ops"][0] == ["c", 0, 5]
    assert Delta.from_dict(d) == delta


def test_from_dict_rejects_unknown_version():
    with pytest.raises(DeltaError):
        Delta.from_dict({"v": 99, "block_size": 64, "ops": []})

"""CPU-only tests for the pure parts of position-shifted KV reuse.

The load-bearing claim is that a RoPE'd key computed at position ``p`` can be moved
to ``p + delta`` by one extra rotation, exactly. Everything else in ``kvshift`` is
plumbing around that identity.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from marathon.kvshift import (  # noqa: E402
    Span,
    byte_span,
    rerotate_keys,
    rotate_half,
    token_span,
)


def _tiny():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    config = Qwen3Config(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=2,
        intermediate_size=64,
        vocab_size=64,
        max_position_embeddings=512,
    )
    return config, Qwen3RotaryEmbedding(config)


def _rope(rot, k, start):
    pos = torch.arange(start, start + k.shape[2])[None]
    cos, sin = rot(k, pos)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return k * cos + rotate_half(k) * sin


@pytest.mark.parametrize("delta", [-7, -1, 0, 1, 13, 64])
def test_rerotate_matches_recompute_at_shifted_position(delta):
    config, rot = _tiny()
    torch.manual_seed(0)
    k = torch.randn(1, config.num_key_value_heads, 11, config.head_dim)
    start = 100
    shifted = rerotate_keys(_rope(rot, k, start), delta, rot.inv_freq)
    assert torch.allclose(shifted, _rope(rot, k, start + delta), atol=1e-5, rtol=1e-4)


def test_rerotate_is_invertible():
    config, rot = _tiny()
    k = _rope(rot, torch.randn(1, 2, 5, config.head_dim), 42)
    back = rerotate_keys(rerotate_keys(k, 9, rot.inv_freq), -9, rot.inv_freq)
    assert torch.allclose(back, k, atol=1e-5, rtol=1e-4)


def test_token_span_finds_the_edit():
    old = [1, 2, 3, 4, 5, 6, 7]
    new = [1, 2, 9, 9, 9, 6, 7]
    assert token_span(old, new) == Span(p=2, e_old=3, e_new=3, s=2)
    grown = [1, 2, 9, 9, 9, 9, 6, 7]
    span = token_span(old, grown)
    assert (span.p, span.delta, span.s, span.new_len) == (2, 1, 2, len(grown))


def test_token_span_identical_sequences():
    span = token_span([1, 2, 3], [1, 2, 3])
    assert span.e_old == span.e_new == 0 and span.delta == 0


def test_byte_span_agrees_with_the_delta_engine():
    base = (b"unchanged prefix " * 20) + b"MIDDLE" + (b" unchanged suffix" * 20)
    target = base.replace(b"MIDDLE", b"MIDDLE-EDITED")
    head, tail = byte_span(base, target)
    assert head > 0 and tail > 0
    assert base[:head] == target[:head]
    assert base[len(base) - tail:] == target[len(target) - tail:]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU-only stitching test")
def test_stitch_shapes():
    from marathon.kvshift import stitch

    inv = 1.0 / (10000 ** (torch.arange(0, 8, 2).float() / 8))
    kv = [(torch.randn(1, 2, 20, 8), torch.randn(1, 2, 20, 8))]
    span = Span(p=5, e_old=4, e_new=6, s=11)
    cache = stitch(kv, span, tail=3, inv_freq=inv)
    assert cache.layers[0].keys.shape[-2] == span.new_len + 3
    assert torch.allclose(cache.layers[0].values[:, :, :5], kv[0][1][:, :, :5])
    assert torch.allclose(cache.layers[0].values[:, :, 11:22], kv[0][1][:, :, 9:20])

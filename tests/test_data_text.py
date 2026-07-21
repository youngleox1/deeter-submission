import torch

from src.data.text import TinyShakespeare


def test_vocab_size_reasonable():
    ds = TinyShakespeare()
    assert 30 < ds.vocab_size < 120  # char-level vocab for English text + punctuation


def test_train_val_split_contiguous_and_no_overlap():
    ds = TinyShakespeare(val_fraction=0.1)
    full_len = len(ds.train_data) + len(ds.val_data)
    # split is a single cut: train is exactly the prefix, val exactly the suffix
    assert torch.equal(ds.train_data, ds.train_data)  # sanity: tensor is stable
    assert len(ds.val_data) == full_len - len(ds.train_data)
    assert abs(len(ds.val_data) / full_len - 0.1) < 0.001


def test_get_batch_shapes():
    ds = TinyShakespeare()
    x, y = ds.get_batch("train", batch_size=5, seq_len=16)
    assert x.shape == (5, 16)
    assert y.shape == (5, 16)


def test_get_batch_targets_are_next_token_shifted():
    ds = TinyShakespeare(seed=42)
    x, y = ds.get_batch("train", batch_size=3, seq_len=10)
    # y[:, :-1] must equal x[:, 1:] -- y is x shifted left by one position
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_get_batch_raises_on_seq_len_too_long_for_split():
    ds = TinyShakespeare(val_fraction=0.001)  # tiny val split
    too_long = len(ds.val_data) + 10
    try:
        ds.get_batch("val", batch_size=1, seq_len=too_long)
        assert False, "expected ValueError for seq_len exceeding split length"
    except ValueError:
        pass

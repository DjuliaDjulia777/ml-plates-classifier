import pytest

from plates.train import split_indices


def test_split_is_disjoint_and_complete():
    train, val = split_indices(total=50, val_split=0.2, seed=42)
    assert len(val) == 10 and len(train) == 40
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(50))


def test_split_is_reproducible():
    assert split_indices(50, 0.2, seed=1) == split_indices(50, 0.2, seed=1)


def test_split_depends_on_seed():
    assert split_indices(100, 0.2, seed=1) != split_indices(100, 0.2, seed=2)


def test_split_rejects_empty_validation():
    with pytest.raises(ValueError):
        split_indices(total=3, val_split=0.1, seed=0)

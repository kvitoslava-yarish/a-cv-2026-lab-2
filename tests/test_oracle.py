import inspect

import numpy as np
import pytest

from src.oracle import Oracle
from src.selection import SELECTORS, check_batch


def make_oracle():
    ids = ["a", "b", "c"]
    return Oracle(ids, np.zeros((3, 4, 4), dtype=np.uint8), root=None)


def test_query_logs_and_returns_masks():
    o = make_oracle()
    masks = o.query(["a", "c"])
    assert masks.shape == (2, 4, 4)
    assert o.log == [["a", "c"]] and o.labeled == {"a", "c"}


def test_refuses_repeat_and_duplicates():
    o = make_oracle()
    o.query(["a"])
    with pytest.raises(ValueError):
        o.query(["a"])
    with pytest.raises(ValueError):
        o.query(["b", "b"])
    with pytest.raises(KeyError):
        o.query(["zzz"])


def test_selectors_never_get_oracle():
    for fn in SELECTORS.values():
        assert list(inspect.signature(fn).parameters) == ["state", "labeled_idx", "candidate_idx", "B"]


def test_check_batch():
    check_batch([3, 4], [1, 2], 2)
    with pytest.raises(AssertionError):
        check_batch([3, 3], [1], 2)
    with pytest.raises(AssertionError):
        check_batch([1, 4], [1], 2)

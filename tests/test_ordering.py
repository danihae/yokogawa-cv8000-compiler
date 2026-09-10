import logging

import pytest

from compiler.processing import _chronological_order

# real run3 BeginTime strings: 6- and 7-digit fractions
RUN3 = [
    "2023-05-03T13:40:03.160167+02:00",
    "2023-05-04T14:00:34.0750864+02:00",
    "2023-05-02T14:54:15.0845387+02:00",
]


def test_mixed_fraction_digits_sort_chronologically():
    assert _chronological_order(RUN3) == [2, 0, 1]


def test_sorted_input_is_identity():
    ordered = [RUN3[i] for i in (2, 0, 1)]
    assert _chronological_order(ordered) == [0, 1, 2]


def test_utc_offsets_are_honoured():
    # string order says [0, 1]; 23:30+02:00 is 21:30 UTC, i.e. earlier
    times = ["2023-05-02T22:00:00+00:00", "2023-05-02T23:30:00+02:00"]
    assert _chronological_order(times) == [1, 0]


def test_seventh_fraction_digit_is_not_truncated():
    times = ["2023-05-02T14:54:15.0000001+02:00", "2023-05-02T14:54:15.0000000+02:00"]
    assert _chronological_order(times) == [1, 0]


def test_duplicate_begin_time_raises():
    with pytest.raises(ValueError, match="Duplicate"):
        _chronological_order([RUN3[0], RUN3[1], RUN3[0]])


def test_same_instant_different_offset_is_duplicate():
    with pytest.raises(ValueError, match="Duplicate"):
        _chronological_order(["2023-05-02T12:00:00+00:00", "2023-05-02T14:00:00+02:00"])


def test_unparseable_falls_back_to_string_order(caplog):
    with caplog.at_level(logging.WARNING, logger="compiler"):
        assert _chronological_order(["b", "a", "not-a-date"]) == [1, 0, 2]
    assert "string order" in caplog.text

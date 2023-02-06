# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test results utils."""

from pytest import mark
from qiskit.result import Counts

from staged_primitives.utils.results.counts import (
    bitflip_counts,
    bitmask_counts,
    map_counts,
)


################################################################################
## TESTS
################################################################################
class TestMapCounts:
    """Test map counts."""

    @mark.parametrize(
        "counts, map, expected",
        [
            ({}, lambda _: None, {}),
            ({0: 1}, lambda _: 0, {0: 1}),
            ({0: 1}, lambda _: 1, {1: 1}),
            ({0: 1, 1: 1}, lambda _: 1, {1: 2}),
            ({0: 0, 1: 1}, lambda k: k + 1, {1: 0, 2: 1}),
        ],
    )
    def test_map_counts(self, counts, map, expected):
        """Test map counts base functionality."""
        counts = Counts(counts)
        assert map_counts(counts, map) == Counts(expected)


class TestBitflipCounts:
    """Test bitflip counts."""

    @mark.parametrize(
        "counts, bitflips, expected",
        [
            ({0: 0, 1: 1}, 0, {0: 0, 1: 1}),
            ({0: 0, 1: 1}, 1, {0: 1, 1: 0}),
            ({0: 0, 1: 1}, 2, {2: 0, 3: 1}),
            ({0: 0, 1: 1}, 3, {2: 1, 3: 0}),
            ({0: 0, 1: 1, 2: 2, 3: 3}, 0, {0: 0, 1: 1, 2: 2, 3: 3}),
            ({0: 0, 1: 1, 2: 2, 3: 3}, 1, {0: 1, 1: 0, 2: 3, 3: 2}),
            ({0: 0, 1: 1, 2: 2, 3: 3}, 2, {0: 2, 1: 3, 2: 0, 3: 1}),
            ({0: 0, 1: 1, 2: 2, 3: 3}, 3, {0: 3, 1: 2, 2: 1, 3: 0}),
        ],
    )
    def test_bitflip_counts(self, counts, bitflips, expected):
        """Test bitflip counts base functionality."""
        counts = Counts(counts)
        assert bitflip_counts(counts, bitflips) == Counts(expected)


class TestMaskCounts:
    """Test mask counts."""

    @mark.parametrize(
        "counts, mask, expected",
        [
            ({0: 0, 1: 1}, 0, {0: 1}),
            ({0: 0, 1: 1}, 1, {0: 0, 1: 1}),
            ({0: 0, 1: 1}, 2, {0: 1}),
            ({0: 0, 1: 1}, 3, {0: 0, 1: 1}),
            ({0: 0, 1: 1, 2: 2, 3: 3}, 0, {0: 6}),
            ({0: 0, 1: 1, 2: 2, 3: 3}, 1, {0: 2, 1: 4}),
            ({0: 0, 1: 1, 2: 2, 3: 3}, 2, {0: 1, 2: 5}),
            ({0: 0, 1: 1, 2: 2, 3: 3}, 3, {0: 0, 1: 1, 2: 2, 3: 3}),
        ],
    )
    def test_bitmask_counts(self, counts, mask, expected):
        """Test mask counts base functionality."""
        counts = Counts(counts)
        assert bitmask_counts(counts, mask) == Counts(expected)

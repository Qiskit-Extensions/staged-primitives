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

from stprimitives.utils.results import bitflip_counts, mask_counts


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
    def test_mask_counts(self, counts, mask, expected):
        """Test mask counts base functionality."""
        counts = Counts(counts)
        assert mask_counts(counts, mask) == Counts(expected)

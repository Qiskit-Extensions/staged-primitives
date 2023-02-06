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

"""Tests for binary utils."""

from __future__ import annotations

from pytest import mark

from staged_primitives.utils.binary import binary_digit, parity_bit


################################################################################
## TESTS
################################################################################
class TestParityBit:
    """Test parity bit."""

    @mark.parametrize(
        "integer, expected_even", [(0, 0), (1, 1), (2, 1), (3, 0), (4, 1), (5, 0), (6, 0), (7, 1)]
    )
    def test_parity_bit(self, integer, expected_even):
        """Test parity bit base functionality."""
        assert parity_bit(integer, even=True) == expected_even
        assert parity_bit(integer, even=False) == 0 if expected_even else 1


class TestBinaryDigit:
    """Test binary digit."""

    @mark.parametrize(
        "integer, place, expected", [(0b100, 0, 0), (0b100, 1, 0), (0b100, 2, 1), (0b100, 3, 0)]
    )
    def test_binary_digit(self, integer, place, expected):
        """Test binary digit base functionality."""
        assert binary_digit(integer, place) == expected

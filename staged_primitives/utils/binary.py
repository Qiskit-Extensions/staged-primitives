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

"""Binary utils."""

from __future__ import annotations


################################################################################
## UTILS
################################################################################
def binary_digit(value: int, place: int) -> bool:
    """Retrieves single binary digit for the given input value and place.

    Example:
        >>> binary_digit(0b100, 0)
        0
        >>> binary_digit(0b100, 1)
        0
        >>> binary_digit(0b100, 2)
        1
        >>> binary_digit(0b100, 3)
        0
    """
    mask = 2**place
    return True if (value & mask) else False  # pylint: disable=simplifiable-if-expression


def parity_bit(integer: int, even: bool = True) -> bool:
    """Return the parity bit for a given integer."""
    even_bit = bin(integer).count("1") % 2
    return bool(even_bit) if even else (not even_bit)

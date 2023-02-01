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
def parity_bit(integer: int, even: bool = True) -> int:
    """Return the parity bit for a given integer."""
    even_bit = bin(integer).count("1") % 2
    return even_bit if even else int(not even_bit)

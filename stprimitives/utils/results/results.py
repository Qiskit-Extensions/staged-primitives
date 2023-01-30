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

"""Results utils."""

from __future__ import annotations

from collections import defaultdict

from qiskit.result import Counts


################################################################################
## UTILS
################################################################################
def bitflip_counts(counts: Counts, bitflips: int) -> Counts:
    """Flip readout bits in counts according to the input bitflips (int encoded).

    Args:
        counts: the counts to process.
        bitflips: the bitflips to be applied.

    Returns:
        New counts with readout bits flipped according to input.
    """
    counts_dict: dict[int, int] = defaultdict(lambda: 0)
    for readout, freq in counts.int_raw.items():
        readout ^= bitflips
        counts_dict[readout] += freq
    return Counts(counts_dict)


def mask_counts(counts: Counts, bitmask: int) -> Counts:
    """Apply mask to readout bits in counts.

    Args:
        counts: the counts to process.
        bitmask: the bit mask to be applied.

    Returns:
        New counts with readout bits masked according to input.
    """
    counts_dict: dict[int, int] = defaultdict(lambda: 0)
    for readout, freq in counts.int_raw.items():
        readout &= bitmask
        counts_dict[readout] += freq
    return Counts(counts_dict)

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

"""Quantum circuits utils."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit


################################################################################
## UTILS
################################################################################
def compose_circuits_w_metadata(*circuits: QuantumCircuit, inplace: bool = False) -> QuantumCircuit:
    """Compose quantum circuits merging metadata."""
    # TODO: `circuit.compose(qc, inplace=True)` return `self` (i.e. Qiskit-Terra)
    # Note: implementation can be simplified after above TODOs using `functools.reduce`
    # composition = reduce(lambda base, next: base.compose(next, inplace=True), circuits)
    # composition.metadata = {k: v for c in circuits for k, v in (c.metadata or {}).items()}
    composition = circuits[0]
    circuits = circuits[1:]
    if not inplace:
        composition = composition.copy()
    # TODO: default `QuantumCircuit.metadata` to {} (i.e. Qiskit-Terra)
    if composition.metadata is None:
        composition.metadata = {}
    for circuit in circuits:
        composition.compose(circuit, inplace=True)
        composition.metadata.update(circuit.metadata or {})
    return composition

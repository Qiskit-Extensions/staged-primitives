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

from qiskit.circuit import Measure, QuantumCircuit, Qubit


################################################################################
## UTILS
################################################################################
def get_measured_qubits(circuit: QuantumCircuit) -> set[Qubit]:
    """Get qubits with at least one measurement gate in them."""
    return {qargs[0] for gate, qargs, _ in circuit if isinstance(gate, Measure)}

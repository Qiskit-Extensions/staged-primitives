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

"""Quantum operators utils."""

from __future__ import annotations

from functools import reduce
from typing import Any

from numpy import arange, bool_, dtype, ndarray, packbits
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.operators import Pauli


################################################################################
## UTILS
################################################################################
# TODO: `QuantumCircuit.measure_pauli(pauli)` (i.e. Qiskit-Terra)
# TODO: skip Pauli I measurements as they always evaluate to one
# TODO: insert pre-transpiled gates to avoid re-transpilation
# TODO: cache
def build_pauli_measurement(pauli: Pauli) -> QuantumCircuit:
    """Build measurement circuit for a given Pauli operator.

    Note: if Pauli is I for all qubits, this function generates a circuit to measure
    only the first qubit. Regardless of whether the result of that only measurement
    is zero or one, the associated expectation value will always evaluate to plus one.
    Therefore, such measurment can be interpreted as a constant (1) and does not need
    to be performed. We leave this behavior as default nonetheless.
    """
    measured_qubit_indices = arange(pauli.num_qubits)[pauli.z | pauli.x]
    measured_qubit_indices = set(measured_qubit_indices.tolist()) or {0}
    circuit = QuantumCircuit(pauli.num_qubits, len(measured_qubit_indices))
    for cbit, qubit in enumerate(measured_qubit_indices):
        if pauli.x[qubit]:
            if pauli.z[qubit]:
                circuit.sdg(qubit)
            circuit.h(qubit)
        circuit.measure(qubit, cbit)
    return circuit


def pauli_integer_mask(pauli: Pauli) -> int:
    """Build integer mask for input Pauli.

    This is an integer representation of the binary string with a
    1 where there are Paulis, and 0 where there are identities.
    """
    pauli_mask: ndarray[Any, dtype[bool_]] = pauli.z | pauli.x
    packed_mask: list[int] = packbits(  # pylint: disable=no-member
        pauli_mask, bitorder="little"
    ).tolist()
    return reduce(lambda value, element: (value << 8) | element, reversed(packed_mask))

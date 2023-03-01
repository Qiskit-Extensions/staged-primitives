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

"""Tests for quantum circuits utils."""

from __future__ import annotations

from pytest import mark
from qiskit.circuit import QuantumCircuit

from staged_primitives.utils.circuits.measurement import get_measured_qubits


################################################################################
## TESTS
################################################################################
class TestGetMeasuredQubits:
    """Test get measured qubits."""

    @mark.parametrize(
        "num_qubits, measured_qubits",
        [
            (1, ()),
            (1, (0,)),
            (2, (0,)),
            (2, (1,)),
            (2, (0, 1)),
            (2, (1, 0)),
            (3, (0,)),
            (3, (1,)),
            (3, (2,)),
            (3, (0, 1)),
            (3, (1, 0)),
            (3, (0, 2)),
            (3, (2, 0)),
            (3, (1, 2)),
            (3, (2, 1)),
            (3, (0, 1, 2)),
            (3, (2, 0, 1)),
            (3, (1, 2, 0)),
            (3, (2, 1, 0)),
            (3, (0, 2, 1)),
            (3, (1, 0, 2)),
        ],
    )
    def test_get_measured_qubits(self, num_qubits, measured_qubits):
        """Test get measured qubits base functionality."""
        num_cbits = len(measured_qubits)
        qc = QuantumCircuit(num_qubits, num_cbits)
        qc.measure(measured_qubits, range(num_cbits))
        measured_qubits = set(qc.qubits[i] for i in measured_qubits)
        assert get_measured_qubits(qc) == measured_qubits

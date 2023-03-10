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

from pytest import mark, raises
from qiskit.circuit import QuantumCircuit

from staged_primitives.utils.circuits.layout import (
    infer_end_layout,
    infer_end_layout_intlist,
    layout_from_intlist,
    transpile_to_layout,
)

################################################################################
## CASES
################################################################################
LAYOUT_NUM_INTLIST_TUPLES = [  # (target_num_qubits, layout_intlist)
    (2, (0,)),
    (2, (1,)),
    (2, (0, 1)),
    (2, (1, 0)),
    (3, (0,)),
    (3, (1,)),
    (3, (2,)),
    (3, (0, 1)),
    (3, (0, 2)),
    (3, (1, 0)),
    (3, (1, 2)),
    (3, (2, 0)),
    (3, (2, 1)),
    (3, (0, 1, 2)),
    (3, (1, 2, 0)),
    (3, (2, 0, 1)),
    (3, (2, 1, 0)),
    (3, (1, 0, 2)),
    (3, (0, 2, 1)),
]


################################################################################
## TESTS
################################################################################
class TestLayoutFromIntlist:
    """Test layout form intlist."""

    @mark.parametrize("target_num_qubits, layout_intlist", LAYOUT_NUM_INTLIST_TUPLES)
    def test_layout_from_intlist(self, target_num_qubits, layout_intlist):
        """Test layout form intlist base functionality."""
        num_qubits = len(layout_intlist)
        circuit = QuantumCircuit(num_qubits)
        layout = layout_from_intlist(circuit, layout_intlist, target_num_qubits)
        assert len(layout) == target_num_qubits
        virtual_qubits = layout.get_virtual_bits()
        assert tuple(virtual_qubits[q] for q in circuit.qubits) == tuple(layout_intlist)
        assert set(virtual_qubits) == set(circuit.qubits)

    @mark.parametrize("layout_intlist", [(0,), (0, 1), (0, 2), (1, 2), (2, 4)])
    def test_no_target_num_qubits(self, layout_intlist):
        """Test default target number of qubits if none provided."""
        num_qubits = len(layout_intlist)
        circuit = QuantumCircuit(num_qubits)
        layout = layout_from_intlist(circuit, layout_intlist)
        assert len(layout) == max(layout_intlist) + 1

    @mark.parametrize(
        "num_qubits, layout_intlist",
        [
            (2, (0,)),
            (2, (1,)),
            (3, (0,)),
            (3, (0, 2)),
            (4, (0, 2, 1)),
        ],
    )
    def test_incompatible_circuit_intlist(self, num_qubits, layout_intlist):
        """Test error is raised if intlist does not have one entry per virtual qubit."""
        circuit = QuantumCircuit(num_qubits)
        with raises(ValueError):
            _ = layout_from_intlist(circuit, layout_intlist)

    @mark.parametrize(
        "num_qubits, target_num_qubits",
        [
            (2, 1),
            (3, 1),
            (3, 2),
            (4, 1),
            (4, 2),
            (4, 3),
        ],
    )
    def test_incompatible_circuit_target(self, num_qubits, target_num_qubits):
        """Test error is raised if less physical qubits than virtual."""
        circuit = QuantumCircuit(num_qubits)
        with raises(ValueError):
            _ = layout_from_intlist(circuit, range(num_qubits), target_num_qubits)

    @mark.parametrize(
        "target_num_qubits, layout_intlist",
        [
            (1, (1,)),
            (1, (0, 1)),
            (2, (2,)),
            (2, (0, 2)),
            (2, (1, 2)),
            (2, (1, 0, 2)),
        ],
    )
    def test_incompatible_intlist_target(self, target_num_qubits, layout_intlist):
        """Test error is raised if intlist spans more qubits than target allows."""
        num_qubits = len(layout_intlist)
        circuit = QuantumCircuit(num_qubits)
        with raises(ValueError):
            _ = layout_from_intlist(circuit, layout_intlist, target_num_qubits)


class TestTranspileToLayout:
    """Test transpilation to layout."""

    @mark.parametrize("target_num_qubits, layout_intlist", LAYOUT_NUM_INTLIST_TUPLES)
    def test_transpile_to_layout(self, target_num_qubits, layout_intlist):
        """Test transpilation to layout base functionality."""
        num_qubits = len(layout_intlist)
        circuit = QuantumCircuit(num_qubits)
        layout = layout_from_intlist(circuit, layout_intlist, target_num_qubits)
        circuit = transpile_to_layout(circuit, layout)
        assert circuit.final_layout == layout


class TestInferEndLayoutIntlist:
    """Test infer end layout."""

    @mark.parametrize("target_num_qubits, layout_intlist", LAYOUT_NUM_INTLIST_TUPLES)
    def test_infer_end_layout_intlist(self, target_num_qubits, layout_intlist):
        """Test infer end layout base functionality."""
        num_qubits = len(layout_intlist)
        circuit = QuantumCircuit(num_qubits)
        circuit.measure_all()
        layout = layout_from_intlist(circuit, layout_intlist, target_num_qubits)
        transpiled_circuit = transpile_to_layout(circuit, layout)
        assert infer_end_layout_intlist(circuit, transpiled_circuit) == layout_intlist


class TestInferEndLayout:
    """Test infer end layout."""

    @mark.parametrize("target_num_qubits, layout_intlist", LAYOUT_NUM_INTLIST_TUPLES)
    def test_infer_end_layout(self, target_num_qubits, layout_intlist):
        """Test infer end layout base functionality."""
        num_qubits = len(layout_intlist)
        circuit = QuantumCircuit(num_qubits)
        circuit.measure_all()
        layout = layout_from_intlist(circuit, layout_intlist, target_num_qubits)
        transpiled_circuit = transpile_to_layout(circuit, layout)
        assert infer_end_layout(circuit, transpiled_circuit) == layout

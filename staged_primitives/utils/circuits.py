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

from collections.abc import Iterator, Sequence
from typing import Any

from qiskit.circuit import Measure, QuantumCircuit, Qubit
from qiskit.transpiler import Layout, PassManager
from qiskit.transpiler.passes import ApplyLayout, SetLayout


################################################################################
## UTILS
################################################################################
def layout_from_intlist(
    circuit: QuantumCircuit,
    layout_intlist: Sequence[int],
    target_num_qubits: int | None = None,
) -> Layout:
    """Build layout from intlist.

    Args:
        circuit: to build the layout form its (virtual) qubits.
        layout_intlist: indeces represent virtual qubits, and values physical qubits.
        target_num_qubtis: the number of qubits for the target layout.

    Return:
        Layout object mapping between the qubits in the input circuit and physical qubits
        (represented by integer indeces in `range(target_num_qubtis)`).
    """
    target_num_qubits = target_num_qubits or max(layout_intlist) + 1
    # TODO: validation
    # if not isinstance(circuit, QuantumCircuit):
    #     raise TypeError("Invalid circuit: expected QuantumCircuit.")
    # if any(not isinstance(i, int) or i < 0 for i in layout_intlist):
    #     raise TypeError("Invalid element(s) in intlist: expected semi-positive integers.")
    # if len(layout_intlist) > len(set(layout_intlist)):
    #     raise ValueError("Repeted elements in provided intlist.")
    # if not isinstance(target_num_qubits, int):
    #     raise TypeError("Invalid target number of qubits: expected int.")
    if circuit.num_qubits != len(layout_intlist):
        raise ValueError("Circuit incompatible with requested layout.")
    if circuit.num_qubits > target_num_qubits:
        raise ValueError("Circuit incompatible with requested target.")
    if target_num_qubits <= max(layout_intlist):
        raise ValueError("Layout intlist spans more qubits than target.")
    layout_dict = dict.fromkeys(range(target_num_qubits))
    layout_dict.update(dict(zip(layout_intlist, circuit.qubits)))
    return Layout(layout_dict)


def transpile_to_layout(circuit: QuantumCircuit, layout: Layout):
    """Transpile circuit to match a given layout.

    Args:
        circuit: virtual circuit to transpile.
        layout: mapping between the virtual qubits in input circuit and physical qubits.

    Returns:
        A transpiled circuit with input layout applied.
    """
    passes = [SetLayout(layout=layout), ApplyLayout()]
    pass_manager = PassManager(passes=passes)
    transpiled = pass_manager.run(circuit)
    transpiled.final_layout = layout  # TODO: update after Qiskit-Terra 0.24
    return transpiled


def infer_end_layout_intlist(
    original_circuit: QuantumCircuit, transpiled_circuit: QuantumCircuit
) -> tuple[int, ...]:
    """Retrieve end layout intlist of physical qubits.

    For every virtual qubit in the original circuit returns the index of the physical qubit
    that it is mapped to at the end of the transpiled circuit.

    Note: Works under the assumption that the original circuit has a `measure_all`
    instruction at its end, and that the transpiler does not affect the classical
    registers.
    """
    return tuple(_generate_end_layout_intlist(original_circuit, transpiled_circuit))


def infer_end_layout(
    original_circuit: QuantumCircuit, transpiled_circuit: QuantumCircuit
) -> Layout:
    """Retrieve end layout from original and transpiled circuits (all measured).

    Note: Works under the assumption that the original circuit has a `measure_all`
    instruction at its end, and that the transpiler does not affect the classical
    registers.
    """
    physical_qubits = _generate_end_layout_intlist(original_circuit, transpiled_circuit)
    layout_dict: dict[int, Any] = dict.fromkeys(range(transpiled_circuit.num_qubits))
    layout_dict.update(dict(zip(physical_qubits, original_circuit.qubits)))
    return Layout(layout_dict)


def get_measured_qubits(circuit: QuantumCircuit) -> set[Qubit]:
    """Get qubits with at least one measurement gate in them."""
    return {qargs[0] for gate, qargs, _ in circuit if isinstance(gate, Measure)}


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


################################################################################
## AUXILIARY
################################################################################
def _generate_end_layout_intlist(
    original_circuit: QuantumCircuit, transpiled_circuit: QuantumCircuit
) -> Iterator[int]:
    """Generate end layout intlist of physical qubits.

    For every virtual qubit in the original circuit yields the index of the physical qubit
    that it is mapped to at the end of the transpiled circuit.

    Note: Works under the assumption that the original circuit has a `measure_all`
    instruction at its end, and that the transpiler does not affect the classical
    registers.
    """
    # TODO: raise error if assumption in docstring is not met
    qubit_index_map = {qubit: i for i, qubit in enumerate(transpiled_circuit.qubits)}
    num_measurements: int = original_circuit.num_qubits
    for i in range(-num_measurements, 0):
        _, qargs, _ = transpiled_circuit[i]  # Note: measurement instruction
        physical_qubit = qargs[0]
        yield qubit_index_map[physical_qubit]

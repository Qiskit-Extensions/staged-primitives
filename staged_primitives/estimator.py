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

"""Staged backend Estimator."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from copy import copy

from numpy import array, sqrt
from qiskit.circuit import Measure, QuantumCircuit
from qiskit.compiler import transpile
from qiskit.primitives import EstimatorResult
from qiskit.providers import Backend, BackendV2, BackendV2Converter
from qiskit.providers import JobV1 as Job
from qiskit.providers import Options
from qiskit.quantum_info.operators import PauliList, SparsePauliOp
from qiskit.result import Counts, Result
from qiskit.transpiler import PassManager, StagedPassManager

from staged_primitives.base import BaseStagedEstimator
from staged_primitives.utils.circuits import (
    compose_circuits_w_metadata,
    infer_end_layout_intlist,
)
from staged_primitives.utils.operators import (
    AbelianDecomposer,
    NaiveDecomposer,
    OperatorDecomposer,
    build_pauli_measurement,
)
from staged_primitives.utils.results import (
    CanonicalReckoner,
    ExpvalReckoner,
    ReckoningResult,
)


class StagedEstimator(
    BaseStagedEstimator
):  # pylint: disable=abstract-method,too-many-instance-attributes
    """Backend based Estimator with a staged pipeline for ease of extension.

    Evaluates expectation value by building several measured circuits whose resulting
    counts can be combined to reconstruct the result of an overall observable.
    Each of these circuits corresponds to Pauli term(s) in such observable, inserted
    as rotation gates and measurements after the corresponding base circuit.

    The :class:`~.StagedEstimator` class is a generic implementation of the
    :class:`~.BaseEstimator` interface through the more specific
    :class:`~.BaseStagedEstimator` interface that is used to provide a well defined
    pipeline for solving expectation value problems from backend counts.

    Note: The generic nature of this class precludes doing any provider- or backend-specific
    optimizations. Use native implementations for improved performance.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        group_commuting: bool = True,
        skip_transpilation: bool = False,
        # TODO: (unbound_)pass_manager: PassManager | None = None,
        bound_pass_manager: PassManager | None = None,
        options: dict | None = None,  # TODO: consistent naming
    ) -> None:
        """Initialize a new StagedEstimator instance.

        Args:
            backend: The backend to run the primitive on.
            group_commuting: Whether commuting observable components should be grouped.
            skip_transpilation: If `True`, transpilation of the input circuits is skipped.
            bound_pass_manager: An optional pass manager to run after parameter binding.
            options: Default execution/run options.
        """
        self.backend = backend
        self.group_commuting = group_commuting
        self.skip_transpilation = skip_transpilation  # TODO: transpilation level
        self.bound_pass_manager = bound_pass_manager
        self._transpile_options = Options()  # TODO: Primitives Options class
        super().__init__(circuits=None, observables=None, parameters=None, options=options)

    ################################################################################
    ## PROPERTIES
    ################################################################################
    # TODO: update property to quality
    @property
    def backend(self) -> BackendV2:
        """Backend to use for circuit execution."""
        return self._backend

    @backend.setter
    def backend(self, backend: Backend) -> None:
        # TODO: clear all transpilation caching
        if not isinstance(backend, Backend):
            raise TypeError(f"Expected `Backend` type, got `{type(backend)}` instead.")
        if not isinstance(backend, BackendV2):
            backend = BackendV2Converter(backend)
        self._backend: BackendV2 = backend

    @property
    def group_commuting(self) -> bool:
        """Groups commuting observable components."""
        return getattr(self, "_group_commuting", True)

    @group_commuting.setter
    def group_commuting(self, group_commuting: bool) -> None:
        self._group_commuting = bool(group_commuting)

    @property
    def skip_transpilation(self) -> bool:
        """If ``True``, transpilation of the input circuits is skipped."""
        return getattr(self, "_skip_transpilation", False)

    @skip_transpilation.setter
    def skip_transpilation(self, skip_transpilation: bool) -> None:
        self._skip_transpilation = bool(skip_transpilation)

    @property
    def bound_pass_manager(self) -> PassManager:
        """Pass manager to run after parameter binding."""
        return getattr(self, "_bound_pass_manager", StagedPassManager())

    @bound_pass_manager.setter
    def bound_pass_manager(self, pass_manager: PassManager | None) -> None:
        if pass_manager is None:
            pass_manager = StagedPassManager()
        elif not isinstance(pass_manager, PassManager):
            raise TypeError(f"Expected `PassManager` type, got `{type(pass_manager)}` instead.")
        self._bound_pass_manager: PassManager = pass_manager

    @property
    def transpile_options(self) -> Options:
        """Options for transpiling the input circuits."""
        return getattr(self, "_transpile_options", Options())  # TODO: Primitives Options class

    def set_transpile_options(self, **fields) -> None:
        """Set the transpiler options for transpiler.

        Args:
            **fields: The fields to update the options
        """
        self._transpile_options.update_options(**fields)

    @property
    def _operator_decomposer(self) -> OperatorDecomposer:
        """Observable decomposer based on object's config."""
        if self.group_commuting:
            return AbelianDecomposer()
        return NaiveDecomposer()

    @property
    def _expval_reckoner(self) -> ExpvalReckoner:
        """Strategy for expectation value reckoning."""
        return CanonicalReckoner()

    ################################################################################
    ## IMPLEMENTATION
    ################################################################################
    def _transpile_single_unbound(self, circuit: QuantumCircuit) -> QuantumCircuit:
        # TODO: use native `layout` attr after Qiskit-Terra 0.24 release
        # Note: We currently need to use a hacky way to account for the end
        # layout of the transpiled circuit. We insert temporary measurements
        # to keep track of the repositioning of the different qubits.
        original_circuit = circuit.copy()  # To insert measurements
        original_circuit.measure_all()  # To keep track of the final layout
        if self.skip_transpilation:
            transpiled_circuit = original_circuit
        else:
            transpile_options = {**self.transpile_options.__dict__}
            transpiled_circuit = transpile(original_circuit, self.backend, **transpile_options)
        end_layout_intlist = infer_end_layout_intlist(original_circuit, transpiled_circuit)
        transpiled_circuit.remove_final_measurements()
        # TODO: default `QuantumCircuit.metadata` to {} (i.e. Qiskit-Terra)
        if transpiled_circuit.metadata is None:
            transpiled_circuit.metadata = {}
        transpiled_circuit.metadata.update({"end_layout_intlist": end_layout_intlist})
        return transpiled_circuit

    def _bind_single_parameters(
        self, circuit: QuantumCircuit, parameter_values: Sequence[float]
    ) -> QuantumCircuit:
        # Note: for improved performance this method edits the input circuits in place.
        # This is fine as long as the input circuits are no longer needed.
        circuit.assign_parameters(parameter_values, inplace=True)
        return circuit

    def _transpile_single_bound(self, circuit: QuantumCircuit) -> QuantumCircuit:
        # TODO: update `end_layout_intlist`
        # TODO: if not self.skip_transpilation:
        circuit = self.bound_pass_manager.run(circuit)
        return circuit

    def _observe_single_circuit(
        self, circuit: QuantumCircuit, observable: SparsePauliOp
    ) -> tuple[QuantumCircuit, ...]:
        layout_intlist = (circuit.metadata or {}).get("end_layout_intlist", None)
        # TODO: if len(layout_intlist) != observable.num_qubits: raise ValueError (and more)
        measurements = (
            self._transpile_single_measurement(meas, layout_intlist)
            for meas in self._generate_measurement_circuits(observable)
        )
        return tuple(compose_circuits_w_metadata(circuit, m) for m in measurements)

    def _execute(self, circuits: Sequence[QuantumCircuit], **run_options) -> list[Counts]:
        # Conversion
        circuits = list(circuits)  # TODO: accept Sequences in `backend.run()` (i.e. Qiskit-Terra)

        # Extract metadata (might be affected in `backend.run`)
        metadata_list = [copy(getattr(c, "metadata", {})) for c in circuits]

        # Max circuits
        total_circuits: int = len(circuits)
        max_circuits: int = getattr(self.backend, "max_circuits", None) or total_circuits or 1

        # Raw results
        jobs: tuple[Job, ...] = tuple(
            self.backend.run(circuits[split : split + max_circuits], **run_options)
            for split in range(0, total_circuits, max_circuits)
        )
        raw_results: tuple[Result, ...] = tuple(job.result() for job in jobs)

        # Annotated counts
        job_counts_iter = (
            job_counts if isinstance(job_counts, list) else [job_counts]
            for job_counts in (result.get_counts() for result in raw_results)
        )
        counts_iter = (counts for job_counts in job_counts_iter for counts in job_counts)
        counts_list: list[Counts] = []
        for counts, metadata in zip(counts_iter, metadata_list):
            # TODO: add `Counts.metadata` attr (i.e. Qiskit-Terra)
            counts.metadata = metadata
            counts_list.append(counts)

        return counts_list

    def _build_single_result(self, counts_list: Sequence[Counts]) -> EstimatorResult:
        expval, std_error = self._reckon_expval(counts_list)
        num_circuits = len(counts_list)
        shots = sum(counts.shots() for counts in counts_list)
        shots_per_circuit = shots / (num_circuits or 1)  # Note: avoid division by zero errors
        variance = shots_per_circuit * std_error**2
        metadatum = {
            "variance": variance,
            "std_dev": sqrt(variance),
            "std_error": std_error,
            "shots": shots,
            "shots_per_circuit": shots_per_circuit,
            "num_circuits": num_circuits,
        }
        return EstimatorResult(values=array([expval]), metadata=[metadatum])

    ################################################################################
    ## AUXILIARY
    ################################################################################
    # TODO: caching
    def _generate_measurement_circuits(
        self,
        observable: SparsePauliOp,
    ) -> Iterator[QuantumCircuit]:
        """Generate all appendage quantum circuits necessary to measure a given observable.

        This will yield one measurement circuit per singly measurable component of the
        observable (i.e. measurable with a single quantum circuit), as retrieved from the
        instance's `operator_decomposer` attribute. Said component will be simplified
        (i.e. removing common identities) and annotated in each measurement circuit's
        metadata for reference —under `observable`. The indices of the measured qubits
        will also be annotated in metadata under `measured_qubit_indices`.

        Notice that the annotated observable can be made out of different components itself,
        but they will all share a single common basis in the form of a Pauli operator; in
        order to be measured simultaneously (e.g. `ZZ` and `ZI`, or `XI` and `IX`).
        """
        # TODO: (component, basis) pairs
        for component in self._operator_decomposer.decompose(observable):
            basis = self._operator_decomposer.extract_pauli_bases(component)[0]
            circuit: QuantumCircuit = build_pauli_measurement(basis)
            measured_qubit_indices = tuple(
                circuit.qubits.index(qargs[0])
                for gate, qargs, _ in circuit
                if isinstance(gate, Measure)
            )
            # Simplified Paulis (keep only measured qubits)
            paulis = PauliList.from_symplectic(
                component.paulis.z[:, measured_qubit_indices],
                component.paulis.x[:, measured_qubit_indices],
                component.paulis.phase,
            )
            # TODO: observable does not need to be hermitian: rename
            circuit.metadata = {
                "observable": SparsePauliOp(paulis, component.coeffs),
                "measured_qubit_indices": measured_qubit_indices,
            }
            yield circuit

    def _transpile_single_measurement(
        self,
        measurement: QuantumCircuit,
        layout_intlist: Sequence[int] | None,
    ) -> QuantumCircuit:
        """Transpile measurement circuit to backend and input layout.

        This method updates the `measured_qubit_indices` metadata annotation
        to match the transpiled layout. If `self.skip_transpilation` is True,
        input layout will not be applied.
        """
        # TODO: pre-transpile rotation gates
        transpile_options = {**self.transpile_options.__dict__}  # TODO: avoid multiple copies
        transpile_options.update({"initial_layout": layout_intlist})
        measured_qubit_indices = measurement.metadata.get("measured_qubit_indices")
        measured_qubit_indices = tuple(layout_intlist[i] for i in measured_qubit_indices)
        measurement.metadata.update({"measured_qubit_indices": measured_qubit_indices})
        # Note: metadata is preserved through transpilation
        return transpile(measurement, self.backend, **transpile_options)

    def _reckon_expval(self, counts_list: Sequence[Counts]) -> ReckoningResult:
        """Compute expectation value and associated std-error for a list of counts.

        Args:
            counts_list: sequence of counts annotated with an observable in their metadata.

        Returns:
            Expectation value and associated std-error of the sum of annotated observables.
        """
        observables = tuple(c.metadata["observable"] for c in counts_list)
        return self._expval_reckoner.reckon(counts_list, observables)

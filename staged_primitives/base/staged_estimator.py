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

"""Staged Estimator abstract base class."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

from numpy import array
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimator, EstimatorResult
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.providers import Backend
from qiskit.quantum_info import SparsePauliOp
from qiskit.result import Counts


class BaseStagedEstimator(BaseEstimator):  # pylint: disable=too-few-public-methods
    """Staged Estimator abstract base class."""

    def _run(
        self,
        circuits: tuple[QuantumCircuit, ...],
        observables: tuple[SparsePauliOp, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ) -> PrimitiveJob:
        job = PrimitiveJob(self._compute, circuits, observables, parameter_values, **run_options)
        job.submit()
        return job

    # TODO: support non-hermitain operators on top of observables
    def _compute(
        self,
        circuits: tuple[QuantumCircuit, ...],
        observables: tuple[SparsePauliOp, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ) -> EstimatorResult:
        """Estimate expectation values for the given circuit-observable pairs."""
        circuits = self._transpile_unbound(circuits)
        circuits = self._bind_parameters(circuits, parameter_values)
        circuits = self._transpile_bound(circuits)
        circuits_matrix = self._observe_circuits(circuits, observables)
        counts_matrix = self._execute_matrix(circuits_matrix, **run_options)
        results = self._build_results(counts_matrix)
        # TODO: return EstimatorResult.compose(results)
        value_metadata_pairs = ((v, m) for r in results for v, m in zip(r.values, r.metadata))
        values, metadata = tuple(zip(*value_metadata_pairs))
        return EstimatorResult(values=array(values), metadata=list(metadata))

    ################################################################################
    ## STAGES / PIPELINE
    ################################################################################
    def _transpile_unbound(self, circuits: Sequence[QuantumCircuit]) -> tuple[QuantumCircuit, ...]:
        """Traspile parametric quantum circuits (i.e. before binding parameters).

        Note: output circuits are annotated with the ``final_layout`` attribute.
        """
        return tuple(self._transpile_single_unbound(qc) for qc in circuits)

    def _bind_parameters(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
    ) -> tuple[QuantumCircuit, ...]:
        """Bind quantum circuit parameters (i.e. from parametric to non-parametric)."""
        return tuple(
            self._bind_single_parameters(circuit, values)
            for circuit, values in zip(circuits, parameter_values)
        )

    def _transpile_bound(self, circuits: Sequence[QuantumCircuit]) -> tuple[QuantumCircuit, ...]:
        """Transpile non-parametric quantum circuits (i.e. after binding all parameters)."""
        return tuple(self._transpile_single_bound(qc) for qc in circuits)

    def _observe_circuits(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[SparsePauliOp],
    ) -> tuple[tuple[QuantumCircuit, ...], ...]:
        """For each circuit-observable pair build all circuits necessary for expval estimation.

        Args:
            circuits: a sequence of transpiled quantum circuits to produce the state to observe.
            observables: an observable for each of the input quantum circuits.

        Returns:
            A matrix of quantum circuits where each row corresponds to all the circuits
            necessary to measure the corresponding observable in the state provided by
            its associated circuit. Each of these circuits has its metadata annotated with
            the observable component (i.e. :class:`~qiskit.quantum_info.SparsePauliOp`)
            which can be directly evaluated through it.

        Note: the number of qubits in the input ciruits and corresponding observables
        will not match (generally), since circuits have already been transpiled to the
        target architecture. Each circuit's `final_layout` attr will contain information
        about where to apply each measurement. This is done for efficiency reasons.
        """
        return tuple(
            self._observe_single_circuit(circuit, observable)
            for circuit, observable in zip(circuits, observables)
        )

    def _execute_matrix(
        self, circuit_matrix: Sequence[Sequence[QuantumCircuit]], **run_options
    ) -> tuple[tuple[Counts, ...], ...]:
        """Execute circuit matrix and return counts in identical (i.e. one-to-one) arrangement.

        Note: Each :class:`qiskit.result.Counts` object is annotated with the metadata
        from the circuit that produced it.
        """
        circuits = list(qc for group in circuit_matrix for qc in group)  # List for performance
        counts = self._execute(circuits, **run_options)
        counts_iter = iter(counts)
        counts_matrix = tuple(tuple(next(counts_iter) for _ in group) for group in circuit_matrix)
        return counts_matrix

    def _build_results(
        self, counts_matrix: Sequence[Sequence[Counts]]
    ) -> tuple[EstimatorResult, ...]:
        """Compute results from counts matrix.

        Args:
            counts_matrix: a sequence of counts groups, each group corresponding to one
                of the original circuit-observable pairs (i.e. one counts entry per term
                in the observable).

        Returns:
            A tuple of ``EstimatorResult`` objects each of which with the expectation value
            and metadata associated to a single circuit-observable pair.
        """
        return tuple(self._build_single_result(counts_list) for counts_list in counts_matrix)

    ################################################################################
    ## ABSTRACT METHODS
    ################################################################################
    @abstractmethod
    def _transpile_single_unbound(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Single circuit equivalent of ``_transpile_unbound``."""

    @abstractmethod
    def _bind_single_parameters(
        self, circuit: QuantumCircuit, parameter_values: Sequence[float]
    ) -> QuantumCircuit:
        """Single circuit equivalent of ``_bind_parameters``."""

    @abstractmethod
    def _transpile_single_bound(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Single circuit equivalent of ``_transpile_bound``."""

    @abstractmethod
    def _observe_single_circuit(
        self, circuit: QuantumCircuit, observable: SparsePauliOp
    ) -> tuple[QuantumCircuit, ...]:
        """Single circuit equivalent of ``_observe_circuit``."""

    @abstractmethod
    def _execute(self, circuits: Sequence[QuantumCircuit], **run_options) -> list[Counts]:
        """Flattened equivalent of ``_execute_matrix`` (i.e. list instead of matrix)."""

    @abstractmethod
    def _build_single_result(self, counts_list: Sequence[Counts]) -> EstimatorResult:
        """Single circuit-observable pair equivalent of ``_build_results``."""

    ################################################################################
    ## DEPRECATED
    ################################################################################
    # TODO: remove
    # Note: to allow `backend` as positional argument while deprecated args in place
    def __new__(  # pylint: disable=signature-differs
        cls,
        backend: Backend,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ):
        self = super().__new__(cls)  # pylint: disable=no-value-for-parameter
        return self

    def _call(
        self,
        circuits: Sequence[int],
        observables: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:  # pragma: no cover
        raise NotImplementedError("This method has been deprecated, use `run` instead.")

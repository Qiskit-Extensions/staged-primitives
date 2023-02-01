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

"""Staged Sampler abstract base class."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseSampler, SamplerResult
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.providers import Backend
from qiskit.result import Counts


class BaseStagedSampler(BaseSampler):  # pylint: disable=too-few-public-methods
    """Staged Sampler abstract base class."""

    def _run(
        self,
        circuits: tuple[QuantumCircuit, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ) -> PrimitiveJob:
        job = PrimitiveJob(self._compute, circuits, parameter_values, **run_options)
        job.submit()
        return job

    # TODO: support non-hermitain operators on top of observables
    def _compute(
        self,
        circuits: tuple[QuantumCircuit, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ) -> SamplerResult:
        """Sample quasi-probability distribution for the given circuits."""
        circuits = self._transpile_unbound(circuits)
        circuits = self._bind_parameters(circuits, parameter_values)
        circuits = self._transpile_bound(circuits)
        counts_list = self._execute(circuits, **run_options)
        results = self._build_results(counts_list)
        # TODO: return SamplerResult.compose(results)
        dist_metadata_pairs = ((d, m) for r in results for d, m in zip(r.quasi_dists, r.metadata))
        quasi_dists, metadata = tuple(zip(*dist_metadata_pairs))
        return SamplerResult(quasi_dists=list(quasi_dists), metadata=list(metadata))

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

    def _build_results(self, counts_list: Sequence[Counts]) -> tuple[SamplerResult, ...]:
        """Compute results from counts list.

        Args:
            counts_list: a sequence of counts, each group corresponding to one of the
                original circuits.

        Returns:
            A tuple of ``SamplerResult`` objects each of which with the quasi distribution
            and metadata associated to a single circuit.
        """
        return tuple(self._build_single_result(counts) for counts in counts_list)

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
    def _execute(self, circuits: Sequence[QuantumCircuit], **run_options) -> list[Counts]:
        """Execute circuits sequence and return counts in identical (i.e. one-to-one) arrangement.

        Note: Each :class:`qiskit.result.Counts` object is annotated with the metadata
        from the circuit that produced it.
        """

    @abstractmethod
    def _build_single_result(self, counts: Counts) -> SamplerResult:
        """Single counts equivalent of ``_build_results``."""

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
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> SamplerResult:  # pragma: no cover
        raise NotImplementedError("This method has been deprecated, use `run` instead.")

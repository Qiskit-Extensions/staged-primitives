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

"""Staged backend Sampler."""

from __future__ import annotations

from collections.abc import Sequence

from numpy import sqrt
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.primitives import SamplerResult
from qiskit.providers import Backend, BackendV2, BackendV2Converter
from qiskit.providers import JobV1 as Job
from qiskit.providers import Options
from qiskit.result import Counts, QuasiDistribution, Result
from qiskit.transpiler import PassManager, StagedPassManager

from stprimitives.base import BaseStagedSampler
from stprimitives.utils.circuits import infer_end_layout_intlist


class StagedSampler(
    BaseStagedSampler
):  # pylint: disable=abstract-method,too-many-instance-attributes
    """Backend based Sampler with a staged pipeline for ease of extension.

    Reconstructs quasi-probability distributions from measured circuits.

    The :class:`~.StagedSampler` class is a generic implementation of the
    :class:`~.BaseSampler` interface through the more specific
    :class:`~.BaseStagedSampler` interface that is used to provide a well defined
    pipeline for solving quasi-probability problems from backend counts.

    Note: The generic nature of this class precludes doing any provider- or backend-specific
    optimizations. Use native implementations for improved performance.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        skip_transpilation: bool = False,
        # TODO: (unbound_)pass_manager: PassManager | None = None,
        bound_pass_manager: PassManager | None = None,
        options: dict | None = None,  # TODO: consistent naming
    ) -> None:
        """Initialize a new StagedSampler instance.

        Args:
            backend: The backend to run the primitive on.
            skip_transpilation: If `True`, transpilation of the input circuits is skipped.
            bound_pass_manager: An optional pass manager to run after parameter binding.
            options: Default execution/run options.
        """
        self.backend = backend
        self.skip_transpilation = skip_transpilation  # TODO: transpilation level
        self.bound_pass_manager = bound_pass_manager
        self._transpile_options = Options()  # TODO: Primitives Options class
        super().__init__(circuits=None, parameters=None, options=options)

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

    def _execute(self, circuits: Sequence[QuantumCircuit], **run_options) -> list[Counts]:
        # Conversion
        circuits = list(circuits)  # TODO: accept Sequences in `backend.run()` (i.e. Qiskit-Terra)

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
        for counts, circuit in zip(counts_iter, circuits):
            # TODO: `Counts.metadata` attr (i.e. Qiskit-Terra)
            counts.metadata = circuit.metadata or {}
            counts_list.append(counts)

        return counts_list

    def _build_single_result(self, counts: Counts) -> SamplerResult:
        shots = counts.shots()
        std_dev = sqrt(1 / shots) if shots else None
        probabilities = {k: v / (shots or 1) for k, v in counts.int_outcomes().items()}
        dist = QuasiDistribution(probabilities, shots=shots, stddev_upper_bound=std_dev)
        metadatum = {"shots": shots}
        return SamplerResult(quasi_dists=[dist], metadata=[metadatum])

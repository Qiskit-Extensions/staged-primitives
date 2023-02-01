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

"""Tests for staged backend Sampler."""


from __future__ import annotations

from itertools import product
from unittest.mock import Mock, patch

from numpy import pi, sqrt
from numpy.random import default_rng
from pytest import fixture, mark, raises
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.providers import BackendV1, BackendV2
from qiskit.providers.fake_provider import FakeManhattan, FakeManhattanV2
from qiskit.result import Counts, QuasiDistribution
from qiskit.transpiler import PassManager

from staged_primitives.sampler import StagedSampler


################################################################################
## FIXTURES
################################################################################
@fixture(scope="function")
def backend_mock():
    """BackendV2 mock."""
    return Mock(BackendV2)


@fixture
def sampler(backend_mock):
    """StagedSampler object with mock backend and default settings."""
    return StagedSampler(backend_mock)


################################################################################
## TESTS
################################################################################
class TestProperties:
    """Test StagedSampler properties."""

    def test_backend(self, sampler):
        """Test backend."""
        backend = FakeManhattanV2()
        assert isinstance(backend, BackendV2)
        sampler.backend = backend
        assert sampler.backend is backend
        backend = FakeManhattan()
        assert isinstance(backend, BackendV1)
        sampler.backend = backend
        assert isinstance(sampler.backend, BackendV2)
        # TODO: assert sampler.backend == backend
        assert sampler.backend is not backend

    @mark.parametrize("backend", ["backend", None, Ellipsis, False])
    def test_backend_validation(self, sampler, backend):
        """Test backend validation."""
        with raises(TypeError):
            sampler.backend = backend

    def test_skip_transpilation(self, sampler):
        """Test skip_transpilation."""
        assert sampler.skip_transpilation is False  # Default
        sampler.skip_transpilation = True
        assert sampler.skip_transpilation is True
        sampler.skip_transpilation = False
        assert sampler.skip_transpilation is False

    @mark.parametrize("skip_transpilation", [0, 1, None, Ellipsis])
    def test_skip_transpilation_validation(self, sampler, skip_transpilation):
        """Test skip_transpilation validation."""
        sampler.skip_transpilation = skip_transpilation
        assert sampler.skip_transpilation is bool(skip_transpilation)

    def test_bound_pass_manager(self, sampler):
        """Test bound_pass_manager."""
        assert isinstance(sampler.bound_pass_manager, PassManager)
        assert sampler.bound_pass_manager.passes() == []  # Default
        pass_manager = PassManager()
        sampler.bound_pass_manager = pass_manager
        assert sampler.bound_pass_manager is pass_manager
        sampler.bound_pass_manager = None
        assert isinstance(sampler.bound_pass_manager, PassManager)
        assert sampler.bound_pass_manager.passes() == []

    @mark.parametrize("pass_manager", ["pass_manager", False, Ellipsis])
    def test_bound_pass_manager_validation(self, sampler, pass_manager):
        """Test bound_pass_manager validation."""
        with raises(TypeError):
            sampler.bound_pass_manager = pass_manager

    @mark.parametrize("options", [{}, {"foo": "bar"}, {"int": 4}])
    def test_transpile_options(self, sampler, options):
        """Test transpile_options setter."""
        assert sampler.transpile_options.__dict__ == {}
        sampler.set_transpile_options(**options)
        assert sampler.transpile_options.__dict__ == options


class TestImplementation:
    """Test StagedSampler implementation."""

    @mark.parametrize(
        "num_qubits, skip_transpilation, seed",
        ((num, skip, i) for i, (num, skip) in enumerate(product(range(1, 5), [False, True]))),
    )
    def test_transpile_single_unbound(self, sampler, num_qubits, skip_transpilation, seed):
        """Test transpile_single_unbound."""
        # Case
        sampler.skip_transpilation = skip_transpilation
        circuit = random_circuit(num_qubits, depth=2, seed=seed)
        # Test
        with patch("staged_primitives.sampler.transpile") as mock:
            circuit_mock = Mock()
            mock.side_effect = lambda *args, **kwargs: circuit_mock
            transpiled_circuit = sampler._transpile_single_unbound(circuit)
        if sampler.skip_transpilation:
            assert transpiled_circuit == circuit
        else:
            assert transpiled_circuit == circuit_mock

    @mark.parametrize("seed", range(10))
    def test_bind_single_parameters(self, sampler, seed):
        """Test bind_single_parameters."""
        ## Build case
        rng = default_rng(seed)
        num_qubits = rng.integers(10) + 1
        num_parameters = rng.integers(10) + 1
        circuit = QuantumCircuit(num_qubits)
        for p in range(num_parameters):
            target = rng.integers(num_qubits)
            circuit.rx(Parameter(str(p)), target)
        ## Test
        assert circuit.num_parameters == num_parameters
        parameter_values = rng.uniform(-pi, pi, size=num_parameters)
        bound_circuit = sampler._bind_single_parameters(circuit, parameter_values)
        assert bound_circuit is circuit  # Inplace binding, otherwise assert equal
        assert bound_circuit.num_parameters == 0
        for instruction, expected in zip(bound_circuit, parameter_values):
            gate = instruction[0]
            value = gate.params[0]
            assert value == expected

    def test_transpile_single_bound(self, sampler):
        """Test transpile_single_bound."""
        circuit = QuantumCircuit(4)
        bound_pass_manager = Mock(PassManager)
        sampler.bound_pass_manager = bound_pass_manager
        transpiled = sampler._transpile_single_bound(circuit)
        bound_pass_manager.run.assert_called_once_with(circuit)
        assert transpiled is bound_pass_manager.run.return_value

    @mark.parametrize("num_circuits", range(4))
    def test_execute(self, sampler, num_circuits):
        """Test execute."""
        # Case
        sampler.backend.max_circuits = 1  # Note: test backend limit is bypassed
        sampler.backend.run.side_effect = lambda *args, **kwargs: Mock()
        circuits = [QuantumCircuit(1) for _ in range(num_circuits)]
        for i, circuit in enumerate(circuits):
            circuit.metadata = {"index": i}  # Note: test metadata gets transferred
        # Test
        counts_list = sampler._execute(circuits, shots=12)
        for counts, circuit in zip(counts_list, circuits):
            assert counts.metadata == circuit.metadata

    @mark.parametrize(
        "counts, probabilities",
        [
            (Counts({}), {}),
            (Counts({0: 0}), {0: 0.0}),
            (Counts({0: 1}), {0: 1.0}),
            (Counts({1: 1}), {1: 1.0}),
            (Counts({0: 0, 1: 1}), {0: 0.0, 1: 1.0}),
            (Counts({0: 1, 1: 0}), {0: 1.0, 1: 0.0}),
            (Counts({0: 1, 1: 1}), {0: 0.5, 1: 0.5}),
            (Counts({0: 3, 1: 1}), {0: 0.75, 1: 0.25}),
            (Counts({0: 1, 1: 3}), {0: 0.25, 1: 0.75}),
            (Counts({1: 1, 7: 2, 3: 3, 4: 4}), {1: 0.1, 3: 0.3, 4: 0.4, 7: 0.2}),
            (Counts({256: 1, 32: 1, 128: 2, 64: 1}), {32: 0.2, 64: 0.2, 128: 0.4, 256: 0.2}),
        ],
    )
    def test_build_single_result(self, sampler, counts, probabilities):
        """Test build_single_result."""
        # Case
        shots = counts.shots()
        std_dev = sqrt(1 / shots) if shots else None
        dist = QuasiDistribution(probabilities, shots=shots, stddev_upper_bound=std_dev)
        metadata = {"shots": shots}
        # Test
        result = sampler._build_single_result(counts)
        assert len(result.quasi_dists) == len(result.metadata) == 1
        assert result.quasi_dists[0] == dist
        assert result.metadata[0]["shots"] == metadata["shots"]

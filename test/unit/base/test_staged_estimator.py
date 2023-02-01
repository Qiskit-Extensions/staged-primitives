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

"""Test BaseStagedEstimator."""

from itertools import product
from unittest.mock import Mock

from pytest import fixture, mark
from qiskit.primitives import EstimatorResult
from qiskit.providers import JobV1

from staged_primitives.base import BaseStagedEstimator


################################################################################
## FIXTURES
################################################################################
@fixture(scope="function")
def estimator():
    """BaseStagedEstimator fixture object."""
    namespace = {
        "backend": Mock(),  # TODO: remove after interface segregation
        "__init__": lambda *args, **kwargs: None,
        "_transpile_single_unbound": Mock(side_effect=lambda circuit: circuit),
        "_bind_single_parameters": Mock(side_effect=lambda circuit, parameters: circuit),
        "_transpile_single_bound": Mock(side_effect=lambda circuit: circuit),
        "_observe_single_circuit": Mock(side_effect=lambda circuit, observable: circuit),
        "_execute": Mock(side_effect=lambda circuits, **options: circuits),
        "_build_single_result": Mock(side_effect=lambda counts: counts),
    }
    cls = type("DummyStagedEstimator", (BaseStagedEstimator,), namespace)
    return cls("backend")


################################################################################
## TESTS
################################################################################
class TestPipeline:
    """Test BaseStagedEstimator pipeline."""

    @mark.parametrize("num_circuits, options", product(range(1, 9), [{}, {"foo": "bar"}]))
    def test_run(self, estimator, num_circuits, options):
        """Test run."""
        # Mock
        estimator._compute = Mock()
        # Case
        circuits = tuple(Mock() for _ in range(num_circuits))
        observables = tuple(Mock() for _ in range(num_circuits))
        parameter_values = tuple(Mock() for _ in range(num_circuits))
        # Test
        job = estimator._run(circuits, observables, parameter_values, **options)
        result = job.result()
        assert isinstance(job, JobV1)
        estimator._compute.assert_called_once_with(
            circuits, observables, parameter_values, **options
        )
        assert result == estimator._compute.return_value

    # TODO: move to integration tests
    @mark.parametrize("num_circuits, options", product(range(1, 9), [{}, {"foo": "bar"}]))
    def test_compute(self, estimator, num_circuits, options):
        """Test compute."""
        # Mock
        estimator._transpile_unbound = Mock(side_effect=lambda circuits: circuits)
        estimator._bind_parameters = Mock(side_effect=lambda circuits, parameters: circuits)
        estimator._transpile_bound = Mock(side_effect=lambda circuits: circuits)
        estimator._observe_circuits = Mock(side_effect=lambda circuits, observables: circuits)
        estimator._execute_matrix = Mock(side_effect=lambda matrix, **options: matrix)
        estimator._build_results = Mock(side_effect=lambda matrix: matrix)
        # Case
        result = Mock(values=[1], metadata=[{}])
        circuits = tuple(result for _ in range(num_circuits))
        observables = tuple(Mock() for _ in range(num_circuits))
        parameter_values = tuple(
            tuple(Mock() for _ in range(num_circuits * 2)) for _ in range(num_circuits)
        )
        # Test
        output = estimator._compute(circuits, observables, parameter_values, **options)
        estimator._transpile_unbound.assert_called_once_with(circuits)
        estimator._bind_parameters.assert_called_once_with(circuits, parameter_values)
        estimator._transpile_bound.assert_called_once_with(circuits)
        estimator._observe_circuits.assert_called_once_with(circuits, observables)
        estimator._execute_matrix.assert_called_once_with(circuits, **options)
        estimator._build_results.assert_called_once_with(circuits)
        assert isinstance(output, EstimatorResult)
        assert output.values.tolist() == [result.values[0] for _ in range(num_circuits)]
        assert output.metadata == [result.metadata[0] for _ in range(num_circuits)]


class TestStages:
    """Test BaseStagedEstimator stages."""

    @mark.parametrize("num_circuits", range(1, 9))
    def test_transpile_unbound(self, estimator, num_circuits):
        """Test transpile_unbound."""
        circuits = tuple(Mock() for _ in range(num_circuits))
        output = estimator._transpile_unbound(circuits)
        assert estimator._transpile_single_unbound.call_count == num_circuits
        for qc in circuits:
            estimator._transpile_single_unbound.assert_any_call(qc)
        assert output == circuits  # Note: thanks to side_effect

    @mark.parametrize("num_circuits", range(1, 9))
    def test_bind_parameters(self, estimator, num_circuits):
        """Test bind_parameters."""
        circuits = tuple(Mock() for _ in range(num_circuits))
        parameter_values = tuple(Mock() for _ in range(num_circuits))
        output = estimator._bind_parameters(circuits, parameter_values)
        assert estimator._bind_single_parameters.call_count == num_circuits
        for qc, params in zip(circuits, parameter_values):
            estimator._bind_single_parameters.assert_any_call(qc, params)
        assert output == circuits  # Note: thanks to side_effect

    @mark.parametrize("num_circuits", range(1, 9))
    def test_transpile_bound(self, estimator, num_circuits):
        """Test transpile_bound."""
        circuits = tuple(Mock() for _ in range(num_circuits))
        output = estimator._transpile_bound(circuits)
        assert estimator._transpile_single_bound.call_count == num_circuits
        for qc in circuits:
            estimator._transpile_single_bound.assert_any_call(qc)
        assert output == circuits  # Note: thanks to side_effect

    @mark.parametrize("num_circuits", range(1, 9))
    def test_observe_circuits(self, estimator, num_circuits):
        """Test observe_circuits."""
        circuits = tuple(Mock() for _ in range(num_circuits))
        observables = tuple(Mock() for _ in range(num_circuits))
        output = estimator._observe_circuits(circuits, observables)
        assert estimator._observe_single_circuit.call_count == num_circuits
        for qc, obs in zip(circuits, observables):
            estimator._observe_single_circuit.assert_any_call(qc, obs)
        assert output == circuits  # Note: thanks to side_effect

    @mark.parametrize("num_circuits, options", product(range(1, 9), [{}, {"foo": "bar"}]))
    def test_execute_matrix(self, estimator, num_circuits, options):
        """Test execute_matrix."""
        circuit_matrix = tuple(
            tuple(Mock() for _ in range(num_circuits * 2)) for _ in range(num_circuits)
        )
        output = estimator._execute_matrix(circuit_matrix, **options)
        estimator._execute.assert_called_once_with(
            [qc for group in circuit_matrix for qc in group], **options
        )
        assert output == circuit_matrix  # Note: thanks to side_effect

    @mark.parametrize("num_circuits", range(1, 9))
    def test_build_results(self, estimator, num_circuits):
        """Test build_results."""
        counts_matrix = tuple(
            tuple(Mock() for _ in range(num_circuits * 2)) for _ in range(num_circuits)
        )
        output = estimator._build_results(counts_matrix)
        assert estimator._build_single_result.call_count == num_circuits
        for counts_list in counts_matrix:
            estimator._build_single_result.assert_any_call(counts_list)
        assert output == counts_matrix  # Note: thanks to side_effect

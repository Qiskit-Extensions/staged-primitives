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

"""Test BaseStagedSampler."""

from itertools import product
from unittest.mock import Mock

from pytest import fixture, mark
from qiskit.primitives import SamplerResult
from qiskit.providers import JobV1

from staged_primitives.base import BaseStagedSampler


################################################################################
## FIXTURES
################################################################################
@fixture(scope="function")
def sampler():
    """BaseStagedSampler fixture object."""
    namespace = {
        "backend": Mock(),  # TODO: remove after interface segregation
        "__init__": lambda *args, **kwargs: None,
        "_transpile_single_unbound": Mock(side_effect=lambda circuit: circuit),
        "_bind_single_parameters": Mock(side_effect=lambda circuit, parameters: circuit),
        "_transpile_single_bound": Mock(side_effect=lambda circuit: circuit),
        "_execute": Mock(side_effect=lambda circuits, **options: circuits),
        "_build_single_result": Mock(side_effect=lambda counts: counts),
    }
    cls = type("DummyStagedSampler", (BaseStagedSampler,), namespace)
    return cls("backend")


################################################################################
## TESTS
################################################################################
class TestPipeline:
    """Test BaseStagedSampler pipeline."""

    # TODO: replace `product` for double `mark.parametrize`
    @mark.parametrize("num_circuits, options", product(range(1, 9), [{}, {"foo": "bar"}]))
    def test_run(self, sampler, num_circuits, options):
        """Test run."""
        # Mock
        sampler._compute = Mock()
        # Case
        circuits = tuple(Mock() for _ in range(num_circuits))
        parameter_values = tuple(Mock() for _ in range(num_circuits))
        # Test
        job = sampler._run(circuits, parameter_values, **options)
        result = job.result()
        assert isinstance(job, JobV1)
        sampler._compute.assert_called_once_with(circuits, parameter_values, **options)
        assert result == sampler._compute.return_value

    # TODO: move to integration tests
    @mark.parametrize("num_circuits, options", product(range(1, 9), [{}, {"foo": "bar"}]))
    def test_compute(self, sampler, num_circuits, options):
        """Test compute."""
        # Mock
        sampler._transpile_unbound = Mock(side_effect=lambda circuits: circuits)
        sampler._bind_parameters = Mock(side_effect=lambda circuits, parameters: circuits)
        sampler._transpile_bound = Mock(side_effect=lambda circuits: circuits)
        sampler._build_results = Mock(side_effect=lambda matrix: matrix)
        # Case
        result = Mock(quasi_dists=[Mock()], metadata=[{}])
        circuits = tuple(result for _ in range(num_circuits))
        parameter_values = tuple(
            tuple(Mock() for _ in range(num_circuits * 2)) for _ in range(num_circuits)
        )
        # Test
        output = sampler._compute(circuits, parameter_values, **options)
        sampler._transpile_unbound.assert_called_once_with(circuits)
        sampler._bind_parameters.assert_called_once_with(circuits, parameter_values)
        sampler._transpile_bound.assert_called_once_with(circuits)
        sampler._build_results.assert_called_once_with(circuits)
        assert isinstance(output, SamplerResult)
        assert output.quasi_dists == [result.quasi_dists[0] for _ in range(num_circuits)]
        assert output.metadata == [result.metadata[0] for _ in range(num_circuits)]


class TestStages:
    """Test BaseStagedSampler stages."""

    @mark.parametrize("num_circuits", range(1, 9))
    def test_transpile_unbound(self, sampler, num_circuits):
        """Test transpile_unbound."""
        circuits = tuple(Mock() for _ in range(num_circuits))
        output = sampler._transpile_unbound(circuits)
        assert sampler._transpile_single_unbound.call_count == num_circuits
        for qc in circuits:
            sampler._transpile_single_unbound.assert_any_call(qc)
        assert output == circuits  # Note: thanks to side_effect

    @mark.parametrize("num_circuits", range(1, 9))
    def test_bind_parameters(self, sampler, num_circuits):
        """Test bind_parameters."""
        circuits = tuple(Mock() for _ in range(num_circuits))
        parameter_values = tuple(Mock() for _ in range(num_circuits))
        output = sampler._bind_parameters(circuits, parameter_values)
        assert sampler._bind_single_parameters.call_count == num_circuits
        for qc, params in zip(circuits, parameter_values):
            sampler._bind_single_parameters.assert_any_call(qc, params)
        assert output == circuits  # Note: thanks to side_effect

    @mark.parametrize("num_circuits", range(1, 9))
    def test_transpile_bound(self, sampler, num_circuits):
        """Test transpile_bound."""
        circuits = tuple(Mock() for _ in range(num_circuits))
        output = sampler._transpile_bound(circuits)
        assert sampler._transpile_single_bound.call_count == num_circuits
        for qc in circuits:
            sampler._transpile_single_bound.assert_any_call(qc)
        assert output == circuits  # Note: thanks to side_effect

    @mark.parametrize("num_circuits", range(1, 9))
    def test_build_results(self, sampler, num_circuits):
        """Test build_results."""
        counts_matrix = tuple(
            tuple(Mock() for _ in range(num_circuits * 2)) for _ in range(num_circuits)
        )
        output = sampler._build_results(counts_matrix)
        assert sampler._build_single_result.call_count == num_circuits
        for counts_list in counts_matrix:
            sampler._build_single_result.assert_any_call(counts_list)
        assert output == counts_matrix  # Note: thanks to side_effect

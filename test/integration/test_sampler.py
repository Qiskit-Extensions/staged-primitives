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


from collections import namedtuple

from numpy import isclose, sqrt
from numpy.random import rand
from pytest import fixture, mark, raises
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import SamplerResult
from qiskit.providers import JobStatus, JobV1
from qiskit.providers.fake_provider import FakeNairobi, FakeNairobiV2
from qiskit.utils import optionals

from staged_primitives.sampler import StagedSampler

from . import TestFromQiskit, TestOnBackends

################################################################################
## FIXTURES
################################################################################
CircuitProbsPair = namedtuple("CircuitProbsPair", ["circuit", "probabilities"])


@fixture
def hadamard():
    """Hadamard circuit and expected probabilities."""
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()
    return CircuitProbsPair(qc, {0: 0.5, 1: 0.5})


@fixture
def bell():
    """Bell circuit and expected probabilities."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return CircuitProbsPair(qc, {0: 0.5, 3: 0.5, 1: 0, 2: 0})


################################################################################
## TESTS
################################################################################
class TestRun(TestOnBackends, TestFromQiskit):
    """Test running different payloads on StagedSampler."""

    def test_sampler_run(self, backend, bell):
        """Test Sampler.run()."""
        circuit, target = bell
        sampler = StagedSampler(backend=backend)
        job = sampler.run(circuits=circuit, shots=1000)
        assert isinstance(job, JobV1)
        result = job.result()
        assert isinstance(result, SamplerResult)
        assert result.quasi_dists[0].shots == 1000
        assert result.quasi_dists[0].stddev_upper_bound == sqrt(1 / 1000)
        compare_probs(result.quasi_dists, target)

    def test_sample_run_multiple_circuits(self, backend, bell):
        """Test Sampler.run() with multiple circuits."""
        # executes three Bell circuits
        # Argument `parameters` is optional.
        circuit, target = bell
        sampler = StagedSampler(backend=backend)
        result = sampler.run([circuit] * 3).result()
        # print([q.binary_probabilities() for q in result.quasi_dists])
        compare_probs(result.quasi_dists[0], target)
        compare_probs(result.quasi_dists[1], target)
        compare_probs(result.quasi_dists[2], target)

    def test_sampler_run_with_parameterized_circuits(self, backend):
        """Test Sampler.run() with parameterized circuits."""
        # parameterized circuits
        pqc = RealAmplitudes(num_qubits=2, reps=2)
        pqc.measure_all()
        pqc2 = RealAmplitudes(num_qubits=2, reps=3)
        pqc2.measure_all()

        theta1 = [0, 1, 1, 2, 3, 5]
        theta2 = [1, 2, 3, 4, 5, 6]
        theta3 = [0, 1, 2, 3, 4, 5, 6, 7]

        sampler = StagedSampler(backend=backend)
        result = sampler.run([pqc, pqc, pqc2], [theta1, theta2, theta3]).result()

        # result of pqc(theta1)
        prob1 = {
            "00": 0.1309248462975777,
            "01": 0.3608720796028448,
            "10": 0.09324865232050054,
            "11": 0.41495442177907715,
        }
        assert dicts_almost_equal(result.quasi_dists[0].binary_probabilities(), prob1, delta=0.1)

        # result of pqc(theta2)
        prob2 = {
            "00": 0.06282290651933871,
            "01": 0.02877144385576705,
            "10": 0.606654494132085,
            "11": 0.3017511554928094,
        }
        assert dicts_almost_equal(result.quasi_dists[1].binary_probabilities(), prob2, delta=0.1)

        # result of pqc2(theta3)
        prob3 = {
            "00": 0.1880263994380416,
            "01": 0.6881971261189544,
            "10": 0.09326232720582443,
            "11": 0.030514147237179892,
        }
        assert dicts_almost_equal(result.quasi_dists[2].binary_probabilities(), prob3, delta=0.1)

    def test_run_1qubit(self, backend):
        """test for 1-qubit cases"""
        qc = QuantumCircuit(1)
        qc.measure_all()
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        qc2.measure_all()

        sampler = StagedSampler(backend=backend)
        result = sampler.run([qc, qc2]).result()
        assert isinstance(result, SamplerResult)
        assert len(result.quasi_dists) == 2

        assert dicts_almost_equal(result.quasi_dists[0], {0: 1}, 0.1)
        assert dicts_almost_equal(result.quasi_dists[1], {1: 1}, 0.1)

    def test_run_2qubit(self, backend):
        """test for 2-qubit cases"""
        qc0 = QuantumCircuit(2)
        qc0.measure_all()
        qc1 = QuantumCircuit(2)
        qc1.x(0)
        qc1.measure_all()
        qc2 = QuantumCircuit(2)
        qc2.x(1)
        qc2.measure_all()
        qc3 = QuantumCircuit(2)
        qc3.x([0, 1])
        qc3.measure_all()

        sampler = StagedSampler(backend=backend)
        result = sampler.run([qc0, qc1, qc2, qc3]).result()
        assert isinstance(result, SamplerResult)
        assert len(result.quasi_dists) == 4

        assert dicts_almost_equal(result.quasi_dists[0], {0: 1}, 0.1)
        assert dicts_almost_equal(result.quasi_dists[1], {1: 1}, 0.1)
        assert dicts_almost_equal(result.quasi_dists[2], {2: 1}, 0.1)
        assert dicts_almost_equal(result.quasi_dists[3], {3: 1}, 0.1)

    def test_run_errors(self, backend):
        """Test for errors."""
        qc1 = QuantumCircuit(1)
        qc1.measure_all()
        qc2 = RealAmplitudes(num_qubits=1, reps=1)
        qc2.measure_all()

        sampler = StagedSampler(backend=backend)
        with raises(ValueError):
            sampler.run([qc1], [[1e2]]).result()
        with raises(ValueError):
            sampler.run([qc2], [[]]).result()
        with raises(ValueError):
            sampler.run([qc2], [[1e2]]).result()

    def test_run_empty_parameter(self, backend):
        """Test for empty parameter"""
        n = 5
        qc = QuantumCircuit(n, n - 1)
        qc.measure(range(n - 1), range(n - 1))
        sampler = StagedSampler(backend=backend)
        # with self.subTest("one circuit"):
        result = sampler.run([qc], shots=1000).result()
        assert len(result.quasi_dists) == 1
        for q_d in result.quasi_dists:
            quasi_dist = {k: v for k, v in q_d.items() if v != 0.0}
            assert dicts_almost_equal(quasi_dist, {0: 1.0}, delta=0.1)
        assert len(result.metadata) == 1

        # with self.subTest("two circuits"):
        result = sampler.run([qc, qc], shots=1000).result()
        assert len(result.quasi_dists) == 2
        for q_d in result.quasi_dists:
            quasi_dist = {k: v for k, v in q_d.items() if v != 0.0}
            assert dicts_almost_equal(quasi_dist, {0: 1.0}, delta=0.1)
        assert len(result.metadata) == 2

    def test_run_numpy_params(self, backend):
        """Test for numpy array as parameter values"""
        qc = RealAmplitudes(num_qubits=2, reps=2)
        qc.measure_all()
        k = 5
        params_array = rand(k, qc.num_parameters)
        params_list = params_array.tolist()
        params_list_array = list(params_array)
        sampler = StagedSampler(backend=backend)
        target = sampler.run([qc] * k, params_list).result()

        # with self.subTest("ndarrary"):
        result = sampler.run([qc] * k, params_array).result()
        assert len(result.metadata) == k
        for i in range(k):
            assert dicts_almost_equal(result.quasi_dists[i], target.quasi_dists[i], delta=0.1)

        # with self.subTest("list of ndarray"):
        result = sampler.run([qc] * k, params_list_array).result()
        assert len(result.metadata) == k
        for i in range(k):
            assert dicts_almost_equal(result.quasi_dists[i], target.quasi_dists[i], delta=0.1)

    @mark.filterwarnings("ignore:.*seed.*")
    def test_run_with_shots_option(self, backend):
        """test with shots option."""
        pqc = RealAmplitudes(num_qubits=2, reps=2)
        pqc.measure_all()
        params = [1.0] * 6
        target = {0: 0.0148, 1: 0.3449, 2: 0.0531, 3: 0.5872}
        sampler = StagedSampler(backend=backend)
        result = sampler.run(circuits=pqc, parameter_values=params, shots=1024, seed=15).result()
        compare_probs(result.quasi_dists, target)

    def test_primitive_job_status_done(self, backend, bell):
        """test primitive job's status"""
        circuit, _ = bell
        sampler = StagedSampler(backend=backend)
        job = sampler.run(circuits=[circuit])
        assert job.status() == JobStatus.DONE

    def test_sequential_run(self, backend):
        """Test sequential run."""
        qc = QuantumCircuit(1)
        qc.measure_all()
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        qc2.measure_all()
        sampler = StagedSampler(backend)
        result = sampler.run([qc]).result()
        assert dicts_almost_equal(result.quasi_dists[0], {0: 1}, 0.1)
        result2 = sampler.run([qc2]).result()
        assert dicts_almost_equal(result2.quasi_dists[0], {1: 1}, 0.1)
        result3 = sampler.run([qc, qc2]).result()
        assert dicts_almost_equal(result3.quasi_dists[0], {0: 1}, 0.1)
        assert dicts_almost_equal(result3.quasi_dists[1], {1: 1}, 0.1)


@mark.skipif(not optionals.HAS_AER, reason="Qiskit-Aer is required to run this test.")
class TestWithAer(TestFromQiskit):
    """Integrtion tests using Qiskit-Aer."""

    def test_circuit_with_dynamic_circuit(self):
        """Test StagedSampler with QuantumCircuit with a dynamic circuit"""
        from unittest.mock import Mock

        from qiskit.providers import BackendV2
        from qiskit_aer import Aer

        qc = QuantumCircuit(2, 1)

        with qc.for_loop(range(5)):
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            qc.break_loop().c_if(0, True)

        backend = Aer.get_backend("aer_simulator")
        backend.set_options(seed_simulator=15)
        sampler = StagedSampler(Mock(BackendV2), skip_transpilation=True)
        sampler._backend = backend  # TODO: BackendV2Converter fails for `aer_simulator`
        sampler.set_transpile_options(seed_transpiler=15)
        result = sampler.run(qc).result()
        assert dicts_almost_equal(result.quasi_dists[0], {0: 0.5029296875, 1: 0.4970703125})


class TestJobExecution(TestFromQiskit):
    """Test job execution under different backend configurations."""

    def test_primitive_job_size_limit_backend_v2(self):
        """Test primitive respects backend's job size limit."""

        class FakeNairobiLimitedCircuits(FakeNairobiV2):
            """FakeNairobiV2 with job size limit."""

            @property
            def max_circuits(self):
                return 1

        qc = QuantumCircuit(1)
        qc.measure_all()
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        qc2.measure_all()
        sampler = StagedSampler(backend=FakeNairobiLimitedCircuits())
        result = sampler.run([qc, qc2]).result()
        assert isinstance(result, SamplerResult)
        assert len(result.quasi_dists) == 2

        assert dicts_almost_equal(result.quasi_dists[0], {0: 1}, 0.1)
        assert dicts_almost_equal(result.quasi_dists[1], {1: 1}, 0.1)

    def test_primitive_job_size_limit_backend_v1(self):
        """Test primitive respects backend's job size limit."""
        backend = FakeNairobi()
        config = backend.configuration()
        config.max_experiments = 1
        backend._configuration = config
        qc = QuantumCircuit(1)
        qc.measure_all()
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        qc2.measure_all()
        sampler = StagedSampler(backend=backend)
        result = sampler.run([qc, qc2]).result()
        assert isinstance(result, SamplerResult)
        assert len(result.quasi_dists) == 2

        assert dicts_almost_equal(result.quasi_dists[0], {0: 1}, 0.1)
        assert dicts_almost_equal(result.quasi_dists[1], {1: 1}, 0.1)


################################################################################
## AUXILIARY
################################################################################
# TODO: make util and use for testing StagedEstimator metadata
def dicts_almost_equal(dict1, dict2, delta=None, places=None, default_value=0):
    """Test if two dictionaries with numeric values are almost equal.

    Note: this is an adaptation of Qiskit-Terra's testing function.

    Fail if the two dictionaries are unequal as determined by
    comparing that the difference between values with the same key are
    not greater than delta (default 1e-8), or that difference rounded
    to the given number of decimal places is not zero. If a key in one
    dictionary is not in the other the default_value keyword argument
    will be used for the missing value (default 0). If the two objects
    compare equal then they will automatically compare almost equal.

    Args:
        dict1 (dict): a dictionary.
        dict2 (dict): a dictionary.
        delta (number): threshold for comparison (defaults to 1e-8).
        places (int): number of decimal places for comparison.
        default_value (number): default value for missing keys.

    Raises:
        TypeError: if the arguments are not valid (both `delta` and
            `places` are specified).

    Returns:
        True if dictionaries are almost equal, False otherwise.
    """

    def valid_comparison(value):
        """compare value to delta, within places accuracy"""
        if places is not None:
            return round(value, places) == 0
        else:
            return value < delta

    # Check arguments.
    if dict1 == dict2:
        return True
    if places is not None:
        if delta is not None:
            raise TypeError("specify delta or places not both")
    else:
        delta = delta or 1e-8

    # Compare all keys in both dicts, populating error_msg.
    for key in set(dict1.keys()) | set(dict2.keys()):
        val1 = dict1.get(key, default_value)
        val2 = dict2.get(key, default_value)
        diff = abs(val1 - val2)
        if not valid_comparison(diff):
            return False

    return True


# TODO: similar to `dicts_almost_equal`
def compare_probs(prob, target):
    """Compare probabilities."""
    if not isinstance(prob, list):
        prob = [prob]
    if not isinstance(target, list):
        target = [target]
    assert len(prob) == len(target)
    for p, targ in zip(prob, target):
        for key, t_val in targ.items():
            if key in p:
                assert isclose(p[key], t_val, atol=0.1)
            else:
                assert isclose(t_val, 0, atol=0.1)

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

"""Tests for staged backend Estimator."""

from unittest.mock import Mock, patch

from numpy.random import rand
from numpy.testing import assert_allclose
from pytest import fixture, raises
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import EstimatorResult
from qiskit.providers import JobV1
from qiskit.providers.fake_provider import FakeNairobi, FakeNairobiV2
from qiskit.quantum_info.operators import SparsePauliOp

from staged_primitives.estimator import StagedEstimator

from . import TestFromQiskit, TestOnBackends


################################################################################
## FIXTURES
################################################################################
@fixture
def ansatz():
    """Example ansatz."""
    return RealAmplitudes(num_qubits=2, reps=2)


@fixture
def observable():
    """Example observable."""
    return SparsePauliOp.from_list(
        [
            ("II", -1.052373245772859),
            ("IZ", 0.39793742484318045),
            ("ZI", -0.39793742484318045),
            ("ZZ", -0.01128010425623538),
            ("XX", 0.18093119978423156),
        ]
    )


################################################################################
## TESTS
################################################################################
class TestRun(TestOnBackends, TestFromQiskit):
    """Tests running different payloads on StagedEstimator."""

    def test_run(self, backend):
        """Integration test of run method."""
        backend.set_options(seed_simulator=123)
        estimator = StagedEstimator(backend=backend)
        psi1, psi2 = (
            RealAmplitudes(num_qubits=2, reps=2),
            RealAmplitudes(num_qubits=2, reps=3),
        )
        hamiltonian1, hamiltonian2, hamiltonian3 = (
            SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)]),
            SparsePauliOp.from_list([("IZ", 1)]),
            SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)]),
        )
        theta1, theta2, theta3 = (
            [0, 1, 1, 2, 3, 5],
            [0, 1, 1, 2, 3, 5, 8, 13],
            [1, 2, 3, 4, 5, 6],
        )

        # Specify the circuit and observable by indices.
        # calculate [ <psi1(theta1)|H1|psi1(theta1)> ]
        job = estimator.run([psi1], [hamiltonian1], [theta1])
        assert isinstance(job, JobV1)
        result = job.result()
        assert isinstance(result, EstimatorResult)
        assert_allclose(result.values, [1.5555572817900956], rtol=0.5, atol=0.2)

        # Objects can be passed instead of indices.
        # Note that passing objects has an overhead
        # since the corresponding indices need to be searched.
        # User can append a circuit and observable.
        # calculate [ <psi2(theta2)|H2|psi2(theta2)> ]
        result2 = estimator.run([psi2], [hamiltonian1], [theta2]).result()
        assert_allclose(result2.values, [2.97797666], rtol=0.5, atol=0.2)

        # calculate [ <psi1(theta1)|H2|psi1(theta1)>, <psi1(theta1)|H3|psi1(theta1)> ]
        result3 = estimator.run([psi1, psi1], [hamiltonian2, hamiltonian3], [theta1] * 2).result()
        assert_allclose(result3.values, [-0.551653, 0.07535239], rtol=0.5, atol=0.2)

        # calculate [ <psi1(theta1)|H1|psi1(theta1)>,
        #             <psi2(theta2)|H2|psi2(theta2)>,
        #             <psi1(theta3)|H3|psi1(theta3)> ]
        result4 = estimator.run(
            [psi1, psi2, psi1], [hamiltonian1, hamiltonian2, hamiltonian3], [theta1, theta2, theta3]
        ).result()
        assert_allclose(result4.values, [1.55555728, 0.17849238, -1.08766318], rtol=0.5, atol=0.2)

    def test_run_1qubit(self, backend):
        """Test for 1-qubit cases"""
        backend.set_options(seed_simulator=123)
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(1)
        qc2.x(0)

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("Z", 1)])

        est = StagedEstimator(backend=backend)
        result = est.run([qc], [op], [[]]).result()
        assert isinstance(result, EstimatorResult)
        assert_allclose(result.values, [1], rtol=0.1)

        result = est.run([qc], [op2], [[]]).result()
        assert isinstance(result, EstimatorResult)
        assert_allclose(result.values, [1], rtol=0.1)

        result = est.run([qc2], [op], [[]]).result()
        assert isinstance(result, EstimatorResult)
        assert_allclose(result.values, [1], rtol=0.1)

        result = est.run([qc2], [op2], [[]]).result()
        assert isinstance(result, EstimatorResult)
        assert_allclose(result.values, [-1], rtol=0.1)

    def test_run_2qubits(self, backend):
        """Test for 2-qubit cases (to check endian)"""
        backend.set_options(seed_simulator=123)
        qc = QuantumCircuit(2)
        qc2 = QuantumCircuit(2)
        qc2.x(0)

        op = SparsePauliOp.from_list([("II", 1)])
        op2 = SparsePauliOp.from_list([("ZI", 1)])
        op3 = SparsePauliOp.from_list([("IZ", 1)])

        est = StagedEstimator(backend=backend)
        result = est.run([qc], [op], [[]]).result()
        assert isinstance(result, EstimatorResult)
        assert_allclose(result.values, [1], rtol=0.1)

        result = est.run([qc2], [op], [[]]).result()
        assert isinstance(result, EstimatorResult)
        assert_allclose(result.values, [1], rtol=0.1)

        result = est.run([qc], [op2], [[]]).result()
        assert isinstance(result, EstimatorResult)
        assert_allclose(result.values, [1], rtol=0.1)

        result = est.run([qc2], [op2], [[]]).result()
        assert isinstance(result, EstimatorResult)
        assert_allclose(result.values, [1], rtol=0.1)

        result = est.run([qc], [op3], [[]]).result()
        assert isinstance(result, EstimatorResult)
        assert_allclose(result.values, [1], rtol=0.1)

        result = est.run([qc2], [op3], [[]]).result()
        assert isinstance(result, EstimatorResult)
        assert_allclose(result.values, [-1], rtol=0.1)

    def test_run_errors(self, backend):
        """Test for errors"""
        backend.set_options(seed_simulator=123)
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(2)

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("II", 1)])

        est = StagedEstimator(backend=backend)
        with raises(ValueError):
            est.run([qc], [op2], [[]]).result()
        with raises(ValueError):
            est.run([qc2], [op], [[]]).result()
        with raises(ValueError):
            est.run([qc], [op], [[1e4]]).result()
        with raises(ValueError):
            est.run([qc2], [op2], [[1, 2]]).result()
        with raises(ValueError):
            est.run([qc, qc2], [op2], [[1]]).result()
        with raises(ValueError):
            est.run([qc], [op, op2], [[1]]).result()

    def test_run_numpy_params(self, backend):
        """Test for numpy array as parameter values"""
        backend.set_options(seed_simulator=123)
        qc = RealAmplitudes(num_qubits=2, reps=2)
        op = SparsePauliOp.from_list([("IZ", 1), ("XI", 2), ("ZY", -1)])
        k = 5
        params_array = rand(k, qc.num_parameters)
        params_list = params_array.tolist()
        params_list_array = list(params_array)
        estimator = StagedEstimator(backend=backend)
        target = estimator.run([qc] * k, [op] * k, params_list).result()

        # with self.subTest("ndarrary"):
        result = estimator.run([qc] * k, [op] * k, params_array).result()
        assert len(result.metadata) == k
        assert_allclose(result.values, target.values, rtol=0.2, atol=0.2)

        # with self.subTest("list of ndarray"):
        result = estimator.run([qc] * k, [op] * k, params_list_array).result()
        assert len(result.metadata) == k
        assert_allclose(result.values, target.values, rtol=0.2, atol=0.2)


class TestOptions(TestOnBackends, TestFromQiskit):
    """Test different configurations."""

    def test_estimator_run_no_params(self, backend, ansatz, observable):
        """test for estimator without parameters"""
        backend.set_options(seed_simulator=123)
        circuit = ansatz.bind_parameters([0, 1, 1, 2, 3, 5])
        est = StagedEstimator(backend=backend)
        result = est.run([circuit], [observable]).result()
        assert isinstance(result, EstimatorResult)
        assert_allclose(result.values, [-1.284366511861733], rtol=0.05)

    def test_run_with_shots_option(self, backend, ansatz, observable):
        """test with shots option."""
        est = StagedEstimator(backend=backend)
        result = est.run(
            [ansatz],
            [observable],
            parameter_values=[[0, 1, 1, 2, 3, 5]],
            shots=1024,
            seed_simulator=15,
        ).result()
        assert isinstance(result, EstimatorResult)
        assert_allclose(result.values, [-1.307397243478641], rtol=0.1)

    def test_options(self, backend, ansatz, observable):
        """Test for options"""
        # with self.subTest("init"):
        estimator = StagedEstimator(backend=backend, options={"shots": 3000})
        assert estimator.options.get("shots") == 3000
        # with self.subTest("set_options"):
        estimator.set_options(shots=1024, seed_simulator=15)
        assert estimator.options.get("shots") == 1024
        assert estimator.options.get("seed_simulator") == 15
        # with self.subTest("run"):
        result = estimator.run(
            [ansatz],
            [observable],
            parameter_values=[[0, 1, 1, 2, 3, 5]],
        ).result()
        assert isinstance(result, EstimatorResult)
        assert_allclose(result.values, [-1.307397243478641], rtol=0.1)


class TestJobExecution(TestFromQiskit):
    """Test job execution under different backend configurations."""

    def test_job_size_limit_v2(self):
        """Test StagedEstimator respects job size limit"""

        class FakeNairobiLimitedCircuits(FakeNairobiV2):
            """FakeNairobiV2 with job size limit."""

            @property
            def max_circuits(self):
                return 1

        backend = FakeNairobiLimitedCircuits()
        backend.set_options(seed_simulator=123)
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        backend.set_options(seed_simulator=123)
        qc = RealAmplitudes(num_qubits=2, reps=2)
        op = SparsePauliOp.from_list([("IZ", 1), ("XI", 2), ("ZY", -1)])
        reps = 5
        params_array = rand(reps, qc.num_parameters)
        params_list = params_array.tolist()
        estimator = StagedEstimator(backend=backend)
        estimator._build_single_result = Mock(return_value=EstimatorResult([1], [{}]))
        obs = len(estimator._operator_decomposer.decompose(op))
        with patch.object(backend, "run") as run_mock:
            estimator.run([qc] * reps, [op] * reps, params_list).result()
        assert run_mock.call_count == reps * obs

    def test_job_size_limit_v1(self):
        """Test StagedEstimator respects job size limit"""
        backend = FakeNairobi()
        config = backend.configuration()
        config.max_experiments = 1
        backend._configuration = config
        backend.set_options(seed_simulator=123)
        qc = RealAmplitudes(num_qubits=2, reps=2)
        op = SparsePauliOp.from_list([("IZ", 1), ("XI", 2), ("ZY", -1)])
        reps = 5
        params_array = rand(reps, qc.num_parameters)
        params_list = params_array.tolist()
        estimator = StagedEstimator(backend=backend)
        estimator._build_single_result = Mock(return_value=EstimatorResult([1], [{}]))
        obs = len(estimator._operator_decomposer.decompose(op))
        with patch.object(backend, "run") as run_mock:
            estimator.run([qc] * reps, [op] * reps, params_list).result()
        assert run_mock.call_count == reps * obs

    def test_no_max_circuits(self):
        """Test StagedEstimator works with BackendV1 and no max_experiments set."""
        backend = FakeNairobi()
        config = backend.configuration()
        del config.max_experiments
        backend._configuration = config
        backend.set_options(seed_simulator=123)
        qc = RealAmplitudes(num_qubits=2, reps=2)
        op = SparsePauliOp.from_list([("IZ", 1), ("XI", 2), ("ZY", -1)])
        k = 5
        params_array = rand(k, qc.num_parameters)
        params_list = params_array.tolist()
        params_list_array = list(params_array)
        estimator = StagedEstimator(backend=backend)
        estimator._build_single_result = Mock(return_value=EstimatorResult([1], [{}]))
        target = estimator.run([qc] * k, [op] * k, params_list).result()
        ## NDARRAY
        result = estimator.run([qc] * k, [op] * k, params_array).result()
        assert len(result.metadata) == k
        assert_allclose(result.values, target.values, rtol=0.2, atol=0.2)
        ## LIST OF NDARRAY
        result = estimator.run([qc] * k, [op] * k, params_list_array).result()
        assert len(result.metadata) == k
        assert_allclose(result.values, target.values, rtol=0.2, atol=0.2)

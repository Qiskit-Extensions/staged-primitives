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


from __future__ import annotations

from itertools import product
from unittest.mock import Mock, patch

from numpy import isclose, pi, sqrt
from numpy.random import default_rng
from pytest import fixture, mark, raises
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile
from qiskit.providers import BackendV1, BackendV2
from qiskit.providers.fake_provider import FakeManhattan, FakeManhattanV2
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.result import Counts
from qiskit.transpiler import PassManager

from staged_primitives.estimator import StagedEstimator
from staged_primitives.utils.operators import AbelianDecomposer, NaiveDecomposer
from staged_primitives.utils.results import CanonicalReckoner, ReckoningResult


################################################################################
## FIXTURES
################################################################################
@fixture(scope="function")
def backend_mock():
    """BackendV2 mock."""
    return Mock(BackendV2)


@fixture
def estimator(backend_mock):
    """StagedEstimator object with mock backend and default settings."""
    return StagedEstimator(backend_mock)


################################################################################
## CASES
################################################################################
def observable_measurement():
    """Generator of observables and corresponding measurement circuits.

    Yields:
        - Observable
        - List of quantum circuits to measure said observable
        - Whether commuting terms were grouped or not
    """

    Z = QuantumCircuit(1, 1)  # pylint: disable=invalid-name
    Z.measure(0, 0)
    yield SparsePauliOp("I"), [Z], False
    yield SparsePauliOp("I"), [Z], True
    yield SparsePauliOp("Z"), [Z], False
    yield SparsePauliOp("Z"), [Z], True
    yield SparsePauliOp(["I", "Z"]), [Z, Z], False
    yield SparsePauliOp(["Z", "I"]), [Z, Z], False
    yield SparsePauliOp(["I", "Z"]), [Z], True
    yield SparsePauliOp(["Z", "I"]), [Z], True

    X = QuantumCircuit(1, 1)  # pylint: disable=invalid-name
    X.h(0)
    X.measure(0, 0)
    yield SparsePauliOp("X"), [X], False
    yield SparsePauliOp("X"), [X], True
    yield SparsePauliOp(["X", "I"]), [X], True
    yield SparsePauliOp(["I", "X"]), [X], True
    yield SparsePauliOp(["X", "I"]), [X, Z], False
    yield SparsePauliOp(["I", "X"]), [Z, X], False

    Y = QuantumCircuit(1, 1)  # pylint: disable=invalid-name
    Y.sdg(0)
    Y.h(0)
    Y.measure(0, 0)
    yield SparsePauliOp("Y"), [Y], False
    yield SparsePauliOp("Y"), [Y], True
    yield SparsePauliOp(["Y", "I"]), [Y], True
    yield SparsePauliOp(["I", "Y"]), [Y], True
    yield SparsePauliOp(["Y", "I"]), [Y, Z], False
    yield SparsePauliOp(["I", "Y"]), [Z, Y], False

    yield SparsePauliOp(["I", "Y", "X"]), [Y, X], True
    yield SparsePauliOp(["I", "Y", "X"]), [Z, Y, X], False
    yield SparsePauliOp(["I", "Y", "X", "Z"]), [Y, X, Z], True
    yield SparsePauliOp(["I", "Y", "X", "Z"]), [Z, Y, X, Z], False

    IZ = QuantumCircuit(2, 1)  # pylint: disable=invalid-name
    IZ.measure(0, 0)
    yield SparsePauliOp("II"), [IZ], False
    yield SparsePauliOp("II"), [IZ], True
    yield SparsePauliOp("IZ"), [IZ], False
    yield SparsePauliOp("IZ"), [IZ], True
    yield SparsePauliOp(["II", "IZ"]), [IZ], True
    yield SparsePauliOp(["II", "IZ"]), [IZ, IZ], False

    XY = QuantumCircuit(2, 2)  # pylint: disable=invalid-name
    XY.h(1)
    XY.sdg(0)
    XY.h(0)
    XY.measure([0, 1], [0, 1])
    yield SparsePauliOp("XY"), [XY], False
    yield SparsePauliOp("XY"), [XY], True
    yield SparsePauliOp(["XY", "II"]), [XY], True
    yield SparsePauliOp(["XY", "II"]), [XY, IZ], False

    XYZ = QuantumCircuit(3, 3)  # pylint: disable=invalid-name
    XYZ.h(2)
    XYZ.sdg(1)
    XYZ.h(1)
    XYZ.measure([0, 1, 2], [0, 1, 2])
    yield SparsePauliOp(["XYZ", "XII", "IYI", "IIZ", "XIZ", "III"]), [XYZ], True

    YIX = QuantumCircuit(3, 2)  # pylint: disable=invalid-name
    YIX.sdg(2)
    YIX.h(2)
    YIX.h(0)
    YIX.measure([0, 2], [0, 1])
    yield SparsePauliOp(["YIX", "IIX", "YII", "III"]), [YIX], True


################################################################################
## TESTS
################################################################################
class TestProperties:
    """Test StagedEstimator properties."""

    def test_backend(self, estimator):
        """Test backend."""
        backend = FakeManhattanV2()
        assert isinstance(backend, BackendV2)
        estimator.backend = backend
        assert estimator.backend is backend
        backend = FakeManhattan()
        assert isinstance(backend, BackendV1)
        estimator.backend = backend
        assert isinstance(estimator.backend, BackendV2)
        # TODO: assert estimator.backend == backend
        assert estimator.backend is not backend

    @mark.parametrize("backend", ["backend", None, Ellipsis, False])
    def test_backend_validation(self, estimator, backend):
        """Test backend validation."""
        with raises(TypeError):
            estimator.backend = backend

    def test_group_commuting(self, estimator):
        """Test abelian_groupipng."""
        assert estimator.group_commuting is True  # Default
        estimator.group_commuting = False
        assert estimator.group_commuting is False
        estimator.group_commuting = True
        assert estimator.group_commuting is True

    @mark.parametrize("group_commuting", [0, 1, None, Ellipsis])
    def test_group_commuting_validation(self, estimator, group_commuting):
        """Test abelian_groupipng validation."""
        estimator.group_commuting = group_commuting
        assert estimator.group_commuting is bool(group_commuting)

    def test_skip_transpilation(self, estimator):
        """Test skip_transpilation."""
        assert estimator.skip_transpilation is False  # Default
        estimator.skip_transpilation = True
        assert estimator.skip_transpilation is True
        estimator.skip_transpilation = False
        assert estimator.skip_transpilation is False

    @mark.parametrize("skip_transpilation", [0, 1, None, Ellipsis])
    def test_skip_transpilation_validation(self, estimator, skip_transpilation):
        """Test skip_transpilation validation."""
        estimator.skip_transpilation = skip_transpilation
        assert estimator.skip_transpilation is bool(skip_transpilation)

    def test_bound_pass_manager(self, estimator):
        """Test bound_pass_manager."""
        assert isinstance(estimator.bound_pass_manager, PassManager)
        assert estimator.bound_pass_manager.passes() == []  # Default
        pass_manager = PassManager()
        estimator.bound_pass_manager = pass_manager
        assert estimator.bound_pass_manager is pass_manager
        estimator.bound_pass_manager = None
        assert isinstance(estimator.bound_pass_manager, PassManager)
        assert estimator.bound_pass_manager.passes() == []

    @mark.parametrize("pass_manager", ["pass_manager", False, Ellipsis])
    def test_bound_pass_manager_validation(self, estimator, pass_manager):
        """Test bound_pass_manager validation."""
        with raises(TypeError):
            estimator.bound_pass_manager = pass_manager

    @mark.parametrize("options", [{}, {"foo": "bar"}, {"int": 4}])
    def test_transpile_options(self, estimator, options):
        """Test transpile_options setter."""
        assert estimator.transpile_options.__dict__ == {}
        estimator.set_transpile_options(**options)
        assert estimator.transpile_options.__dict__ == options

    def test_get_operator_decomposer(self, estimator):
        """Test operator_decomposer getter."""
        assert estimator._operator_decomposer is not estimator._operator_decomposer
        assert isinstance(estimator._operator_decomposer, AbelianDecomposer)  # Default
        estimator.group_commuting = False
        assert isinstance(estimator._operator_decomposer, NaiveDecomposer)
        estimator.group_commuting = True
        assert isinstance(estimator._operator_decomposer, AbelianDecomposer)

    def test_get_expval_reckoner(self, estimator):
        """Test expval_reckoner getter."""
        assert estimator._expval_reckoner is not estimator._expval_reckoner
        assert isinstance(estimator._expval_reckoner, CanonicalReckoner)  # Default


class TestImplementation:
    """Test StagedEstimator implementation."""

    @mark.parametrize(
        "num_qubits, skip_transpilation, seed",
        ((num, skip, i) for i, (num, skip) in enumerate(product(range(1, 5), [False, True]))),
    )
    def test_transpile_single_unbound(self, estimator, num_qubits, skip_transpilation, seed):
        """Test transpile_single_unbound."""
        # Case
        estimator.skip_transpilation = skip_transpilation
        layout_intlist = default_rng(seed).permutation(num_qubits + 4).tolist()[:num_qubits]
        circuit = random_circuit(num_qubits, depth=2, seed=seed)
        if num_qubits % 2:  # Note: only sometimes to test both w/ and w/o metadata
            circuit.metadata = {}
        # Test
        with patch("staged_primitives.estimator.transpile") as mock:
            mock.side_effect = lambda c, *_, **__: transpile(c, initial_layout=layout_intlist)
            transpiled_circuit = estimator._transpile_single_unbound(circuit)
        if estimator.skip_transpilation:
            assert transpiled_circuit == circuit
            assert transpiled_circuit.metadata.get("end_layout_intlist") == tuple(range(num_qubits))
        else:
            assert transpiled_circuit == transpile(circuit, initial_layout=layout_intlist)
            assert transpiled_circuit.metadata.get("end_layout_intlist") == tuple(layout_intlist)

    @mark.parametrize("seed", range(10))
    def test_bind_single_parameters(self, estimator, seed):
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
        bound_circuit = estimator._bind_single_parameters(circuit, parameter_values)
        assert bound_circuit is circuit  # Inplace binding, otherwise assert equal
        assert bound_circuit.num_parameters == 0
        for instruction, expected in zip(bound_circuit, parameter_values):
            gate = instruction[0]
            value = gate.params[0]
            assert value == expected

    def test_transpile_single_bound(self, estimator):
        """Test transpile_single_bound."""
        circuit = QuantumCircuit(4)
        bound_pass_manager = Mock(PassManager)
        estimator.bound_pass_manager = bound_pass_manager
        transpiled = estimator._transpile_single_bound(circuit)
        bound_pass_manager.run.assert_called_once_with(circuit)
        assert transpiled is bound_pass_manager.run.return_value

    @mark.parametrize("observable, measurements, group_commuting", [*observable_measurement()])
    def test_observe_single_circuit(self, estimator, observable, measurements, group_commuting):
        """Test observe_single_circuit."""
        # Case
        estimator.group_commuting = group_commuting
        num_qubits = observable.num_qubits
        base_num_qubits = max(5, num_qubits)  # Note: test num_qubits mismatch
        base = random_circuit(base_num_qubits, depth=2, seed=0)
        layout_intlist = default_rng(0).permutation(base_num_qubits).tolist()[:num_qubits]
        layout_intlist = tuple(layout_intlist)
        base.metadata = {"end_layout_intlist": layout_intlist}
        # Test
        with patch("staged_primitives.estimator.transpile", spec=True) as mock:
            mock.side_effect = transpile_layout_only  # Note: auxiliary function
            circuits = estimator._observe_single_circuit(base.copy(), observable)
        assert len(circuits) == len(measurements)
        for circuit, measurement in zip(circuits, measurements):
            measurement = transpile(measurement, initial_layout=layout_intlist)
            assert circuit == base.compose(measurement)
            assert isinstance(circuit.metadata.get("end_layout_intlist"), tuple)
            assert circuit.metadata.get("end_layout_intlist") == layout_intlist
            assert isinstance(circuit.metadata.get("observable"), SparsePauliOp)
            assert circuit.metadata.get("observable")  # TODO
            assert isinstance(circuit.metadata.get("measured_qubit_indices"), tuple)
            assert circuit.metadata.get("measured_qubit_indices")  # TODO

    @mark.parametrize("num_circuits", range(4))
    def test_execute(self, estimator, num_circuits):
        """Test execute."""
        # Case
        estimator.backend.max_circuits = 1  # Note: test backend limit is bypassed
        estimator.backend.run.side_effect = lambda *args, **kwargs: Mock()
        circuits = [QuantumCircuit(1) for _ in range(num_circuits)]
        for i, circuit in enumerate(circuits):
            circuit.metadata = {"index": i}  # Note: test metadata gets transferred
        # Test
        counts_list = estimator._execute(circuits, shots=12)
        for counts, circuit in zip(counts_list, circuits):
            assert counts.metadata == circuit.metadata

    @mark.parametrize(
        "counts_list, observables, reckoned",
        [
            ([], [], ReckoningResult(0, 0)),
            ([Counts({})], [SparsePauliOp("Z")], ReckoningResult(0, 1)),
            ([Counts({0: 0})], [SparsePauliOp("I")], ReckoningResult(0, 1)),
            ([Counts({0: 1})], [SparsePauliOp("I")], ReckoningResult(1, 0)),
            ([Counts({1: 1})], [SparsePauliOp("I")], ReckoningResult(1, 0)),
            ([Counts({0: 1})], [SparsePauliOp("Z")], ReckoningResult(1, 0)),
            ([Counts({1: 1})], [SparsePauliOp("Z")], ReckoningResult(-1, 0)),
            ([Counts({0: 1})], [SparsePauliOp("X")], ReckoningResult(1, 0)),
            ([Counts({1: 1})], [SparsePauliOp("X")], ReckoningResult(-1, 0)),
            ([Counts({0: 1})], [SparsePauliOp("Y")], ReckoningResult(1, 0)),
            ([Counts({1: 1})], [SparsePauliOp("Y")], ReckoningResult(-1, 0)),
            ([Counts({0: 1, 1: 1})], [SparsePauliOp("I")], ReckoningResult(1, 0)),
            ([Counts({0: 1, 1: 1})], [SparsePauliOp("Z")], ReckoningResult(0, 1 / sqrt(2))),
            ([Counts({0: 1, 1: 1})], [SparsePauliOp("X")], ReckoningResult(0, 1 / sqrt(2))),
            ([Counts({0: 1, 1: 1})], [SparsePauliOp("Y")], ReckoningResult(0, 1 / sqrt(2))),
            (
                [Counts({0: 1, 1: 1}), Counts({0: 1, 1: 1})],
                [SparsePauliOp("I"), SparsePauliOp("Z")],
                ReckoningResult(1, 1 / sqrt(2)),
            ),
            (
                [Counts({0: 1, 1: 1}), Counts({0: 1, 1: 1})],
                [SparsePauliOp("X"), SparsePauliOp("Z")],
                ReckoningResult(0, 1),
            ),
            (
                [Counts({0: 1, 1: 1}), Counts({0: 1, 1: 1})],
                [SparsePauliOp(["X", "Z"], [1, 2]), SparsePauliOp(["I"])],
                ReckoningResult(1, sqrt(5 / 2)),
            ),
        ],
    )
    def test_build_single_result(self, estimator, counts_list, observables, reckoned):
        """Test build_single_result."""
        # Case
        expval, std_error = reckoned
        num_circuits = len(counts_list)
        shots = sum(counts.shots() for counts in counts_list)
        shots_per_circuit = shots / (num_circuits or 1)
        variance = shots_per_circuit * std_error**2
        metadata = {
            "variance": variance,
            "std_dev": sqrt(variance),
            "std_error": std_error,
            "shots": shots,
            "shots_per_circuit": shots_per_circuit,
            "num_circuits": num_circuits,
        }
        # Test
        for counts, observable in zip(counts_list, observables):
            counts.metadata = {"observable": observable}
        result = estimator._build_single_result(counts_list)
        assert len(result.values) == len(result.metadata) == 1
        assert result.values[0] == expval
        assert isclose(result.metadata[0]["variance"], metadata["variance"])
        assert isclose(result.metadata[0]["std_dev"], metadata["std_dev"])
        assert isclose(result.metadata[0]["std_error"], metadata["std_error"])
        assert isclose(result.metadata[0]["shots_per_circuit"], metadata["shots_per_circuit"])
        assert result.metadata[0]["shots"] == metadata["shots"]
        assert result.metadata[0]["num_circuits"] == metadata["num_circuits"]


################################################################################
## AUXILIARY
################################################################################
def transpile_layout_only(circuit, *_, **options):
    """Bypass transpilation except for `initial_layout`."""
    initial_layout = options.get("initial_layout", None)
    return transpile(circuit, initial_layout=initial_layout)

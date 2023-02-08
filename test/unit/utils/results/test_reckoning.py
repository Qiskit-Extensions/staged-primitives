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

from test import NO_ITERS

from numpy import isclose, sqrt
from numpy.random import default_rng
from pytest import mark, raises
from qiskit.quantum_info.operators import Pauli, SparsePauliOp
from qiskit.result import Counts

from staged_primitives.utils.results.reckoning import (
    CanonicalReckoner,
    ExpvalReckoner,
    ReckoningResult,
)


################################################################################
## TESTS
################################################################################
class TestExpvalReckoner:
    """Test ExpvalReckoner interface."""

    @mark.parametrize(
        "counts, expected",
        [
            (Counts({}), (Counts({}),)),
            (Counts({0: 1}), (Counts({0: 1}),)),
            (Counts({0: 0, 1: 1}), (Counts({0: 0, 1: 1}),)),
            ([Counts({})], (Counts({}),)),
            ([Counts({0: 1})], (Counts({0: 1}),)),
            ([Counts({0: 0, 1: 1})], (Counts({0: 0, 1: 1}),)),
            ([Counts({}), Counts({0: 0, 1: 1})], (Counts({}), Counts({0: 0, 1: 1}))),
        ],
    )
    def test_validate_counts_list(self, counts, expected):
        """Test validate counts."""
        valid = ExpvalReckoner._validate_counts_list(counts)
        assert isinstance(valid, tuple)
        assert all(isinstance(c, Counts) for c in valid)
        assert valid == expected

    @mark.parametrize("counts", NO_ITERS)
    def test_validate_counts_list_type_error(self, counts):
        """Test validate counts raises errors."""
        with raises(TypeError):
            ExpvalReckoner._validate_counts_list(counts)
        with raises(TypeError):
            ExpvalReckoner._validate_counts_list([counts])

    @mark.parametrize(
        "operators, expected",
        [
            ("I", (SparsePauliOp("I"),)),
            ("Z", (SparsePauliOp("Z"),)),
            ("X", (SparsePauliOp("X"),)),
            ("Y", (SparsePauliOp("Y"),)),
            ("IXYZ", (SparsePauliOp("IXYZ"),)),
            (Pauli("I"), (SparsePauliOp("I"),)),
            (Pauli("Z"), (SparsePauliOp("Z"),)),
            (Pauli("X"), (SparsePauliOp("X"),)),
            (Pauli("Y"), (SparsePauliOp("Y"),)),
            (Pauli("IXYZ"), (SparsePauliOp("IXYZ"),)),
            ([Pauli("I")], (SparsePauliOp("I"),)),
            ([Pauli("Z")], (SparsePauliOp("Z"),)),
            ([Pauli("X")], (SparsePauliOp("X"),)),
            ([Pauli("Y")], (SparsePauliOp("Y"),)),
            ([Pauli("IXYZ")], (SparsePauliOp("IXYZ"),)),
            (["ZYXI", Pauli("IXYZ")], (SparsePauliOp("ZYXI"), SparsePauliOp("IXYZ"))),
        ],
    )
    def test_validate_operator_list(self, operators, expected):
        """Test validate operators."""
        valid = ExpvalReckoner._validate_operator_list(operators)
        assert isinstance(valid, tuple)
        assert all(isinstance(c, SparsePauliOp) for c in valid)
        assert valid == expected

    @mark.parametrize("operators", NO_ITERS)
    def test_validate_operator_list_type_error(self, operators):
        """Test validate operators raises errors."""
        with raises(TypeError):
            ExpvalReckoner._validate_operator_list(operators)
        with raises(TypeError):
            ExpvalReckoner._validate_operator_list([operators])

    @mark.parametrize("pauli", ["I", "X", "Y", "Z", "IXYZ"])
    def test_validate_pauli(self, pauli):
        """Test validate Pauli."""
        pauli = Pauli(pauli)
        assert ExpvalReckoner._validate_pauli(pauli) == pauli

    @mark.parametrize("pauli", NO_ITERS)
    def test_validate_pauli_type_error(self, pauli):
        """Test validate Pauli raises errors."""
        with raises(TypeError):
            ExpvalReckoner._validate_pauli(pauli)

    @mark.parametrize("seed", range(8))
    def test_cross_validate_lists(self, seed):
        """Test cross validate counts and operators."""
        rng = default_rng(seed)
        size = rng.integers(256)
        ExpvalReckoner._cross_validate_lists(["c"] * size, ["o"] * size)
        with raises(ValueError):
            ExpvalReckoner._cross_validate_lists(["c"] * size, ["o"] * rng.integers(256))


@mark.parametrize("reckoner", [CanonicalReckoner()])
class TestCanonicalReckoner:
    """Test CanonicalReckoner."""

    @mark.parametrize(
        "counts, operators, expected",
        [
            ([], [], ReckoningResult(0, 0)),
            ([Counts({})], ["Z"], ReckoningResult(0, 1)),
            ([Counts({0: 0})], ["I"], ReckoningResult(0, 1)),
            ([Counts({0: 1})], ["I"], ReckoningResult(1, 0)),
            ([Counts({1: 1})], ["I"], ReckoningResult(1, 0)),
            ([Counts({0: 1})], ["Z"], ReckoningResult(1, 0)),
            ([Counts({1: 1})], ["Z"], ReckoningResult(-1, 0)),
            ([Counts({0: 1})], ["X"], ReckoningResult(1, 0)),
            ([Counts({1: 1})], ["X"], ReckoningResult(-1, 0)),
            ([Counts({0: 1})], ["Y"], ReckoningResult(1, 0)),
            ([Counts({1: 1})], ["Y"], ReckoningResult(-1, 0)),
            ([Counts({0: 1, 1: 1})], ["I"], ReckoningResult(1, 0)),
            ([Counts({0: 1, 1: 1})], ["Z"], ReckoningResult(0, 1 / sqrt(2))),
            ([Counts({0: 1, 1: 1})], ["X"], ReckoningResult(0, 1 / sqrt(2))),
            ([Counts({0: 1, 1: 1})], ["Y"], ReckoningResult(0, 1 / sqrt(2))),
            (
                [Counts({0: 1, 1: 1}), Counts({0: 1, 1: 1})],
                ["I", "Z"],
                ReckoningResult(1, 1 / sqrt(2)),
            ),
            (
                [Counts({0: 1, 1: 1}), Counts({0: 1, 1: 1})],
                ["X", "Z"],
                ReckoningResult(0, 1),
            ),
            (
                [Counts({0: 1, 1: 1}), Counts({0: 1, 1: 1})],
                [SparsePauliOp(["X", "Z"], [1, 2]), SparsePauliOp(["I"])],
                ReckoningResult(1, sqrt(5 / 2)),
            ),
        ],
    )
    @mark.parametrize("global_coeff", [1, 1j, 1 + 1j, 0.5 + 2j])
    def test_reckon(self, reckoner, counts, operators, expected, global_coeff):
        """Test reckon."""
        operators = (SparsePauliOp(o) if not isinstance(o, SparsePauliOp) else o for o in operators)
        operators = [global_coeff * o for o in operators]
        result = reckoner.reckon(counts, operators)
        assert isinstance(result, ReckoningResult)
        for r, e in zip(result, expected):
            assert isclose(r, global_coeff * e)

    @mark.parametrize(
        "counts, operator, expected",
        [
            ({}, SparsePauliOp("I"), ReckoningResult(0, 1)),
            ({}, SparsePauliOp(["I", "I"]), ReckoningResult(0, sqrt(2))),
            ({}, SparsePauliOp(["I", "Z"]), ReckoningResult(0, sqrt(2))),
            ({}, SparsePauliOp(["I", "X"]), ReckoningResult(0, sqrt(2))),
            ({}, SparsePauliOp(["I", "Y"]), ReckoningResult(0, sqrt(2))),
            ({}, SparsePauliOp(["I", "I"], [1, 2]), ReckoningResult(0, sqrt(5))),
            ({}, SparsePauliOp(["I", "Z"], [1, 2]), ReckoningResult(0, sqrt(5))),
            ({}, SparsePauliOp(["I", "X"], [1, 2]), ReckoningResult(0, sqrt(5))),
            ({}, SparsePauliOp(["I", "Y"], [1, 2]), ReckoningResult(0, sqrt(5))),
            ({0: 1, 1: 0}, SparsePauliOp(["I", "I"], [1, 2]), ReckoningResult(3, 0)),
            ({0: 1, 1: 0}, SparsePauliOp(["I", "Z"], [1, 2]), ReckoningResult(3, 0)),
            ({0: 1, 1: 0}, SparsePauliOp(["I", "X"], [1, 2]), ReckoningResult(3, 0)),
            ({0: 1, 1: 0}, SparsePauliOp(["I", "Y"], [1, 2]), ReckoningResult(3, 0)),
            ({0: 0, 1: 1}, SparsePauliOp(["I", "I"], [1, 2]), ReckoningResult(3, 0)),
            ({0: 0, 1: 1}, SparsePauliOp(["I", "Z"], [1, 2]), ReckoningResult(-1, 0)),
            ({0: 0, 1: 1}, SparsePauliOp(["I", "X"], [1, 2]), ReckoningResult(-1, 0)),
            ({0: 0, 1: 1}, SparsePauliOp(["I", "Y"], [1, 2]), ReckoningResult(-1, 0)),
            ({0: 1, 1: 1}, SparsePauliOp(["I", "I"], [1, 2]), ReckoningResult(3, 0)),
            ({0: 1, 1: 1}, SparsePauliOp(["I", "Z"], [1, 2]), ReckoningResult(1, sqrt(2))),
            ({0: 1, 1: 1}, SparsePauliOp(["I", "X"], [1, 2]), ReckoningResult(1, sqrt(2))),
            ({0: 1, 1: 1}, SparsePauliOp(["I", "Y"], [1, 2]), ReckoningResult(1, sqrt(2))),
            ({0: 1, 1: 1}, SparsePauliOp(["Z", "I"], [1, 2]), ReckoningResult(2, 1 / sqrt(2))),
            ({0: 1, 1: 1}, SparsePauliOp(["X", "I"], [1, 2]), ReckoningResult(2, 1 / sqrt(2))),
            ({0: 1, 1: 1}, SparsePauliOp(["Y", "I"], [1, 2]), ReckoningResult(2, 1 / sqrt(2))),
            ({0: 1, 1: 1}, SparsePauliOp(["Z", "Z"], [1, 2]), ReckoningResult(0, sqrt(5 / 2))),
            ({0: 1, 1: 1}, SparsePauliOp(["Z", "X"], [1, 2]), ReckoningResult(0, sqrt(5 / 2))),
            ({0: 1, 1: 1}, SparsePauliOp(["Z", "Y"], [1, 2]), ReckoningResult(0, sqrt(5 / 2))),
            ({0: 1, 1: 1}, SparsePauliOp(["X", "Z"], [1, 2]), ReckoningResult(0, sqrt(5 / 2))),
            ({0: 1, 1: 1}, SparsePauliOp(["X", "X"], [1, 2]), ReckoningResult(0, sqrt(5 / 2))),
            ({0: 1, 1: 1}, SparsePauliOp(["X", "Y"], [1, 2]), ReckoningResult(0, sqrt(5 / 2))),
            ({0: 1, 1: 1}, SparsePauliOp(["Y", "Z"], [1, 2]), ReckoningResult(0, sqrt(5 / 2))),
            ({0: 1, 1: 1}, SparsePauliOp(["Y", "X"], [1, 2]), ReckoningResult(0, sqrt(5 / 2))),
            ({0: 1, 1: 1}, SparsePauliOp(["Y", "Y"], [1, 2]), ReckoningResult(0, sqrt(5 / 2))),
        ],
    )
    @mark.parametrize("global_coeff", [1, 1j, 1 + 1j, 0.5 + 2j])
    def test_reckon_operator(self, reckoner, counts, operator, expected, global_coeff):
        """Test reckon operator."""
        counts = Counts(counts)
        operator = global_coeff * operator
        result = reckoner.reckon_operator(counts, operator)
        assert isinstance(result, ReckoningResult)
        for r, e in zip(result, expected):
            assert isclose(r, global_coeff * e)

    @mark.parametrize(
        "counts, pauli, expected",
        [
            ({}, Pauli("I"), ReckoningResult(0, 1)),
            ({}, Pauli("Z"), ReckoningResult(0, 1)),
            ({}, Pauli("X"), ReckoningResult(0, 1)),
            ({}, Pauli("Y"), ReckoningResult(0, 1)),
            ({0: 1}, Pauli("I"), ReckoningResult(1, 0)),
            ({0: 1}, Pauli("Z"), ReckoningResult(1, 0)),
            ({0: 1}, Pauli("X"), ReckoningResult(1, 0)),
            ({0: 1}, Pauli("Y"), ReckoningResult(1, 0)),
            ({1: 1}, Pauli("I"), ReckoningResult(1, 0)),
            ({1: 1}, Pauli("Z"), ReckoningResult(-1, 0)),
            ({1: 1}, Pauli("X"), ReckoningResult(-1, 0)),
            ({1: 1}, Pauli("Y"), ReckoningResult(-1, 0)),
            ({0: 1, 1: 1}, Pauli("I"), ReckoningResult(1, 0)),
            ({0: 1, 1: 1}, Pauli("Z"), ReckoningResult(0, 1 / sqrt(2))),
            ({0: 1, 1: 1}, Pauli("X"), ReckoningResult(0, 1 / sqrt(2))),
            ({0: 1, 1: 1}, Pauli("Y"), ReckoningResult(0, 1 / sqrt(2))),
            ({0: 0, 1: 1, 2: 2, 3: 3}, Pauli("II"), ReckoningResult(1, 0)),
            ({0: 0, 1: 1, 2: 2, 3: 3}, Pauli("IZ"), ReckoningResult(-1 / 3, sqrt(4 / 27))),
            ({0: 0, 1: 1, 2: 2, 3: 3}, Pauli("IX"), ReckoningResult(-1 / 3, sqrt(4 / 27))),
            ({0: 0, 1: 1, 2: 2, 3: 3}, Pauli("IY"), ReckoningResult(-1 / 3, sqrt(4 / 27))),
            ({0: 0, 1: 1, 2: 2, 3: 3}, Pauli("ZI"), ReckoningResult(-2 / 3, sqrt(5 / 54))),
            ({0: 0, 1: 1, 2: 2, 3: 3}, Pauli("XI"), ReckoningResult(-2 / 3, sqrt(5 / 54))),
            ({0: 0, 1: 1, 2: 2, 3: 3}, Pauli("YI"), ReckoningResult(-2 / 3, sqrt(5 / 54))),
            ({0: 0, 1: 1, 2: 2, 3: 3}, Pauli("ZZ"), ReckoningResult(0, sqrt(1 / 6))),
            ({0: 0, 1: 1, 2: 2, 3: 3}, Pauli("ZX"), ReckoningResult(0, sqrt(1 / 6))),
            ({0: 0, 1: 1, 2: 2, 3: 3}, Pauli("ZY"), ReckoningResult(0, sqrt(1 / 6))),
            ({0: 0, 1: 1, 2: 2, 3: 3}, Pauli("XZ"), ReckoningResult(0, sqrt(1 / 6))),
            ({0: 0, 1: 1, 2: 2, 3: 3}, Pauli("XX"), ReckoningResult(0, sqrt(1 / 6))),
            ({0: 0, 1: 1, 2: 2, 3: 3}, Pauli("XY"), ReckoningResult(0, sqrt(1 / 6))),
            ({0: 0, 1: 1, 2: 2, 3: 3}, Pauli("YZ"), ReckoningResult(0, sqrt(1 / 6))),
            ({0: 0, 1: 1, 2: 2, 3: 3}, Pauli("YX"), ReckoningResult(0, sqrt(1 / 6))),
            ({0: 0, 1: 1, 2: 2, 3: 3}, Pauli("YY"), ReckoningResult(0, sqrt(1 / 6))),
        ],
    )
    def test_reckon_pauli(self, reckoner, counts, pauli, expected):
        """Test reckon Pauli."""
        counts = Counts(counts)
        result = reckoner.reckon_pauli(counts, pauli)
        assert isinstance(result, ReckoningResult)
        for r, e in zip(result, expected):
            assert isclose(r, e)

    @mark.parametrize(
        "counts, expected",
        [
            ({}, ReckoningResult(0, 1)),
            ({0: 0}, ReckoningResult(0, 1)),
            ({0: 1}, ReckoningResult(1, 0)),
            ({1: 0}, ReckoningResult(0, 1)),
            ({1: 1}, ReckoningResult(-1, 0)),
            ({0: 1, 1: 1}, ReckoningResult(0, 1 / sqrt(2))),
        ],
    )
    def test_reckon_counts(self, reckoner, counts, expected):
        """Test reckon counts."""
        counts = Counts(counts)
        result = reckoner.reckon_counts(counts)
        assert isinstance(result, ReckoningResult)
        for r, e in zip(result, expected):
            assert isclose(r, e)

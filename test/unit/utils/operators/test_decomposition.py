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

"""Tests for quantum operators decomposition utils."""

from __future__ import annotations

from pytest import mark
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators import Pauli, PauliList, SparsePauliOp

from staged_primitives.utils.operators.decomposition import (
    AbelianDecomposer,
    NaiveDecomposer,
    OperatorDecomposer,
)


################################################################################
## TESTS
################################################################################
@mark.parametrize("decomposer", [NaiveDecomposer(), AbelianDecomposer()])
class TestOperatorDecomposer:
    """Test OperatorDecomposer."""

    def test_interface(self, decomposer):
        """Test decomposers extend the OperatorDecomposer interface."""
        assert isinstance(decomposer, OperatorDecomposer)

    @mark.parametrize(
        "unnormlized, normalized",
        [
            ("IXYZ", SparsePauliOp("IXYZ")),
            (PauliList(["IXYZ", "ZYXI"]), SparsePauliOp(["IXYZ", "ZYXI"])),
            (PauliSumOp(SparsePauliOp(["IXYZ", "IXII"])), SparsePauliOp(["IXYZ", "IXII"])),
            (
                PauliSumOp(SparsePauliOp(["IXYZ", "ZYXI", "IXII", "ZYII"])),
                SparsePauliOp(["IXYZ", "ZYXI", "IXII", "ZYII"]),
            ),
        ],
    )
    def test_normalization(self, decomposer, unnormlized, normalized):
        """Test that input operators are normalized appropriately."""
        assert decomposer.decompose(unnormlized) == decomposer.decompose(normalized)

    @mark.parametrize(
        "operator",
        [
            SparsePauliOp("IXYZ"),
            SparsePauliOp(["IXYZ", "ZYXI"]),
            SparsePauliOp(["IXYZ", "IXII"]),
            SparsePauliOp(["IXYZ", "ZYXI", "IXII", "ZYII"]),
        ],
    )
    def test_singlet_pauli_basis(self, decomposer, operator):
        """Test that each component from decomposition produces only one basis."""
        components = decomposer.decompose(operator)
        assert len(decomposer.extract_pauli_bases(operator)) == len(components)
        for component in components:
            bases = decomposer.extract_pauli_bases(component)
            assert len(bases) == 1


class TestNaiveDecomposer:
    """Test NaiveDecomposer."""

    @mark.parametrize(
        "operator, expected",
        [
            [SparsePauliOp("IXYZ"), (SparsePauliOp("IXYZ"),)],
            [SparsePauliOp(["IXYZ", "ZYXI"]), (SparsePauliOp("IXYZ"), SparsePauliOp("ZYXI"))],
            [SparsePauliOp(["IXYZ", "IXII"]), (SparsePauliOp("IXYZ"), SparsePauliOp("IXII"))],
            [
                SparsePauliOp(["IXYZ", "ZYXI", "IXII", "ZYII"]),
                (
                    SparsePauliOp("IXYZ"),
                    SparsePauliOp("ZYXI"),
                    SparsePauliOp("IXII"),
                    SparsePauliOp("ZYII"),
                ),
            ],
        ],
    )
    def test_decompose(self, operator, expected):
        """Test decompose in ObservableDecomposer strategies."""
        decomposer = NaiveDecomposer()
        components = decomposer.decompose(operator)
        assert components == expected

    @mark.parametrize(
        "operator, expected",
        [
            [SparsePauliOp("IXYZ"), PauliList(Pauli("IXYZ"))],
            [SparsePauliOp(["IXYZ", "ZYXI"]), PauliList([Pauli("IXYZ"), Pauli("ZYXI")])],
            [SparsePauliOp(["IXYZ", "IXII"]), PauliList([Pauli("IXYZ"), Pauli("IXII")])],
            [
                SparsePauliOp(["IXYZ", "ZYXI", "IXII", "ZYII"]),
                PauliList([Pauli("IXYZ"), Pauli("ZYXI"), Pauli("IXII"), Pauli("ZYII")]),
            ],
        ],
    )
    def test_pauli_bases(self, operator, expected):
        """Test Pauli basis in ObservableDecomposer strategies."""
        decomposer = NaiveDecomposer()
        basis = decomposer.extract_pauli_bases(operator)
        assert basis == expected


class TestAbelianDecomposer:
    """Test AbelianDecomposer."""

    @mark.parametrize(
        "operator, expected",
        [
            [SparsePauliOp("IXYZ"), (SparsePauliOp("IXYZ"),)],
            [SparsePauliOp(["IXYZ", "ZYXI"]), (SparsePauliOp("IXYZ"), SparsePauliOp("ZYXI"))],
            [SparsePauliOp(["IXYZ", "IXII"]), (SparsePauliOp(["IXYZ", "IXII"]),)],
            [
                SparsePauliOp(["IXYZ", "ZYXI", "IXII", "ZYII"]),
                (SparsePauliOp(["IXYZ", "IXII"]), SparsePauliOp(["ZYXI", "ZYII"])),
            ],
        ],
    )
    def test_decompose(self, operator, expected):
        """Test decompose in ObservableDecomposer strategies."""
        decomposer = AbelianDecomposer()
        components = decomposer.decompose(operator)
        assert components == expected

    @mark.parametrize(
        "operator, expected",
        [
            [SparsePauliOp("IXYZ"), PauliList(Pauli("IXYZ"))],
            [SparsePauliOp(["IXYZ", "ZYXI"]), PauliList([Pauli("IXYZ"), Pauli("ZYXI")])],
            [SparsePauliOp(["IXYZ", "IXII"]), PauliList([Pauli("IXYZ")])],
            [
                SparsePauliOp(["IXYZ", "ZYXI", "IXII", "ZYII"]),
                PauliList([Pauli("IXYZ"), Pauli("ZYXI")]),
            ],
        ],
    )
    def test_pauli_bases(self, operator, expected):
        """Test Pauli basis in ObservableDecomposer strategies."""
        decomposer = AbelianDecomposer()
        basis = decomposer.extract_pauli_bases(operator)
        assert basis == expected

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

"""Result reckoning utils."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import namedtuple
from collections.abc import Sequence
from typing import Union

from numpy import array, dot, real_if_close, sqrt, vstack
from qiskit.opflow import PauliSumOp
from qiskit.primitives.utils import init_observable as normalize_operator
from qiskit.quantum_info.operators import Pauli, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result import Counts

from staged_primitives.utils.binary import parity_bit
from staged_primitives.utils.operators import pauli_integer_mask

from .counts import bitmask_counts

ReckoningResult = namedtuple("ReckoningResult", ("expval", "std_error"))
OperatorType = Union[BaseOperator, PauliSumOp, str]  # TODO: to types


################################################################################
## EXPECTATION VALUE RECKONER INTERFACE
################################################################################
class ExpvalReckoner(ABC):
    """Expectation value reckoning interface.

    Classes implementing this interface provide methods for constructing expectation values
    and associated errors out of raw Counts and operators.
    """

    ################################################################################
    ## API
    ################################################################################
    def reckon(
        self,
        counts_list: Sequence[Counts] | Counts,
        operator_list: Sequence[OperatorType] | OperatorType,
    ) -> ReckoningResult:
        """Compute expectation value and associated std-error for input operators from counts.

        Note: the input operators need to be measurable entirely within one circuit
        execution (i.e. resulting in the one-to-one associated input counts). Users must
        ensure that all counts entries come from the appropriate circuit execution.

        args:
            counts: a :class:`~qiskit.result.Counts` object from circuit execution.
            operators: a list of operators associated one-to-one to the input counts.

        Returns:
            The expectation value and associated std-error for the sum of the input operators.
        """
        counts_list = self._validate_counts_list(counts_list)
        operator_list = self._validate_operator_list(operator_list)
        self._cross_validate_lists(counts_list, operator_list)
        expval, std_error = self._reckon(counts_list, operator_list)
        expval = real_if_close(expval).tolist()  # Note: `tolist` casts to python core numeric type
        std_error = array(std_error).tolist()
        return ReckoningResult(expval, std_error)

    def reckon_operator(self, counts: Counts, operator: OperatorType) -> ReckoningResult:
        """Reckon expectation value and associated std error from counts and operator.

        Note: This function assumes that the input operators are measurable entirely within
        one circuit execution (i.e. resulting in the input counts), and that the appropriate
        changes of bases (i.e. rotations) were actively performed in the relevant qubits before
        readout; hence diagonalizing the input operators.

        Returns:
            A two-tuple containing the expectation value and associated std error for the
            input operator. Expectation values can have both real and imaginary components,
            which can be interpreted as corresponding to the hermitian and anti-hermitian
            components of the input operator respectively. Standard errors will always be real.
        """
        counts = self._validate_counts(counts)
        operator = self._validate_operator(operator)
        # TODO: cross-validation
        expval, std_error = self._reckon_operator(counts, operator)
        expval = real_if_close(expval).tolist()  # Note: `tolist` casts to python core numeric type
        std_error = array(std_error).tolist()
        return ReckoningResult(expval, std_error)

    def reckon_pauli(self, counts: Counts, pauli: Pauli) -> ReckoningResult:
        """Reckon expectation value and associated std error from counts and pauli.

        Note: This function treats X, Y, and Z Paulis identically, assuming that the appropriate
        changes of bases (i.e. rotations) were actively performed in the relevant qubits before
        readout; hence diagonalizing the input Pauli.
        """
        counts = self._validate_counts(counts)
        pauli = self._validate_pauli(pauli)
        # TODO: cross-validation
        expval, std_error = self._reckon_pauli(counts, pauli)
        expval = real_if_close(expval).tolist()  # Note: `tolist` casts to python core numeric type
        std_error = array(std_error).tolist()
        return ReckoningResult(expval, std_error)

    def reckon_counts(self, counts: Counts) -> ReckoningResult:
        """Reckon expectation value and associated std error from counts.

        Note: The measurement basis is implicit in the way the input counts were produced,
        therefore the resulting value can be regarded as coming from a multi-qubit Pauli-Z
        operator (i.e. a fully diagonal Pauli operator).
        """
        counts = self._validate_counts(counts)
        expval, std_error = self._reckon_counts(counts)
        expval = real_if_close(expval).tolist()  # Note: `tolist` casts to python core numeric type
        std_error = array(std_error).tolist()
        return ReckoningResult(expval, std_error)

    ################################################################################
    ## ABSTRACT METHODS
    ################################################################################
    @abstractmethod
    def _reckon(
        self,
        counts_list: Sequence[Counts],
        operator_list: Sequence[SparsePauliOp],
    ) -> ReckoningResult:
        expval = 0.0
        variance = 0.0
        for value, error in (
            self._reckon_operator(counts, operator)
            for counts, operator in zip(counts_list, operator_list)
        ):
            expval += value
            variance += error**2
        return ReckoningResult(expval, sqrt(variance))

    @abstractmethod
    def _reckon_operator(self, counts: Counts, operator: SparsePauliOp) -> ReckoningResult:
        value_std_error_pairs = [self._reckon_pauli(counts, pauli) for pauli in operator.paulis]
        values, std_errors = vstack(value_std_error_pairs).T  # Note: like zip but array output
        coeffs = array(operator.coeffs)
        expval = dot(values, coeffs)
        variance = dot(std_errors**2, (coeffs.real**2 + coeffs.imag**2)).real
        return ReckoningResult(expval, sqrt(variance))

    @abstractmethod
    def _reckon_pauli(self, counts: Counts, pauli: Pauli) -> ReckoningResult:
        mask = pauli_integer_mask(pauli)
        counts = bitmask_counts(counts, mask)
        coeff = (-1j) ** pauli.phase
        expval, std_error = self._reckon_counts(counts)
        return ReckoningResult(coeff * expval, std_error)

    @abstractmethod
    def _reckon_counts(self, counts: Counts) -> ReckoningResult:
        shots = counts.shots() or 1  # Note: avoid division by zero errors
        expval: float = 0.0
        for readout, freq in counts.int_outcomes().items():
            observation = (-1) ** parity_bit(readout, even=True)
            expval += observation * freq / shots
        variance = 1 - expval**2
        std_error = sqrt(variance / shots)
        return ReckoningResult(expval, std_error)

    ################################################################################
    ## AUXILIARY
    ################################################################################
    @classmethod
    def _validate_counts_list(cls, counts_list: Sequence[Counts] | Counts) -> tuple[Counts, ...]:
        """Validate counts list."""
        if isinstance(counts_list, (Counts, dict)):
            counts_list = (counts_list,)
        if not isinstance(counts_list, Sequence):
            raise TypeError("Expected Sequence object.")
        return tuple(cls._validate_counts(c) for c in counts_list)

    @staticmethod
    def _validate_counts(counts: Counts | dict) -> Counts:
        """Validate counts."""
        if isinstance(counts, dict):
            counts = Counts(counts)
        if not isinstance(counts, Counts):
            raise TypeError("Expected Counts object.")
        return counts

    @classmethod
    def _validate_operator_list(
        cls,
        operator_list: Sequence[OperatorType] | OperatorType,
    ) -> tuple[SparsePauliOp, ...]:
        """Validate operator list."""
        if isinstance(operator_list, (BaseOperator, PauliSumOp, str)):
            operator_list = (operator_list,)
        if not isinstance(operator_list, Sequence):
            raise TypeError("Expected Sequence object.")
        return tuple(cls._validate_operator(o) for o in operator_list)

    @staticmethod
    def _validate_operator(operator: OperatorType) -> SparsePauliOp:
        """Validate operator."""
        if isinstance(operator, (BaseOperator, PauliSumOp, str)):
            return normalize_operator(operator)
        raise TypeError("Expected OperatorType object.")

    @staticmethod
    def _validate_pauli(pauli: Pauli | str) -> Pauli:
        """Validate Pauli."""
        if isinstance(pauli, str):
            pauli = Pauli(pauli)
        elif not isinstance(pauli, Pauli):
            raise TypeError(f"Expected Pauli, got {pauli!r} instead.")
        return pauli

    @staticmethod
    def _cross_validate_lists(
        counts_list: Sequence[Counts], operator_list: Sequence[SparsePauliOp]
    ) -> None:
        """Cross validate counts and operator lists."""
        # TODO: validate num_bits -> Need to check every entry in counts (expensive)
        if len(counts_list) != len(operator_list):
            raise ValueError(
                f"The number of counts entries ({len(counts_list)}) does not match "
                f"the number of operators ({len(operator_list)})."
            )


################################################################################
## IMPLEMENTATION
################################################################################
# pylint: disable=useless-parent-delegation
class CanonicalReckoner(ExpvalReckoner):
    """Canonical expectation value reckoning class."""

    def _reckon(
        self, counts_list: Sequence[Counts], operator_list: Sequence[SparsePauliOp]
    ) -> ReckoningResult:
        return super()._reckon(counts_list, operator_list)

    def _reckon_operator(self, counts: Counts, operator: SparsePauliOp) -> ReckoningResult:
        return super()._reckon_operator(counts, operator)

    def _reckon_pauli(self, counts: Counts, pauli: Pauli) -> ReckoningResult:
        return super()._reckon_pauli(counts, pauli)

    def _reckon_counts(self, counts: Counts) -> ReckoningResult:
        return super()._reckon_counts(counts)

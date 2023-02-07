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

from numpy import array, dot, sqrt, vstack
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
    and associated errors out of raw Counts and observables.
    """

    ################################################################################
    ## API
    ################################################################################
    def reckon(
        self,
        counts_list: Sequence[Counts] | Counts,
        observable_list: Sequence[OperatorType] | OperatorType,
    ) -> ReckoningResult:
        """Compute expectation value and associated std-error for input observables from counts.

        Note: the input observables need to be measurable entirely within one circuit
        execution (i.e. resulting in the one-to-one associated input counts). Users must
        ensure that all counts entries come from the appropriate circuit execution.

        args:
            counts: a :class:`~qiskit.result.Counts` object from circuit execution.
            observables: a list of observables associated one-to-one to the input counts.

        Returns:
            The expectation value and associated std-error for the sum of the input observables.
        """
        counts_list = self._validate_counts_list(counts_list)
        observable_list = self._validate_observable_list(observable_list)
        self._cross_validate_lists(counts_list, observable_list)
        return self._reckon(counts_list, observable_list)

    # TODO: reckon operator for non-hermitian
    def reckon_observable(self, counts: Counts, observable: OperatorType) -> ReckoningResult:
        """Reckon expectation value and associated std error from counts and observable.

        Note: This function assumes that the input observables are measurable entirely within
        one circuit execution (i.e. resulting in the input counts), and that the appropriate
        changes of bases (i.e. rotations) were actively performed in the relevant qubits before
        readout; hence diagonalizing the input observables.
        """
        counts = self._validate_counts(counts)
        observable = self._validate_observable(observable)
        # TODO: cross-validation
        return self._reckon_observable(counts, observable)

    def reckon_pauli(self, counts: Counts, pauli: Pauli) -> ReckoningResult:
        """Reckon expectation value and associated std error from counts and pauli.

        Note: This function treats X, Y, and Z Paulis identically, assuming that the appropriate
        changes of bases (i.e. rotations) were actively performed in the relevant qubits before
        readout; hence diagonalizing the input Pauli.
        """
        counts = self._validate_counts(counts)
        pauli = self._validate_pauli(pauli)
        # TODO: cross-validation
        return self._reckon_pauli(counts, pauli)

    def reckon_counts(self, counts: Counts) -> ReckoningResult:
        """Reckon expectation value and associated std error from counts.

        Note: The measurement basis is implicit in the way the input counts were produced,
        therefore the resulting value can be regarded as coming from a multi-qubit Pauli-Z
        observable (i.e. a fully diagonal Pauli observable).
        """
        counts = self._validate_counts(counts)
        return self._reckon_counts(counts)

    ################################################################################
    ## ABSTRACT METHODS
    ################################################################################
    @abstractmethod
    def _reckon(
        self,
        counts_list: Sequence[Counts],
        observable_list: Sequence[SparsePauliOp],
    ) -> ReckoningResult:
        expval = 0.0
        variance = 0.0
        for value, error in (
            self._reckon_observable(counts, observable)
            for counts, observable in zip(counts_list, observable_list)
        ):
            expval += value
            variance += error**2
        return ReckoningResult(expval, sqrt(variance))

    @abstractmethod
    def _reckon_observable(self, counts: Counts, observable: SparsePauliOp) -> ReckoningResult:
        observable = normalize_operator(observable)
        values, std_errors = vstack(
            [self._reckon_pauli(counts, pauli) for pauli in observable.paulis]
        ).T
        coeffs = array(observable.coeffs)
        expval = dot(values, coeffs)
        variance = dot(std_errors**2, coeffs**2)  # TODO: complex coeffs
        return ReckoningResult(expval, sqrt(variance))

    @abstractmethod
    def _reckon_pauli(self, counts: Counts, pauli: Pauli) -> ReckoningResult:
        mask = pauli_integer_mask(pauli)
        counts = bitmask_counts(counts, mask)
        return self._reckon_counts(counts)

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
        if isinstance(counts_list, Counts):
            counts_list = (counts_list,)
        if not isinstance(counts_list, Sequence):
            raise TypeError("Expected Sequence object.")
        return tuple(cls._validate_counts(c) for c in counts_list)

    @staticmethod
    def _validate_counts(counts: Counts) -> Counts:
        """Validate counts."""
        # TODO: accept dict[int, int]
        if isinstance(counts, Counts):
            return counts
        raise TypeError("Expected Counts object.")

    @classmethod
    def _validate_observable_list(
        cls,
        observable_list: Sequence[OperatorType] | OperatorType,
    ) -> tuple[SparsePauliOp, ...]:
        """Validate observable list."""
        if isinstance(observable_list, (BaseOperator, PauliSumOp, str)):
            observable_list = (observable_list,)
        if not isinstance(observable_list, Sequence):
            raise TypeError("Expected Sequence object.")
        return tuple(cls._validate_observable(o) for o in observable_list)

    @staticmethod
    def _validate_observable(observable: OperatorType) -> SparsePauliOp:
        """Validate observable."""
        if isinstance(observable, (BaseOperator, PauliSumOp, str)):
            return normalize_operator(observable)
        raise TypeError("Expected OperatorType object.")

    @staticmethod
    def _validate_pauli(pauli: Pauli) -> Pauli:
        """Validate Pauli."""
        # TODO: accept str
        if isinstance(pauli, Pauli):
            return pauli
        raise TypeError("Expected Pauli object.")

    @staticmethod
    def _cross_validate_lists(
        counts_list: Sequence[Counts], observable_list: Sequence[SparsePauliOp]
    ) -> None:
        """Cross validate counts and observable lists."""
        # TODO: validate num_bits -> Need to check every entry in counts (expensive)
        if len(counts_list) != len(observable_list):
            raise ValueError(
                f"The number of counts entries ({len(counts_list)}) does not match "
                f"the number of observables ({len(observable_list)})."
            )


################################################################################
## IMPLEMENTATION
################################################################################
# pylint: disable=useless-parent-delegation
class CanonicalReckoner(ExpvalReckoner):
    """Canonical expectation value reckoning class."""

    def _reckon(
        self, counts_list: Sequence[Counts], observable_list: Sequence[SparsePauliOp]
    ) -> ReckoningResult:
        return super()._reckon(counts_list, observable_list)

    def _reckon_observable(self, counts: Counts, observable: SparsePauliOp) -> ReckoningResult:
        return super()._reckon_observable(counts, observable)

    def _reckon_pauli(self, counts: Counts, pauli: Pauli) -> ReckoningResult:
        return super()._reckon_pauli(counts, pauli)

    def _reckon_counts(self, counts: Counts) -> ReckoningResult:
        return super()._reckon_counts(counts)

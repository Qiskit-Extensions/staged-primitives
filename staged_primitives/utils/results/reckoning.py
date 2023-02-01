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

from .results import bitmask_counts

################################################################################
## UTILS
################################################################################
ReckoningResult = namedtuple("ReckoningResult", ("expval", "std_error"))
OperatorType = Union[BaseOperator, PauliSumOp, str]  # TODO: to types


def reckon_expval(counts: Counts) -> ReckoningResult:
    """Reckon expectation value and associated std error from counts.

    Note: The measurement basis is implicit in the way the input counts were produced,
    therefore the resulting value can be regarded as coming from a multi-qubit Pauli-Z
    observable (i.e. a fully diagonal Pauli observable).
    """
    shots = counts.shots() or 1  # Note: avoid division by zero errors
    expval: float = 0.0
    for readout, freq in counts.int_outcomes().items():
        observation = (-1) ** parity_bit(readout, even=True)
        expval += observation * freq / shots
    variance = 1 - expval**2
    std_error = sqrt(variance / shots)
    return ReckoningResult(expval, std_error)


def reckon_pauli(counts: Counts, pauli: Pauli) -> ReckoningResult:
    """Reckon expectation value and associated std error from counts and pauli.

    Note: This function treats X, Y, and Z Paulis identically, assuming that the appropriate
    changes of bases (i.e. rotations) were actively performed in the relevant qubits before
    readout; hence diagonalizing the input Pauli.
    """
    mask = pauli_integer_mask(pauli)
    counts = bitmask_counts(counts, mask)
    return reckon_expval(counts)


# TODO: `reckon_operator` for non-hermitian inputs
def reckon_observable(counts: Counts, observable: OperatorType) -> ReckoningResult:
    """Reckon expectation value and associated std error from counts and observable.

    Note: This function assumes that the input observables are measurable entirely within
    one circuit execution (i.e. resulting in the input counts), and that the appropriate
    changes of bases (i.e. rotations) were actively performed in the relevant qubits before
    readout; hence diagonalizing the input observables.
    """
    observable = normalize_operator(observable)
    values, std_errors = vstack([reckon_pauli(counts, pauli) for pauli in observable.paulis]).T
    coeffs = array(observable.coeffs)
    expval = dot(values, coeffs)
    variance = dot(std_errors**2, coeffs**2)  # TODO: complex coeffs
    return ReckoningResult(expval, sqrt(variance))


################################################################################
## EXPECTATION VALUE RECKONER INTERFACE
################################################################################
class ExpvalReckoner(ABC):  # pylint: disable=too-few-public-methods
    """Expectation value reckoning interface.

    Classes implementing this interface provide methods for constructing expectation values
    and associated errors out of raw Counts and observables.
    """

    ################################################################################
    ## INTERFACE
    ################################################################################
    def reckon(
        self,
        counts_list: Sequence[Counts] | Counts,
        observables: Sequence[OperatorType] | OperatorType,
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
        counts_list = self._validate_counts(counts_list)
        observables = self._validate_observables(observables)
        self._cross_validate(counts_list, observables)
        expval = 0.0
        variance = 0.0
        for value, error in (self._reckon_single(c, o) for c, o in zip(counts_list, observables)):
            expval += value
            variance += error**2
        return ReckoningResult(expval, sqrt(variance))

    @abstractmethod
    def _reckon_single(
        self,
        counts: Counts,
        observable: SparsePauliOp,
    ) -> ReckoningResult:
        """Single input version of `reckon`."""

    ################################################################################
    ## AUXILIARY
    ################################################################################
    @staticmethod
    def _validate_counts(counts_list: Sequence[Counts] | Counts) -> tuple[Counts, ...]:
        """Validate counts."""
        if isinstance(counts_list, Counts):
            counts_list = (counts_list,)
        if not isinstance(counts_list, Sequence):
            raise TypeError("Expected Sequence object.")
        if any(not isinstance(c, Counts) for c in counts_list):
            raise TypeError("Expected Counts object.")
        return tuple(counts_list)

    @staticmethod
    def _validate_observables(
        observables: Sequence[OperatorType] | OperatorType,
    ) -> tuple[SparsePauliOp, ...]:
        """Validate observables."""
        if isinstance(observables, (BaseOperator, PauliSumOp, str)):
            observables = (observables,)
        if not isinstance(observables, Sequence):
            raise TypeError("Expected Sequence object.")
        if any(not isinstance(o, (BaseOperator, PauliSumOp, str)) for o in observables):
            raise TypeError("Expected OperatorType object.")
        return tuple(normalize_operator(o) for o in observables)

    @staticmethod
    def _cross_validate(
        counts_list: Sequence[Counts], observables: Sequence[SparsePauliOp]
    ) -> None:
        """Cross validate counts and observables."""
        # TODO: validate num_bits -> Need to check every entry in counts (expensive)
        if len(counts_list) != len(observables):
            raise ValueError(
                f"The number of counts entries ({len(counts_list)}) does not match "
                f"the number of observables ({len(observables)})."
            )


################################################################################
## IMPLEMENTATION
################################################################################
class CanonicalReckoner(ExpvalReckoner):  # pylint: disable=too-few-public-methods
    """Canonical expectation value reckoning class."""

    def _reckon_single(
        self,
        counts: Counts,
        observable: SparsePauliOp,
    ) -> ReckoningResult:
        return reckon_observable(counts, observable)

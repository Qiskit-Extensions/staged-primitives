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

"""Quantum operators decomposition utils."""

from __future__ import annotations

from abc import ABC, abstractmethod

from numpy import logical_or
from qiskit.opflow import PauliSumOp
from qiskit.primitives.utils import init_observable as normalize_operator  # TODO
from qiskit.quantum_info.operators import Pauli, PauliList, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator


################################################################################
## OPERATOR DECOMPOSER INTERFACE
################################################################################
class OperatorDecomposer(ABC):
    """Strategy interface for decomposing operators and getting associated measurement bases."""

    def decompose(self, operator: BaseOperator | PauliSumOp | str) -> tuple[SparsePauliOp, ...]:
        """Decomposes a given operator into singly measurable components.

        Note that component decomposition is not unique, for instance, commuting components
        could be grouped together in different ways (i.e. partitioning the set).

        Args:
            operator: the operator to decompose into its core components.

        Returns:
            A list of operators each of which measurable with a single quantum circuit
            (i.e. on a singlet Pauli basis).
        """
        operator = normalize_operator(operator)
        return self._decompose(operator)

    @abstractmethod
    def _decompose(
        self,
        operator: SparsePauliOp,
    ) -> tuple[SparsePauliOp, ...]:
        """Input-standardized version of `decompose`."""

    def extract_pauli_bases(self, operator: BaseOperator | PauliSumOp | str) -> PauliList:
        """Extract Pauli bases for a given operator.

        Note that the resulting basis may be overcomplete depending on the implementation.

        Args:
            operator: an operator for which to obtain a Pauli basis for measurement.

        Returns:
            A `PauliList` of operators serving as a basis for the input operator. Each
            entry conrresponds one-to-one to the components retrieved from `.decompose()`.
        """
        components = self.decompose(operator)
        paulis = tuple(self._extract_singlet_pauli_basis(component) for component in components)
        return PauliList(paulis)  # TODO: Allow `PauliList` from generator (i.e. Qiskit-Terra)

    @abstractmethod
    def _extract_singlet_pauli_basis(self, operator: SparsePauliOp) -> Pauli:
        """Extract singlet Pauli basis for a given operator.

        The input operator comes from `._decompose()`, and must be singly measurable.
        """


################################################################################
## IMPLEMENTATIONS
################################################################################
class NaiveDecomposer(OperatorDecomposer):
    """Trivial operator decomposition without grouping components."""

    def _decompose(
        self,
        operator: SparsePauliOp,
    ) -> tuple[SparsePauliOp, ...]:
        return tuple(operator)

    def _extract_singlet_pauli_basis(self, operator: SparsePauliOp) -> Pauli:
        return operator.paulis[0]


class AbelianDecomposer(OperatorDecomposer):
    """Abelian operator decomposition grouping commuting components."""

    def _decompose(
        self,
        operator: SparsePauliOp,
    ) -> tuple[SparsePauliOp, ...]:
        components = operator.group_commuting(qubit_wise=True)
        return tuple(components)

    def _extract_singlet_pauli_basis(self, operator: SparsePauliOp) -> Pauli:
        or_reduce = logical_or.reduce
        zx_data_tuple = or_reduce(operator.paulis.z), or_reduce(operator.paulis.x)
        return Pauli(zx_data_tuple)

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

"""Integration testing."""

from pytest import mark
from qiskit.providers.fake_provider import FakeNairobi, FakeNairobiV2


@mark.filterwarnings("ignore:.*Aer.*")
class TestFromQiskit:
    """All tests under this class and subclasses were ported from Qiskit-Terra."""


@mark.filterwarnings("ignore:.*Aer.*")
@mark.parametrize("backend", [FakeNairobi(), FakeNairobiV2()])
class TestOnBackends:
    """Integration tests on backends"""

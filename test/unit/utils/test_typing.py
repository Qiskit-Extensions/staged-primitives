# This code is part of Qiskit.
#
# (C) Copyright IBM 2022-2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from pytest import mark

from staged_primitives.utils.typing import isinteger, isreal


################################################################################
## TYPE CHECKING
################################################################################
@mark.parametrize("object", [0, 1, -1, 1.0, -1.0, True, False])
def test_isinteger_true(object):
    assert isinteger(object)


@mark.parametrize("object", [1.2, -2.4])
def test_isinteger_false(object):
    assert not isinteger(object)


@mark.parametrize("object", [0, 1, -1, 1.2, -2.4, True, False])
def test_isreal_true(object):
    assert isreal(object)


@mark.parametrize("object", [float("nan"), float("inf"), float("-inf"), {}, (0, 1)])
def test_isreal_false(object):
    assert not isreal(object)

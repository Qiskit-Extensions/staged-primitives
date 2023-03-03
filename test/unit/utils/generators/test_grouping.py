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

from test import NO_INTS, NO_ITERS

from pytest import mark, raises

from staged_primitives.utils.generators import group_elements


@mark.parametrize(
    "elements, size, expected",
    cases := [
        ([], 1, []),
        ([], 2, []),
        ([], 3, []),
        ([0], 1, [(0,)]),
        ([0], 2, []),
        ([0], 3, []),
        ([0, 1], 1, [(0,), (1,)]),
        ([0, 1], 2, [(0, 1)]),
        ([0, 1], 3, []),
        ([0, 1, 2], 1, [(0,), (1,), (2,)]),
        ([0, 1, 2], 2, [(0, 1)]),
        ([0, 1, 2], 3, [(0, 1, 2)]),
        ([0, 1, 2, 3], 1, [(0,), (1,), (2,), (3,)]),
        ([0, 1, 2, 3], 2, [(0, 1), (2, 3)]),
        ([0, 1, 2, 3], 3, [(0, 1, 2)]),
        ([0, 1, 2, 3], 4, [(0, 1, 2, 3)]),
    ],
    ids=[f"list<{len(value)}>-{size}" for value, size, expected in cases],
)
def test_group_elements(elements, size, expected):
    groups = group_elements(elements, group_size=size)
    assert list(groups) == list(expected)


@mark.parametrize(
    "elements",
    NO_ITERS,
    ids=[str(type(i).__name__) for i in NO_ITERS],
)
def test_group_elements_type_error_iter(elements):
    with raises(TypeError):
        generator = group_elements(elements, group_size=1)
        assert next(generator)


@mark.parametrize(
    "size",
    NO_INTS + [0, -1],
    ids=[str(type(i).__name__) for i in NO_INTS] + [0, -1],
)
def test_group_elements_type_error_int(size):
    with raises(TypeError):
        generator = group_elements(elements=[], group_size=size)
        assert next(generator)

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

"""Tests for binary utils."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from staged_primitives.utils.time import isotimestamp


################################################################################
## TESTS
################################################################################
class TestISOTimeStamp:
    """Test ISO time stamp."""

    def test_utc(self):
        """Test ISO time stamp UTC."""
        expected = "1994-03-04T14:30:00Z"
        with patch("staged_primitives.utils.time.datetime") as datetime_mock:
            datetime_mock.utcnow.return_value = datetime.fromisoformat(expected.strip("Z"))
            stamp = isotimestamp()
        assert stamp == expected

    def test_timezone(self):
        """Test ISO time stamp timezone."""
        expected = "1994-03-04T15:30:00+01:00"
        with patch("staged_primitives.utils.time.datetime") as datetime_mock:
            datetime_mock.now().astimezone.return_value = datetime.fromisoformat(expected)
            stamp = isotimestamp(timezone=True)
        assert stamp == expected

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

from pytest import mark

from staged_primitives.utils.time import (
    elapsed_days,
    elapsed_hours,
    elapsed_minutes,
    elapsed_seconds,
    isotimestamp,
)


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


class TestElapsedTime:
    """Test elapsed time."""

    @mark.parametrize(
        "timestamp, now, expected",
        [
            ("1970-01-01T00:00:00Z", "1970-01-01T00:00:04Z", 4),
            ("1970-01-01T00:00:00Z", "1970-01-01T00:00:12Z", 12),
            ("1970-01-01T00:00:00Z", "1970-01-01T00:01:00Z", 60),
            ("1970-01-01T00:00:00Z", "1970-01-01T00:12:00Z", 60 * 12),
            ("1970-01-01T00:00:00Z", "1970-01-01T02:00:00Z", 3600 * 2),
            ("1970-01-01T00:00:00Z", "1970-01-02T00:00:00Z", 3600 * 24),
            ("1970-01-01T00:00:00Z", "1970-02-01T00:00:00Z", 3600 * 24 * 31),
        ],
    )
    def test_elapsed_seconds(self, timestamp, now, expected):
        """Test elapsed seconds."""
        now = now.replace("Z", "+00:00")
        with patch("staged_primitives.utils.time.datetime") as datetime_mock:
            datetime_mock.now.return_value = datetime.fromisoformat(now)
            datetime_mock.fromisoformat.side_effect = datetime.fromisoformat
            assert elapsed_seconds(timestamp) == expected

    @mark.parametrize(
        "timestamp, now, expected",
        [
            ("1970-01-01T00:00:00Z", "1970-01-01T00:00:04Z", 0),
            ("1970-01-01T00:00:00Z", "1970-01-01T00:00:12Z", 0),
            ("1970-01-01T00:00:00Z", "1970-01-01T00:01:00Z", 1),
            ("1970-01-01T00:00:00Z", "1970-01-01T00:12:00Z", 12),
            ("1970-01-01T00:00:00Z", "1970-01-01T02:00:00Z", 60 * 2),
            ("1970-01-01T00:00:00Z", "1970-01-02T00:00:00Z", 60 * 24),
            ("1970-01-01T00:00:00Z", "1970-02-01T00:00:00Z", 60 * 24 * 31),
        ],
    )
    def test_elapsed_minutes(self, timestamp, now, expected):
        """Test elapsed minutes."""
        now = now.replace("Z", "+00:00")
        with patch("staged_primitives.utils.time.datetime") as datetime_mock:
            datetime_mock.now.return_value = datetime.fromisoformat(now)
            datetime_mock.fromisoformat.side_effect = datetime.fromisoformat
            assert elapsed_minutes(timestamp) == expected

    @mark.parametrize(
        "timestamp, now, expected",
        [
            ("1970-01-01T00:00:00Z", "1970-01-01T00:00:04Z", 0),
            ("1970-01-01T00:00:00Z", "1970-01-01T00:00:12Z", 0),
            ("1970-01-01T00:00:00Z", "1970-01-01T00:01:00Z", 0),
            ("1970-01-01T00:00:00Z", "1970-01-01T00:12:00Z", 0),
            ("1970-01-01T00:00:00Z", "1970-01-01T02:00:00Z", 2),
            ("1970-01-01T00:00:00Z", "1970-01-02T00:00:00Z", 24),
            ("1970-01-01T00:00:00Z", "1970-02-01T00:00:00Z", 24 * 31),
        ],
    )
    def test_elapsed_hours(self, timestamp, now, expected):
        """Test elapsed hours."""
        now = now.replace("Z", "+00:00")
        with patch("staged_primitives.utils.time.datetime") as datetime_mock:
            datetime_mock.now.return_value = datetime.fromisoformat(now)
            datetime_mock.fromisoformat.side_effect = datetime.fromisoformat
            assert elapsed_hours(timestamp) == expected

    @mark.parametrize(
        "timestamp, now, expected",
        [
            ("1970-01-01T00:00:00Z", "1970-01-01T00:00:04Z", 0),
            ("1970-01-01T00:00:00Z", "1970-01-01T00:00:12Z", 0),
            ("1970-01-01T00:00:00Z", "1970-01-01T00:01:00Z", 0),
            ("1970-01-01T00:00:00Z", "1970-01-01T00:12:00Z", 0),
            ("1970-01-01T00:00:00Z", "1970-01-01T02:00:00Z", 0),
            ("1970-01-01T00:00:00Z", "1970-01-02T00:00:00Z", 1),
            ("1970-01-01T00:00:00Z", "1970-02-01T00:00:00Z", 31),
        ],
    )
    def test_elapsed_days(self, timestamp, now, expected):
        """Test elapsed days."""
        now = now.replace("Z", "+00:00")
        with patch("staged_primitives.utils.time.datetime") as datetime_mock:
            datetime_mock.now.return_value = datetime.fromisoformat(now)
            datetime_mock.fromisoformat.side_effect = datetime.fromisoformat
            assert elapsed_days(timestamp) == expected

    @mark.parametrize(
        "timestamp, now, expected",
        [
            ("1970-01-01T00:00:00+00:00", "1970-01-01T00:00:00Z", 0),
            ("1970-01-01T00:00:00Z", "1970-01-01T00:00:00+00:00", 0),
            ("1970-01-01T00:00:00+01:00", "1970-01-01T00:00:00Z", 1),
            ("1970-01-01T00:00:00Z", "1970-01-01T12:00:00+01:00", 11),
            ("1970-01-01T00:00:00+01:00", "1970-01-01T00:00:00-05:00", 6),
        ],
    )
    def test_timezone(self, timestamp, now, expected):
        """Test elapsed time in different timezones."""
        now = now.replace("Z", "+00:00")
        with patch("staged_primitives.utils.time.datetime") as datetime_mock:
            datetime_mock.now.return_value = datetime.fromisoformat(now)
            datetime_mock.fromisoformat.side_effect = datetime.fromisoformat
            assert elapsed_hours(timestamp) == expected

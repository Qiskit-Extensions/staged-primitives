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

"""Time utils."""

from __future__ import annotations

from datetime import datetime, timezone


################################################################################
## UTILS
################################################################################
def isotimestamp(timezone: bool = False) -> str:  # pylint: disable=redefined-outer-name
    """Generate ISO-8601 timestamp."""
    if timezone:
        return f"{datetime.now().astimezone().isoformat()}"
    return f"{datetime.utcnow().isoformat()}Z"


def elapsed_seconds(isotimestamp: str) -> int:  # pylint: disable=redefined-outer-name
    """Elapsed time in seconds from input ISO-8601 timestamp."""
    isotimestamp = isotimestamp.replace("Z", "+00:00")
    elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(isotimestamp)
    return (elapsed.days * 24 * 3600) + elapsed.seconds


def elapsed_minutes(isotimestamp: str) -> int:  # pylint: disable=redefined-outer-name
    """Elapsed time in minutes from input ISO-8601 timestamp."""
    return elapsed_seconds(isotimestamp) // 60


def elapsed_hours(isotimestamp: str) -> int:  # pylint: disable=redefined-outer-name
    """Elapsed time in hours from input ISO-8601 timestamp."""
    return elapsed_minutes(isotimestamp) // 60


def elapsed_days(isotimestamp: str) -> int:  # pylint: disable=redefined-outer-name
    """Elapsed time in days from input ISO-8601 timestamp."""
    isotimestamp = isotimestamp.replace("Z", "+00:00")
    elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(isotimestamp.strip("Z"))
    return elapsed.days

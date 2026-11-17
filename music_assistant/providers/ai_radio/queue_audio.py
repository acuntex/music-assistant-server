"""
The voice level and ducking depth a queue's DJ mixes its program at.

Both are a per-queue setting, stored next to the queue's host in the queue DJ file. They are
additive: an entry written before the settings existed carries neither, and reads as the
default (full voice level, the ducking depth the streams bus uses for every queue).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from music_assistant_models.errors import InvalidDataError

from music_assistant.controllers.streams.program_levels import (
    DUCK_MAX_DB,
    DUCK_MIN_DB,
    VOICE_LEVEL_MAX_DB,
    VOICE_LEVEL_MIN_DB,
    checked_db,
)

VOICE_LEVEL_DEFAULT_DB = VOICE_LEVEL_MAX_DB


@dataclass(frozen=True, slots=True)
class QueueAudioLevels:
    """How loud a queue's DJ speaks, and how far it ducks the music under itself."""

    voice_level_db: float = VOICE_LEVEL_DEFAULT_DB
    # None keeps the ducking depth a bus without a queue setting mixes with
    duck_db: float | None = None


def normalize_voice_level_db(value: Any) -> float:
    """
    Validate the level a queue's spoken contributions are mixed in at.

    A queue stored before the setting existed carries no value, and reads as the default.

    :param value: The configured level in decibels, or None.
    :raises InvalidDataError: When the value is not a number or above 0 dB or below -9 dB.
    """
    if value is None:
        return VOICE_LEVEL_DEFAULT_DB
    return _checked_db(value, "voice_level_db", VOICE_LEVEL_MIN_DB, VOICE_LEVEL_MAX_DB)


def normalize_duck_db(value: Any) -> float | None:
    """
    Validate how far the queue's music goes down under a spoken contribution.

    A queue stored before the setting existed carries no value and keeps the depth the whole
    server ducks with, which is also what None means here.

    :param value: The configured level in decibels, or None to inherit the global depth.
    :raises InvalidDataError: When the value is not a number or above 0 dB or below -12 dB.
    """
    if value is None:
        return None
    return _checked_db(value, "duck_db", DUCK_MIN_DB, DUCK_MAX_DB)


def levels_from_stored(entry: Any) -> QueueAudioLevels:
    """
    Return the levels a stored queue DJ entry carries, defaulting what it does not have.

    A file written before the settings existed has neither field, and a value that is not
    one this version accepts is replaced by the default instead of dropping the DJ: the
    queue keeps speaking, at the level it spoke at before there was a setting.

    :param entry: One queue's entry of the queue DJ file, if it is a mapping at all.
    """
    if not isinstance(entry, dict):
        return QueueAudioLevels()
    try:
        return QueueAudioLevels(
            voice_level_db=normalize_voice_level_db(entry.get("voice_level_db")),
            duck_db=normalize_duck_db(entry.get("duck_db")),
        )
    except InvalidDataError:
        return QueueAudioLevels()


def _checked_db(value: Any, field: str, minimum: float, maximum: float) -> float:
    """Return one decibel value inside its range, as this provider reports a bad one."""
    try:
        return checked_db(value, field, minimum, maximum)
    except ValueError as err:
        # the settings api of this provider answers a bad argument with its own error type
        raise InvalidDataError(str(err)) from err

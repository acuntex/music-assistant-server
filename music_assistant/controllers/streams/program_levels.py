"""
The level a queue mixes its programs at, and how far it ducks the music under them.

The voice level sits on the bus, before a program is summed over the music, so it moves the
voice and not the music below it: an output gain would take both down. The ducking depth is
the envelope the same queue's music is pulled down with while a program plays. Both are a
per-queue setting, and a queue without one mixes exactly as a bus always did.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from music_assistant.controllers.streams.smart_fades.filters import VOICE_OVER_DUCK_DEPTH

# a program may be attenuated but never boosted: the bus mixes under a fixed ceiling and has
# no headroom to give away
VOICE_LEVEL_MAX_DB = 0.0
VOICE_LEVEL_MIN_DB = -9.0
# how far the music goes down under a program; 0 leaves it where it is
DUCK_MAX_DB = 0.0
DUCK_MIN_DB = -12.0


def gain_of_db(db: float) -> float:
    """Return the linear gain of the given decibel value."""
    return 10 ** (db / 20)


def duck_depth_of_db(db: float | None) -> float:
    """
    Return the share of the music the ducking pulls down, as the envelope expects it.

    :param db: How far the music goes down under a program, or None to keep the depth a
        boundary's voice-over uses.
    """
    if db is None:
        return VOICE_OVER_DUCK_DEPTH
    return 1.0 - gain_of_db(db)


@dataclass(frozen=True, slots=True)
class ProgramLevels:
    """How loud a queue's programs are mixed in, and how deeply its music ducks under them."""

    # gain applied to the program's own audio, before it is summed over the music
    voice_db: float = VOICE_LEVEL_MAX_DB
    # level the music goes down to under a program; None keeps the global ducking depth
    duck_db: float | None = None

    @property
    def voice_gain(self) -> float:
        """Return the linear gain the program's audio is mixed in with."""
        return gain_of_db(self.voice_db)

    @property
    def duck_depth(self) -> float:
        """Return the share of the music the ducking pulls down."""
        return duck_depth_of_db(self.duck_db)


def checked_levels(voice_db: Any = VOICE_LEVEL_MAX_DB, duck_db: Any = None) -> ProgramLevels:
    """
    Validate the levels a queue asked to mix its programs at.

    :param voice_db: Gain of the program's audio in decibels, 0 to -9 dB.
    :param duck_db: Level the music goes down to under a program, 0 to -12 dB, or None for
        the global ducking depth.
    :raises ValueError: When a value is not a number or outside the range it may take.
    """
    return ProgramLevels(
        voice_db=checked_db(voice_db, "voice_db", VOICE_LEVEL_MIN_DB, VOICE_LEVEL_MAX_DB),
        duck_db=(
            None if duck_db is None else checked_db(duck_db, "duck_db", DUCK_MIN_DB, DUCK_MAX_DB)
        ),
    )


def checked_db(value: Any, field: str, minimum: float, maximum: float) -> float:
    """
    Return one decibel value inside its range, refusing anything else.

    Every setting that takes a level is validated here, so a queue's programs and the queue
    DJ that mixes them refuse the same values with the same wording; a caller with an error
    type of its own translates this one.

    :param value: The value a caller asked for.
    :param field: Name of the setting, for the error.
    :param minimum: Lowest value it may take.
    :param maximum: Highest value it may take.
    :raises ValueError: When the value is not a number of decibels or outside its range.
    """
    if isinstance(value, bool):
        # one exception type for every invalid level, so a caller has one thing to catch
        raise ValueError(f"{field} must be a number of decibels")  # noqa: TRY004
    try:
        number = float(value)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{field} must be a number of decibels") from err
    if not minimum <= number <= maximum:
        raise ValueError(f"{field} must be between {minimum:g} and {maximum:g} dB")
    return number

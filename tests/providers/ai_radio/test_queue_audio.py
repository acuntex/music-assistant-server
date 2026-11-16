"""The voice level and ducking depth a queue's AI DJ mixes its program at."""

from __future__ import annotations

from typing import Any

import pytest
from music_assistant_models.errors import InvalidDataError

from music_assistant.providers.ai_radio.queue_audio import (
    QueueAudioLevels,
    levels_from_stored,
    normalize_duck_db,
    normalize_voice_level_db,
)


def test_a_stored_entry_without_levels_reads_the_defaults() -> None:
    """A queue DJ stored before the settings existed speaks as it always did."""
    levels = levels_from_stored({"host_id": "rick", "station_id": ""})

    assert levels == QueueAudioLevels(voice_level_db=0.0, duck_db=None)


def test_no_entry_at_all_reads_the_defaults() -> None:
    """A queue whose entry is not even a mapping still yields a usable level."""
    assert levels_from_stored(None) == QueueAudioLevels()
    assert levels_from_stored("rick") == QueueAudioLevels()


def test_stored_levels_come_back_as_they_were_written() -> None:
    """Both settings survive a round trip through the queue DJ file."""
    levels = levels_from_stored({"voice_level_db": -4.5, "duck_db": -12})

    assert levels.voice_level_db == -4.5
    assert levels.duck_db == -12.0


def test_an_all_zero_entry_keeps_an_explicit_duck_of_zero() -> None:
    """0 dB is a setting, not a missing value: it leaves the music unducked."""
    levels = levels_from_stored({"voice_level_db": 0, "duck_db": 0})

    assert levels.voice_level_db == 0.0
    assert levels.duck_db == 0.0


@pytest.mark.parametrize(
    "entry",
    [
        {"voice_level_db": -20},
        {"voice_level_db": "loud"},
        {"duck_db": -20},
        {"duck_db": "deep"},
        {"voice_level_db": True},
    ],
)
def test_a_broken_stored_value_falls_back_instead_of_dropping_the_dj(
    entry: dict[str, Any],
) -> None:
    """A value written by a version that meant something else never breaks the DJ."""
    assert levels_from_stored(entry) == QueueAudioLevels()


@pytest.mark.parametrize("value", [None, 0, -9, -4.5, 0.0, "-6"])
def test_a_voice_level_inside_its_range_is_taken(value: Any) -> None:
    """0 to -9 dB are valid settings; nothing at all means the default of 0 dB."""
    level = normalize_voice_level_db(value)

    assert level == (0.0 if value is None else pytest.approx(float(value)))


@pytest.mark.parametrize("value", [-9.1, 0.5, 3, "loud", True, [], {}])
def test_a_voice_level_outside_its_range_is_refused(value: Any) -> None:
    """A boost, a value below the floor or a non-number is rejected, not clamped."""
    with pytest.raises(InvalidDataError, match="voice_level_db"):
        normalize_voice_level_db(value)


@pytest.mark.parametrize("value", [None, 0, -12, -6.5, 0.0, "-12"])
def test_a_ducking_depth_inside_its_range_is_taken(value: Any) -> None:
    """0 to -12 dB are valid; nothing at all inherits the depth the whole server uses."""
    depth = normalize_duck_db(value)

    assert depth is None if value is None else depth == pytest.approx(float(value))


@pytest.mark.parametrize("value", [-12.1, 0.5, 3, "deep", True, [], {}])
def test_a_ducking_depth_outside_its_range_is_refused(value: Any) -> None:
    """The music may be ducked 12 dB at most and never pushed up."""
    with pytest.raises(InvalidDataError, match="duck_db"):
        normalize_duck_db(value)

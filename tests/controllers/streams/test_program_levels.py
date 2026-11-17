"""The voice level and ducking depth a queue mixes its programs at."""

from __future__ import annotations

import struct
from typing import Any

import pytest

from music_assistant.controllers.streams.program_levels import (
    DUCK_MIN_DB,
    VOICE_LEVEL_MIN_DB,
    ProgramLevels,
    checked_levels,
)
from music_assistant.controllers.streams.smart_fades.filters import VOICE_OVER_DUCK_DEPTH
from tests.controllers.streams.program_bus_helpers import (
    QUEUE_ID,
    SECOND,
    bare_streams_controller,
    music,
    register_bus,
)

MUSIC_LEVEL = 100
VOICE_LEVEL = 8
VOICE_PCM = struct.pack("<hh", VOICE_LEVEL, VOICE_LEVEL) * SECOND


def _levels(pcm: bytes) -> list[int]:
    """Return the first channel's samples of a PCM chunk."""
    return list(struct.unpack(f"<{len(pcm) // 2}h", pcm))[::2]


def _queue(voice_db: float = 0.0, duck_db: float | None = None) -> tuple[Any, Any]:
    """Return a live bus of the queue and the controller that published it."""
    controller = bare_streams_controller()
    program_bus = register_bus(controller)
    controller.set_program_levels(QUEUE_ID, voice_db=voice_db, duck_db=duck_db)
    return controller, program_bus


async def _under_a_program(program_bus: Any, *, seconds: float = 1.0) -> list[int]:
    """Return the music one second into a program, after the ducking ramp completed."""
    program_bus.schedule(VOICE_PCM, SECOND, label="test")
    await program_bus.feed(music(1.0, MUSIC_LEVEL))
    return _levels(await program_bus.feed(music(seconds, MUSIC_LEVEL)))


def _ducked_music(db: float) -> int:
    """Return the music's level at the given ducking depth, as the mix rounds it."""
    return int(MUSIC_LEVEL * 10 ** (db / 20))


def _quieter_voice(db: float) -> int:
    """Return the voice's level at the given gain, as the 16-bit mix truncates it."""
    return int(VOICE_LEVEL * 10 ** (db / 20))


async def test_a_queue_without_levels_mixes_exactly_as_a_bus_always_did() -> None:
    """A queue that never set a level produces the same bytes a bus without levels did."""
    plain = register_bus(bare_streams_controller())
    _controller, with_levels = _queue()
    plain.schedule(VOICE_PCM, SECOND, label="test")
    with_levels.schedule(VOICE_PCM, SECOND, label="test")

    identical = [
        await plain.feed(music(1.0)) == await with_levels.feed(music(1.0)),
        await plain.feed(music(1.0)) == await with_levels.feed(music(1.0)),
        await plain.feed(music(1.0)) == await with_levels.feed(music(1.0)),
    ]

    assert identical == [True, True, True]
    # and that is today's mix: the global duck with the program summed on top
    assert (await _under_a_program(register_bus(bare_streams_controller())))[-1] == int(
        MUSIC_LEVEL * (1 - VOICE_OVER_DUCK_DEPTH)
    ) + VOICE_LEVEL


async def test_a_lower_voice_level_leaves_the_music_where_it_is() -> None:
    """Attenuating the program takes the voice down and does not touch the ducked music."""
    _controller, program_bus = _queue(voice_db=-6.0)

    under = await _under_a_program(program_bus)

    # the music sits exactly where the global duck puts it, only the voice is quieter
    assert under[-1] == int(MUSIC_LEVEL * (1 - VOICE_OVER_DUCK_DEPTH)) + _quieter_voice(-6.0)


async def test_the_voice_level_of_a_queue_shows_up_as_a_gain_factor() -> None:
    """A 6 dB lower voice arrives at about half the level of the default one."""
    _controller, program_bus = _queue(voice_db=-6.0)
    under = await _under_a_program(program_bus)
    ducked_music = int(MUSIC_LEVEL * (1 - VOICE_OVER_DUCK_DEPTH))

    # the 16-bit mix rounds the voice to whole samples, so the factor is checked with a bound
    assert (under[-1] - ducked_music) / VOICE_LEVEL == pytest.approx(10 ** (-6 / 20), abs=0.07)


async def test_the_ducking_depth_of_a_queue_sets_its_envelope() -> None:
    """The queue's own ducking depth decides how far the music goes down under a program."""
    _controller, program_bus = _queue(duck_db=DUCK_MIN_DB)

    under = await _under_a_program(program_bus)

    assert under[-1] == _ducked_music(DUCK_MIN_DB) + VOICE_LEVEL
    # 12 dB down is the deepest duck, so the music is quieter than the global 8 dB one leaves it
    assert under[-1] < int(MUSIC_LEVEL * (1 - VOICE_OVER_DUCK_DEPTH)) + VOICE_LEVEL


async def test_a_queue_with_no_duck_of_its_own_keeps_the_global_depth() -> None:
    """A queue that only set its voice level ducks exactly as the whole server does."""
    _controller, program_bus = _queue(voice_db=-3.0, duck_db=None)

    under = await _under_a_program(program_bus)

    assert under[-1] == int(MUSIC_LEVEL * (1 - VOICE_OVER_DUCK_DEPTH)) + _quieter_voice(-3.0)


async def test_a_level_of_zero_ducks_nothing_at_all() -> None:
    """A ducking depth of 0 dB leaves the music at its own level under the voice."""
    _controller, program_bus = _queue(duck_db=0.0)

    assert (await _under_a_program(program_bus))[-1] == MUSIC_LEVEL + VOICE_LEVEL


async def test_the_levels_of_a_queue_reach_the_session_that_opens_next() -> None:
    """A flow restart builds a new bus, and it mixes at the level the queue already had."""
    controller = bare_streams_controller()
    first = register_bus(controller)
    controller.set_program_levels(QUEUE_ID, voice_db=-6.0, duck_db=DUCK_MIN_DB)
    controller.release_program_bus(QUEUE_ID, first)

    second = register_bus(controller)

    assert second.duck_depth == pytest.approx(1 - 10 ** (DUCK_MIN_DB / 20))
    assert (await _under_a_program(second))[-1] == _ducked_music(DUCK_MIN_DB) + _quieter_voice(-6.0)


async def test_a_level_change_applies_to_the_bus_that_runs() -> None:
    """Setting a level while the session streams takes effect on the live bus at once."""
    controller, program_bus = _queue()

    controller.set_program_levels(QUEUE_ID, duck_db=DUCK_MIN_DB)

    assert (await _under_a_program(program_bus))[-1] == _ducked_music(DUCK_MIN_DB) + VOICE_LEVEL


def test_the_levels_of_one_queue_do_not_touch_another() -> None:
    """A level belongs to the queue it was set for."""
    controller = bare_streams_controller()
    first = register_bus(controller, queue_id=QUEUE_ID)
    second = register_bus(controller, queue_id="queue-2")
    controller.set_program_levels(QUEUE_ID, voice_db=-6.0)

    assert first.voice_gain == pytest.approx(10 ** (-6 / 20))
    assert second.voice_gain == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("voice_db", "duck_db"),
    [
        (0.0, None),
        (VOICE_LEVEL_MIN_DB, DUCK_MIN_DB),
        (-4.5, -6.0),
        (0, 0),
    ],
)
def test_the_levels_inside_their_ranges_are_taken(voice_db: Any, duck_db: Any) -> None:
    """The bounds of both ranges are valid settings, and so is anything between them."""
    levels = checked_levels(voice_db=voice_db, duck_db=duck_db)

    assert levels.voice_db == pytest.approx(float(voice_db))
    assert levels.duck_db == pytest.approx(None if duck_db is None else float(duck_db))


@pytest.mark.parametrize("voice_db", [-9.1, 0.5, 3, None, "loud", True, [], {}])
def test_a_voice_level_outside_its_range_is_refused(voice_db: Any) -> None:
    """A boost, a value below the floor or a non-number is not a setting."""
    with pytest.raises(ValueError, match="voice_db"):
        checked_levels(voice_db=voice_db)


@pytest.mark.parametrize("duck_db", [-12.5, 0.5, 3, "deep", True, [], {}])
def test_a_ducking_depth_outside_its_range_is_refused(duck_db: Any) -> None:
    """The ducking depth may lower the music 12 dB at most and never raise it."""
    with pytest.raises(ValueError, match="duck_db"):
        checked_levels(duck_db=duck_db)


def test_the_default_levels_are_a_gain_of_one_and_the_global_duck() -> None:
    """The defaults are exactly what a bus without a queue setting mixes."""
    levels = ProgramLevels()

    assert levels.voice_gain == pytest.approx(1.0)
    assert levels.duck_depth == pytest.approx(VOICE_OVER_DUCK_DEPTH)

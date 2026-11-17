"""Tests for the program bus access: an overlay mixed into a live flow session."""

from __future__ import annotations

import struct
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from music_assistant.controllers.streams.audio import FlowProgramBus
from music_assistant.controllers.streams.constants import (
    CONF_STREAM_PROGRAM_BUS,
    LEAD_MIN_LEAD_SECONDS,
    MAX_BUFFERED_LEAD_SECONDS,
    VOICE_OVER_RAMP,
)
from music_assistant.controllers.streams.program_schedule import ProgramKind, ScheduleReason
from music_assistant.controllers.streams.smart_fades.filters import VOICE_OVER_DUCK_DEPTH
from tests.controllers.streams.program_bus_helpers import (
    bare_streams_controller,
    hear,
    player_at,
)
from tests.controllers.streams.test_crossfade_transition import TEST_PCM_FORMAT

SAMPLE_SIZE = TEST_PCM_FORMAT.pcm_sample_size
SECOND = TEST_PCM_FORMAT.sample_rate
MUSIC_LEVEL = 100
VOICE_LEVEL = 8
# a flow session that already produced this much audio anchors an overlay at 5 s
PRODUCED_BEFORE = 3
OVERLAY_START = 5.0
OVERLAY_SECONDS = 3
ITEM_SECONDS = 30
# a player holding 1.5 s of the session's audio, so the listener is at the produced end
# minus that buffer
BUFFER_SECONDS = 1.5
AUDIBLE_BEFORE = PRODUCED_BEFORE - BUFFER_SECONDS
NO_POSITION_LINE = (
    "PROGRAM-BUS no audible position for queue %s - %s is placed %ss after the produced frames"
)


def _controller(*, audible: float | None = None) -> Any:
    """
    Return a streams controller with nothing registered yet.

    :param audible: Second of the stream the queue's player has heard, or None for a player
        that reports no position.
    """
    return bare_streams_controller(audible=audible)


def _register(controller: Any, queue_id: str = "queue-1") -> tuple[FlowProgramBus, MagicMock]:
    """Build a live flow session's bus and publish it for the queue."""
    logger = MagicMock()
    program_bus = FlowProgramBus(TEST_PCM_FORMAT, cast("Any", logger))
    controller.register_program_bus(queue_id, program_bus)
    return program_bus, logger


def _scheduled_anchor(logger: MagicMock) -> int:
    """Return the frame the bus logged the last scheduled program at."""
    for call in reversed(logger.debug.call_args_list):
        if call.args and call.args[0] == "PROGRAM-BUS schedule label=%s anchor=%d clamp=%d":
            return cast("int", call.args[2])
    raise AssertionError("the bus did not log a scheduled program")


def _fallback_lines(logger: MagicMock) -> int:
    """Return how often the bus reported placing a program without a position to go by."""
    return sum(
        1 for call in logger.debug.call_args_list if call.args and call.args[0] == NO_POSITION_LINE
    )


def _music(seconds: float, level: int = MUSIC_LEVEL) -> bytes:
    """Build music at the given level for the given number of seconds."""
    return struct.pack("<hh", level, level) * int(seconds * SECOND)


def _voice(seconds: float, level: int = VOICE_LEVEL) -> bytes:
    """Build a voice at the given level for the given number of seconds."""
    return struct.pack("<hh", level, level) * int(seconds * SECOND)


def _levels(pcm: bytes) -> list[int]:
    """Return the first channel's samples of a PCM chunk."""
    return list(struct.unpack(f"<{len(pcm) // 2}h", pcm))[::2]


def test_scheduling_without_a_live_flow_stream_reports_false() -> None:
    """A queue that is not streaming has nothing to mix an overlay into."""
    controller = _controller()

    scheduled = controller.schedule_overlay("queue-1", _voice(1.0), label="announcement")

    assert scheduled.scheduled is False
    # the caller learns why, and the request never raises
    controller.logger.info.assert_called_once()
    assert "queue-1" in controller.logger.info.call_args.args


def test_a_delayed_overlay_without_a_live_flow_stream_reports_false() -> None:
    """A queue with nothing to mix into reports false for a delayed program too."""
    controller = _controller()

    scheduled = controller.schedule_overlay(
        "queue-1", _voice(1.0), label="announcement", start_in_seconds=5.0
    )

    assert scheduled.scheduled is False


async def test_an_overlay_without_a_delay_is_mixed_exactly_as_before() -> None:
    """Leaving the delay out keeps the anchor and the mixed audio byte for byte."""
    undelayed = _controller()
    undelayed_bus, undelayed_logger = _register(undelayed)
    await undelayed_bus.feed(_music(PRODUCED_BEFORE))
    zero_delay = _controller()
    zero_delay_bus, zero_delay_logger = _register(zero_delay)
    await zero_delay_bus.feed(_music(PRODUCED_BEFORE))

    assert undelayed.schedule_overlay(
        "queue-1", _voice(OVERLAY_SECONDS), label="announcement"
    ).scheduled
    assert zero_delay.schedule_overlay(
        "queue-1", _voice(OVERLAY_SECONDS), label="announcement", start_in_seconds=0.0
    ).scheduled

    assert _scheduled_anchor(zero_delay_logger) == _scheduled_anchor(undelayed_logger)
    assert await zero_delay_bus.feed(_music(2.0)) == await undelayed_bus.feed(_music(2.0))


async def test_a_delay_moves_the_anchor_one_second_of_frames_per_second() -> None:
    """A delay of five seconds anchors the program five seconds after the listener."""
    controller = _controller(audible=0.0)
    program_bus, logger = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))

    scheduled = controller.schedule_overlay(
        "queue-1", _voice(OVERLAY_SECONDS), label="announcement", start_in_seconds=5.0
    )

    assert scheduled.scheduled is True
    assert _scheduled_anchor(logger) == 5 * SECOND


async def test_a_delay_is_counted_from_what_the_listener_hears() -> None:
    """
    A break timed for a boundary starts that many seconds after the listener, not the bus.

    The session may be several seconds ahead of the player, and a break planned for a
    boundary still to come is placed relative to what is heard, or it lands that much
    after the boundary it was written for.
    """
    controller = _controller(audible=AUDIBLE_BEFORE)
    program_bus, logger = _register(controller)
    # the session produced five seconds more than the player has heard
    await program_bus.feed(_music(AUDIBLE_BEFORE + 5))
    break_seconds = 12.0

    scheduled = controller.schedule_overlay(
        "queue-1",
        _voice(OVERLAY_SECONDS),
        label="announcement",
        start_in_seconds=break_seconds,
    )

    assert scheduled.scheduled is True
    # the boundary is twelve seconds from the listener, whatever the bus already produced
    assert _scheduled_anchor(logger) == int((AUDIBLE_BEFORE + break_seconds) * SECOND)


async def test_a_delay_shorter_than_the_lead_still_gets_the_programs_lead() -> None:
    """A boundary closer than the lead the bus needs is placed at that lead, not earlier."""
    controller = _controller(audible=AUDIBLE_BEFORE)
    program_bus, logger = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))

    assert controller.schedule_overlay(
        "queue-1", _voice(OVERLAY_SECONDS), label="announcement", start_in_seconds=0.5
    ).scheduled

    assert _scheduled_anchor(logger) == int((AUDIBLE_BEFORE + LEAD_MIN_LEAD_SECONDS) * SECOND)


async def test_a_negative_delay_never_pulls_a_program_behind_the_lead() -> None:
    """A delay before now leaves the placement the program gets without one."""
    controller = _controller(audible=AUDIBLE_BEFORE)
    program_bus, logger = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))

    assert controller.schedule_overlay(
        "queue-1",
        _voice(OVERLAY_SECONDS),
        label="announcement",
        start_in_seconds=-30.0,
        kind=ProgramKind.ANNOUNCEMENT,
    ).scheduled

    assert _scheduled_anchor(logger) == int((AUDIBLE_BEFORE + LEAD_MIN_LEAD_SECONDS) * SECOND)


async def test_a_fractional_delay_is_counted_in_whole_frames() -> None:
    """A delay of less than a second lands on the frame its seconds round to."""
    controller = _controller(audible=0.0)
    program_bus, logger = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))

    assert controller.schedule_overlay(
        "queue-1", _voice(OVERLAY_SECONDS), label="announcement", start_in_seconds=5.4
    ).scheduled

    assert _scheduled_anchor(logger) == round(5.4 * SECOND)


async def test_a_negative_delay_never_anchors_in_the_past() -> None:
    """A delay before now is clamped to the placement the program would get without one."""
    controller = _controller()
    program_bus, logger = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))

    assert controller.schedule_overlay(
        "queue-1", _voice(OVERLAY_SECONDS), label="announcement", start_in_seconds=-30.0
    ).scheduled

    assert _scheduled_anchor(logger) == int(OVERLAY_START * SECOND)
    assert program_bus.produced_frames == PRODUCED_BEFORE * SECOND


async def test_a_delay_past_the_produced_end_waits_for_the_music_to_get_there() -> None:
    """A delay longer than the buffer lead is accepted and heard at that later frame."""
    controller = _controller(audible=0.0)
    program_bus, logger = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))
    delay = MAX_BUFFERED_LEAD_SECONDS + 60

    scheduled = controller.schedule_overlay(
        "queue-1", _voice(OVERLAY_SECONDS), label="announcement", start_in_seconds=delay
    )

    assert scheduled.scheduled is True
    # the bus keeps the program for the frame it was anchored at, however far ahead it is
    assert _scheduled_anchor(logger) == int(delay * SECOND)
    assert await program_bus.feed(_music(1.0)) == _music(1.0)


async def test_an_overlay_is_placed_a_lead_after_what_the_stream_produced() -> None:
    """Without a position to go by the overlay lands a lead ahead of the produced audio."""
    controller = _controller()
    program_bus, logger = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))

    scheduled = controller.schedule_overlay(
        "queue-1", _voice(OVERLAY_SECONDS), label="announcement"
    )

    assert scheduled.scheduled is True
    assert program_bus.active
    # the anchor is the current frame plus the minimum lead the bus needs, not the frame
    # the session happens to be at
    assert _scheduled_anchor(logger) == int(OVERLAY_START * SECOND)
    assert program_bus.produced_frames == PRODUCED_BEFORE * SECOND


async def test_an_overlay_is_placed_a_lead_after_what_is_audible() -> None:
    """
    With 1.5 s in the player's buffer the overlay lands a lead after the listener.

    Half of the lead the bus needs is already in the player's buffer, so the program is
    placed half a second past the produced end instead of two.
    """
    controller = _controller(audible=AUDIBLE_BEFORE)
    program_bus, logger = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))

    scheduled = controller.schedule_overlay(
        "queue-1", _voice(OVERLAY_SECONDS), label="announcement", kind=ProgramKind.ANNOUNCEMENT
    )

    assert scheduled.scheduled is True
    assert program_bus.active
    anchor = _scheduled_anchor(logger)
    # the anchor, less the minimum lead, is the audible frame within a frame
    assert (anchor - int(LEAD_MIN_LEAD_SECONDS * SECOND)) / SECOND == pytest.approx(
        AUDIBLE_BEFORE, abs=1 / SECOND
    )
    # the buffer already carries 1.5 s of the lead the bus needs, so the program lands half a
    # second past the produced end instead of two
    assert anchor == int((PRODUCED_BEFORE + LEAD_MIN_LEAD_SECONDS - BUFFER_SECONDS) * SECOND)
    assert anchor < int(OVERLAY_START * SECOND)
    # the session produced nothing more for it
    assert program_bus.produced_frames == PRODUCED_BEFORE * SECOND
    # a position to go by needs no fallback
    assert _fallback_lines(controller.logger) == 0


async def test_a_buffer_deeper_than_the_lead_keeps_the_overlay_out_of_produced_audio() -> None:
    """A player more than a lead behind leaves the program at the produced end, not behind it."""
    controller = _controller(audible=0.5)
    program_bus, logger = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))

    scheduled = controller.schedule_overlay(
        "queue-1", _voice(OVERLAY_SECONDS), label="announcement", kind=ProgramKind.ANNOUNCEMENT
    )

    assert scheduled.scheduled is True
    # a lead after the audible frame was already produced, so the program starts where the
    # session is: it is never mixed into frames that went out already
    assert _scheduled_anchor(logger) == PRODUCED_BEFORE * SECOND


def test_a_position_further_back_than_any_buffer_is_not_this_streams_clock() -> None:
    """A position of another stream (an item's own) is not used to place the program."""
    controller = _controller(audible=1.0)
    program_bus, logger = _register(controller)
    produced = int(MAX_BUFFERED_LEAD_SECONDS * SECOND) + 2 * SECOND
    program_bus.produced_frames = produced

    scheduled = controller.schedule_overlay(
        "queue-1", _voice(OVERLAY_SECONDS), label="announcement"
    )

    assert scheduled.scheduled is True
    assert _scheduled_anchor(logger) == produced + int(LEAD_MIN_LEAD_SECONDS * SECOND)
    assert _fallback_lines(controller.logger) == 1


async def test_an_unmeasurable_position_is_reported_once() -> None:
    """Without a position to go by every program keeps today's lead, and that is logged once."""
    controller = _controller()
    program_bus, logger = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))

    assert controller.schedule_overlay("queue-1", _voice(1.0), label="announcement").scheduled
    assert _scheduled_anchor(logger) == int(OVERLAY_START * SECOND)

    assert controller.schedule_overlay("queue-1", _voice(1.0), label="announcement").scheduled

    # the second program is placed the same way without reporting the fallback again
    assert _fallback_lines(controller.logger) == 1
    assert _scheduled_anchor(logger) > int(OVERLAY_START * SECOND)


async def test_a_thirty_second_item_keeps_its_length_and_is_ducked_in_the_overlay_window() -> None:
    """An overlay at +5 s ducks the item for its length and the item stays whole afterwards."""
    controller = _controller()
    program_bus, _logger = _register(controller)
    output = bytearray(await program_bus.feed(_music(PRODUCED_BEFORE)))
    assert controller.schedule_overlay(
        "queue-1", _voice(OVERLAY_SECONDS), label="announcement"
    ).scheduled

    for _ in range(ITEM_SECONDS - PRODUCED_BEFORE):
        output.extend(await program_bus.feed(_music(1.0)))

    # the item's own length is untouched
    assert len(output) == ITEM_SECONDS * SAMPLE_SIZE
    levels = _levels(bytes(output))
    ducked = int(MUSIC_LEVEL * (1 - VOICE_OVER_DUCK_DEPTH)) + VOICE_LEVEL
    # untouched right up to the anchor
    assert levels[int((OVERLAY_START - 0.01) * SECOND)] == MUSIC_LEVEL
    # the music is pulled down over the duck-in ramp, then held under the voice
    assert levels[int((OVERLAY_START + 0.01) * SECOND)] > ducked
    assert levels[int((OVERLAY_START + VOICE_OVER_RAMP + 0.01) * SECOND)] == ducked
    assert levels[int((OVERLAY_START + OVERLAY_SECONDS - 0.01) * SECOND)] == ducked
    # the ramp back is over well before the item ends
    back = int((OVERLAY_START + OVERLAY_SECONDS + VOICE_OVER_RAMP + 0.01) * SECOND)
    assert levels[back] == MUSIC_LEVEL
    assert levels[-1] == MUSIC_LEVEL
    assert not program_bus.active


async def test_cancelling_drops_the_overlay_at_once() -> None:
    """Cancelling mid-item removes the overlay and leaves the item's audio alone."""
    controller = _controller()
    program_bus, _logger = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))
    assert controller.schedule_overlay(
        "queue-1", _voice(OVERLAY_SECONDS), label="announcement"
    ).scheduled
    await program_bus.feed(_music(OVERLAY_START))

    controller.cancel_overlays("queue-1", reason="next")

    assert not program_bus.active
    assert await program_bus.feed(_music(1.0)) == _music(1.0)


def test_cancelling_without_a_live_flow_stream_does_nothing() -> None:
    """A cancel that races the end of the session is a no-op."""
    controller = _controller()

    controller.cancel_overlays("queue-1", reason="next")

    controller.logger.info.assert_not_called()


def test_a_superseded_session_does_not_withdraw_the_new_bus() -> None:
    """The session that took over keeps its bus when the old one ends."""
    controller = _controller()
    old_bus, _old_logger = _register(controller)
    new_bus, _new_logger = _register(controller)

    controller.release_program_bus("queue-1", old_bus)
    still_active = controller.program_bus_active("queue-1")
    registered_format = controller.program_bus_format("queue-1")
    controller.release_program_bus("queue-1", new_bus)

    assert still_active
    assert registered_format == new_bus.pcm_format
    assert not controller.program_bus_active("queue-1")


def test_no_flow_stream_reports_no_pcm_format() -> None:
    """Without a live session there is no format to render an overlay in."""
    controller = _controller()

    assert controller.program_bus_format("queue-1") is None


def test_the_program_bus_switch_is_the_streams_config_setting() -> None:
    """Whether sessions may mix programs is the experimental streams setting."""
    controller = _controller()
    controller.config = MagicMock()
    controller.config.get_value.return_value = True

    assert controller.program_bus_enabled() is True
    controller.config.get_value.assert_called_once_with(CONF_STREAM_PROGRAM_BUS, False)


async def test_an_overlay_without_a_position_and_without_a_delay_gets_the_lead_after_produced() -> (
    None
):
    """With no heard position the legacy fallback is produced plus the lead, unchanged."""
    controller = _controller()
    program_bus, logger = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))

    assert controller.schedule_overlay(
        "queue-1", _voice(OVERLAY_SECONDS), label="announcement"
    ).scheduled

    assert _scheduled_anchor(logger) == int(PRODUCED_BEFORE * SECOND) + int(
        LEAD_MIN_LEAD_SECONDS * SECOND
    )


async def test_an_undelayed_overlay_uses_the_later_of_produced_and_audible_lead() -> None:
    """The anchor is max(produced, audible + lead) for a program without a delay."""
    # the listener is deep in the session, so the lead after it is already produced
    deep = _controller(audible=0.5)
    deep_bus, deep_logger = _register(deep)
    await deep_bus.feed(_music(PRODUCED_BEFORE))
    # the listener is a lead behind the produced end, so the anchor is still to come
    near = _controller(audible=AUDIBLE_BEFORE)
    near_bus, near_logger = _register(near)
    await near_bus.feed(_music(PRODUCED_BEFORE))

    assert deep.schedule_overlay(
        "queue-1", _voice(OVERLAY_SECONDS), label="announcement", kind=ProgramKind.ANNOUNCEMENT
    ).scheduled
    assert near.schedule_overlay(
        "queue-1", _voice(OVERLAY_SECONDS), label="announcement", kind=ProgramKind.ANNOUNCEMENT
    ).scheduled

    assert _scheduled_anchor(deep_logger) == int(PRODUCED_BEFORE * SECOND)
    assert _scheduled_anchor(near_logger) == int((AUDIBLE_BEFORE + LEAD_MIN_LEAD_SECONDS) * SECOND)


async def test_cancelling_drops_every_program_of_the_queue_at_once() -> None:
    """The collective cancel drops all programs still to be heard, not just the last one."""
    controller = _controller(audible=0.0)
    program_bus, _logger = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))
    assert controller.schedule_overlay(
        "queue-1", _voice(OVERLAY_SECONDS), label="first", start_in_seconds=5.0
    ).scheduled
    assert controller.schedule_overlay(
        "queue-1", _voice(OVERLAY_SECONDS), label="second", start_in_seconds=9.0
    ).scheduled

    controller.cancel_overlays("queue-1", reason="next")

    assert program_bus.active is False
    assert program_bus.scheduled_voice_frames("first") is None
    assert program_bus.scheduled_voice_frames("second") is None


async def test_the_missing_position_is_reported_again_after_a_position_was_known() -> None:
    """A position that became known resets the once-per-queue fallback report."""
    controller = _controller()
    program_bus, _logger = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))
    assert controller.schedule_overlay("queue-1", _voice(1.0), label="announcement").scheduled
    assert _fallback_lines(controller.logger) == 1

    controller.mass.players.get_player.return_value = player_at(AUDIBLE_BEFORE)
    assert controller.schedule_overlay(
        "queue-1", _voice(1.0), label="announcement", kind=ProgramKind.ANNOUNCEMENT
    ).scheduled

    controller.mass.players.get_player.return_value = player_at(None)
    assert controller.schedule_overlay("queue-1", _voice(1.0), label="announcement").scheduled
    assert _fallback_lines(controller.logger) == 2


async def test_a_position_further_ahead_than_a_lead_is_not_this_streams_clock() -> None:
    """A player reporting past what the session produced is not a position of this stream."""
    controller = _controller(audible=0.0)
    program_bus, _logger = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))
    # the bus was replaced, and the player still reports where the previous one was
    hear(controller, PRODUCED_BEFORE + MAX_BUFFERED_LEAD_SECONDS + 5)

    assert controller._audible_frame("queue-1", program_bus) is None


async def test_a_boundary_ahead_needs_a_position_of_this_stream() -> None:
    """A term timed for a boundary is not placed on a clock that is not this session's."""
    controller = _controller(audible=0.0)
    program_bus, _logger = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))
    hear(controller, PRODUCED_BEFORE + MAX_BUFFERED_LEAD_SECONDS + 5)

    scheduled = controller.schedule_overlay(
        "queue-1", _voice(OVERLAY_SECONDS), label="term", start_in_seconds=12.0
    )

    assert scheduled.scheduled is False
    assert scheduled.reason is ScheduleReason.NO_AUDIBLE_POSITION


async def test_a_position_within_the_produced_frames_is_used_as_it_was() -> None:
    """A position the session has produced stays the frame the listener is at."""
    controller = _controller(audible=0.0)
    program_bus, _logger = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))
    hear(controller, AUDIBLE_BEFORE)

    assert controller._audible_frame("queue-1", program_bus) == int(AUDIBLE_BEFORE * SECOND)


async def test_a_position_a_hair_ahead_of_the_producer_still_counts() -> None:
    """A step-shaped report may overshoot a little: only a whole lead is another clock."""
    controller = _controller(audible=0.0)
    program_bus, _logger = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))
    hear(controller, PRODUCED_BEFORE + 0.5)

    assert controller._audible_frame("queue-1", program_bus) == program_bus.produced_frames


def test_a_position_further_back_than_a_lead_is_still_no_position() -> None:
    """The other direction keeps its guard: a player a whole lead behind says nothing."""
    controller = _controller(audible=0.0)
    program_bus, _logger = _register(controller)
    produced = int(MAX_BUFFERED_LEAD_SECONDS * SECOND) + 2 * SECOND
    program_bus.produced_frames = produced
    hear(controller, 1.0)

    assert controller._audible_frame("queue-1", program_bus) is None

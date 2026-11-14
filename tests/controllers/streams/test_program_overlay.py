"""Tests for the program bus access: an overlay mixed into a live flow session."""

from __future__ import annotations

import struct
from typing import Any, cast
from unittest.mock import MagicMock

from music_assistant.controllers.streams import StreamsController
from music_assistant.controllers.streams.audio import _FlowProgramBus
from music_assistant.controllers.streams.constants import VOICE_OVER_RAMP
from music_assistant.controllers.streams.smart_fades.filters import VOICE_OVER_DUCK_DEPTH
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


def _controller() -> Any:
    """Return a streams controller with nothing registered yet."""
    controller: Any = StreamsController.__new__(StreamsController)
    controller._program_buses = {}
    controller._program_bus_owners = {}
    controller.logger = MagicMock()
    return controller


def _register(controller: Any, queue_id: str = "queue-1") -> tuple[_FlowProgramBus, MagicMock]:
    """Build a live flow session's bus and publish it for the queue."""
    logger = MagicMock()
    program_bus = _FlowProgramBus(TEST_PCM_FORMAT, cast("Any", logger))
    controller.register_program_bus(queue_id, program_bus)
    return program_bus, logger


def _scheduled_anchor(logger: MagicMock) -> int:
    """Return the frame the bus logged the last scheduled program at."""
    for call in logger.debug.call_args_list:
        if call.args and call.args[0] == "PROGRAM-BUS schedule label=%s anchor=%d clamp=%d":
            return cast("int", call.args[2])
    raise AssertionError("the bus did not log a scheduled program")


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

    assert scheduled is False
    # the caller learns why, and the request never raises
    controller.logger.info.assert_called_once()
    assert "queue-1" in controller.logger.info.call_args.args


def test_an_overlay_is_placed_a_lead_after_what_the_stream_produced() -> None:
    """The overlay lands far enough ahead of the produced audio to reach the player in time."""
    controller = _controller()
    program_bus, logger = _register(controller)
    program_bus.feed(_music(PRODUCED_BEFORE))

    scheduled = controller.schedule_overlay(
        "queue-1", _voice(OVERLAY_SECONDS), label="announcement"
    )

    assert scheduled is True
    assert program_bus.active
    # the anchor is the current frame plus the minimum lead the bus needs, not the frame
    # the session happens to be at
    assert _scheduled_anchor(logger) == int(OVERLAY_START * SECOND)
    assert program_bus.produced_frames == PRODUCED_BEFORE * SECOND


def test_a_thirty_second_item_keeps_its_length_and_is_ducked_in_the_overlay_window() -> None:
    """An overlay at +5 s ducks the item for its length and the item stays whole afterwards."""
    controller = _controller()
    program_bus, _logger = _register(controller)
    output = bytearray(program_bus.feed(_music(PRODUCED_BEFORE)))
    assert controller.schedule_overlay("queue-1", _voice(OVERLAY_SECONDS), label="announcement")

    for _ in range(ITEM_SECONDS - PRODUCED_BEFORE):
        output.extend(program_bus.feed(_music(1.0)))

    # the item's own length is untouched
    assert len(output) == ITEM_SECONDS * SAMPLE_SIZE
    levels = _levels(bytes(output))
    ducked = int(MUSIC_LEVEL * (1 - VOICE_OVER_DUCK_DEPTH)) + VOICE_LEVEL
    # untouched right up to the anchor
    assert levels[int((OVERLAY_START - 0.01) * SECOND)] == MUSIC_LEVEL
    # ducked with the voice summed in, for the whole overlay
    assert levels[int((OVERLAY_START + 0.01) * SECOND)] == ducked
    assert levels[int((OVERLAY_START + OVERLAY_SECONDS - 0.01) * SECOND)] == ducked
    # the ramp back is over well before the item ends
    back = int((OVERLAY_START + OVERLAY_SECONDS + VOICE_OVER_RAMP + 0.01) * SECOND)
    assert levels[back] == MUSIC_LEVEL
    assert levels[-1] == MUSIC_LEVEL
    assert not program_bus.active


def test_cancelling_drops_the_overlay_at_once() -> None:
    """Cancelling mid-item removes the overlay and leaves the item's audio alone."""
    controller = _controller()
    program_bus, _logger = _register(controller)
    program_bus.feed(_music(PRODUCED_BEFORE))
    assert controller.schedule_overlay("queue-1", _voice(OVERLAY_SECONDS), label="announcement")
    program_bus.feed(_music(OVERLAY_START))

    controller.cancel_overlays("queue-1", reason="next")

    assert not program_bus.active
    assert program_bus.feed(_music(1.0)) == _music(1.0)


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


def test_the_program_bus_claim_holds_while_any_owner_requires_it() -> None:
    """Releasing one owner's claim leaves the bus required while another holds one."""
    controller = _controller()

    controller.set_program_bus_required("queue-1", "ai_radio", True)
    controller.set_program_bus_required("queue-1", "other_plugin", True)
    controller.set_program_bus_required("queue-1", "ai_radio", False)
    still_required = controller.program_bus_required("queue-1")
    controller.set_program_bus_required("queue-1", "other_plugin", False)

    assert still_required
    assert not controller.program_bus_required("queue-1")
    assert not controller.program_bus_required("queue-2")

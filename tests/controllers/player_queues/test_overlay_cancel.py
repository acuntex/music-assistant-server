"""Tests for what a move of the queue does to the programs mixed into its stream."""

from __future__ import annotations

import asyncio
import struct
import time
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from music_assistant_models.enums import PlaybackState
from music_assistant_models.player_queue import PlayerQueue
from music_assistant_models.queue_item import QueueItem

from music_assistant.controllers.player_queues import PlayerQueuesController
from music_assistant.controllers.player_queues.state import PlayerQueueData
from music_assistant.controllers.streams.audio import FlowProgramBus
from music_assistant.controllers.streams.program_schedule import (
    ProgramKind,
    ScheduleReason,
)
from music_assistant.providers.ai_radio.constants import OVERLAY_LABEL_INTRO
from tests.controllers.streams.program_bus_helpers import bare_streams_controller
from tests.controllers.streams.test_crossfade_transition import TEST_PCM_FORMAT

QUEUE_ID = "q1"
ANNOUNCEMENT = "announcement:one"

# the drop reasons the design fixes for a program the ledger ends, verbatim
DESIGN_DROP_REASONS = frozenset(
    {
        "too_late",
        "unknown_boundary",
        "no_duration",
        "skipped",
        "seeked",
        "flow_ended",
        "render_failed",
        "no_bus",
        "no_successor",
        "format",
        "bus_replaced",
    }
)
# what every move of the queue is expected to record, and where it came from
JUMP_ORIGINS = {
    "next": "skipped",
    "previous": "skipped",
    "skip": "seeked",
    "play_index": "skipped",
}


def _controller() -> tuple[PlayerQueuesController, PlayerQueue]:
    """Build a bare controller playing the first of two items."""
    ctrl = PlayerQueuesController.__new__(PlayerQueuesController)
    queue = PlayerQueue(queue_id=QUEUE_ID, active=True, display_name="Q1", available=True, items=2)
    items = [
        QueueItem(queue_id=QUEUE_ID, queue_item_id="item1", name="one", duration=200),
        QueueItem(queue_id=QUEUE_ID, queue_item_id="item2", name="two", duration=200),
    ]
    queue.current_index = 0
    queue.current_item = items[0]
    queue.state = PlaybackState.PLAYING
    queue.elapsed_time = 100
    queue.elapsed_time_last_updated = time.time()
    queue_data = PlayerQueueData(queue=queue)
    queue_data.items = items
    ctrl._queue_data = {QUEUE_ID: queue_data}
    ctrl.logger = MagicMock()
    ctrl.mass = MagicMock()
    ctrl.mass.players.get_player.return_value.offline_by_power_control = False
    ctrl.mass.players.get_player.return_value.extra_data = {}
    ctrl.mass.players._handle_cmd_stop = AsyncMock()
    ctrl.mass.players.get_group_and_player_lock = MagicMock()
    ctrl._managed_pool = MagicMock()
    ctrl.signal_update = Mock()  # type: ignore[method-assign]
    ctrl.on_player_update = Mock()  # type: ignore[method-assign]
    return ctrl, queue


def _jumps(ctrl: PlayerQueuesController) -> list[tuple[str, str, str]]:
    """Return the jumps the controller asked the streams controller to settle."""
    streams = cast("Any", ctrl.mass.streams)
    return [
        (call.args[0], call.args[1], call.kwargs["origin"])
        for call in streams.cancel_overlays_of_a_jump.call_args_list
    ]


def _with_streams(ctrl: PlayerQueuesController) -> Any:
    """Give the controller a real streams controller whose listener is at the start."""
    streams: Any = bare_streams_controller(audible=0.0)
    streams.mass = ctrl.mass
    ctrl.mass.streams = streams
    player = ctrl.mass.players.get_player.return_value
    player.state.playback_state = PlaybackState.PLAYING
    player.state.corrected_elapsed_time = 0.0
    return streams


async def _bus_with_music(streams: Any, seconds: float = 3.0) -> FlowProgramBus:
    """Publish a live flow session's bus that already produced some audio."""
    program_bus = FlowProgramBus(TEST_PCM_FORMAT, cast("Any", MagicMock()))
    streams.register_program_bus(QUEUE_ID, program_bus)
    await program_bus.feed(
        struct.pack("<hh", 100, 100) * TEST_PCM_FORMAT.sample_rate * int(seconds)
    )
    return program_bus


async def _run(ctrl: PlayerQueuesController, command: str) -> None:
    """Run one of the four queue commands that move playback."""
    if command == "next":
        ctrl._get_next_index = Mock(return_value=1)  # type: ignore[method-assign]
        await ctrl.next(QUEUE_ID)
        return
    if command == "previous":
        await ctrl.previous(QUEUE_ID)
        return
    if command == "skip":
        ctrl._apply_pending_skip = AsyncMock()  # type: ignore[method-assign]
        await ctrl.skip(QUEUE_ID, 10)
        return
    ctrl._play_index = AsyncMock()  # type: ignore[method-assign]
    await ctrl.play_index(QUEUE_ID, 1)


@pytest.mark.parametrize(("command", "reason"), JUMP_ORIGINS.items())
async def test_a_command_that_moves_the_queue_settles_the_jump(command: str, reason: str) -> None:
    """Each of the four commands hands the jump to the streams controller, naming itself."""
    ctrl, _queue = _controller()

    await _run(ctrl, command)

    assert _jumps(ctrl) == [(QUEUE_ID, reason, command)]


@pytest.mark.parametrize("command", JUMP_ORIGINS)
async def test_every_jump_records_a_reason_of_the_design_vocabulary(command: str) -> None:
    """What the ledger keeps is the vocabulary word, never the command that caused it."""
    ctrl, _queue = _controller()
    streams = _with_streams(ctrl)
    program_bus = await _bus_with_music(streams)
    assert streams.schedule_overlay(
        QUEUE_ID,
        struct.pack("<hh", 8, 8) * TEST_PCM_FORMAT.sample_rate,
        label="term",
        start_in_seconds=10.0,
    ).scheduled

    await _run(ctrl, command)

    assert program_bus.scheduled_voice_frames("term") is None
    states = [(state.label, state.state, state.reason) for state in streams.overlay_state(QUEUE_ID)]
    assert states == [("term", "dropped", JUMP_ORIGINS[command])]
    assert states[0][2] in DESIGN_DROP_REASONS


async def test_next_keeps_the_announcement_that_is_already_rendered() -> None:
    """A rendered announcement is heard over the stream that the next item keeps playing."""
    ctrl, _queue = _controller()
    streams = _with_streams(ctrl)
    program_bus = await _bus_with_music(streams)
    assert streams.schedule_overlay(
        QUEUE_ID,
        struct.pack("<hh", 8, 8) * TEST_PCM_FORMAT.sample_rate,
        label=ANNOUNCEMENT,
        kind=ProgramKind.ANNOUNCEMENT,
    ).scheduled

    await _run(ctrl, "next")

    assert program_bus.scheduled_voice_frames(ANNOUNCEMENT) is not None
    assert [(state.label, state.state) for state in streams.overlay_state(QUEUE_ID)] == [
        (ANNOUNCEMENT, "scheduled")
    ]


async def test_skip_leaves_no_overlay_or_task_behind() -> None:
    """Dropping a live overlay is synchronous: nothing of it stays scheduled."""
    ctrl, _queue = _controller()
    ctrl._apply_pending_skip = AsyncMock()  # type: ignore[method-assign]
    streams = _with_streams(ctrl)
    # the session produced nothing yet, so the program sits right at the listener
    program_bus = await _bus_with_music(streams, 0.0)
    speech = struct.pack("<hh", 8, 8) * TEST_PCM_FORMAT.sample_rate * 3
    assert streams.schedule_overlay(QUEUE_ID, speech, label=ANNOUNCEMENT).scheduled
    before = asyncio.all_tasks()

    await ctrl.skip(QUEUE_ID, 10)

    assert program_bus.active is False
    assert asyncio.all_tasks() == before


async def test_skip_drops_a_delayed_program_before_its_boundary() -> None:
    """A program waiting for a boundary that playback never reaches is dropped, not played late."""
    ctrl, _queue = _controller()
    ctrl._apply_pending_skip = AsyncMock()  # type: ignore[method-assign]
    streams = _with_streams(ctrl)
    program_bus = await _bus_with_music(streams)
    voice = struct.pack("<hh", 8, 8) * TEST_PCM_FORMAT.sample_rate
    assert streams.schedule_overlay(
        QUEUE_ID, voice, label=OVERLAY_LABEL_INTRO, start_in_seconds=10.0
    ).scheduled

    await ctrl.skip(QUEUE_ID, 10)

    assert program_bus.active is False
    # what the stream produces after the skip is untouched by the dropped program
    assert (
        await program_bus.feed(struct.pack("<hh", 100, 100) * 64)
        == struct.pack("<hh", 100, 100) * 64
    )


async def test_stop_leaves_the_overlay_to_the_ending_stream() -> None:
    """A stop ends the stream, which drops what it was still to play by itself."""
    ctrl, _queue = _controller()
    ctrl._end_session = Mock()  # type: ignore[method-assign]

    await ctrl._handle_stop(QUEUE_ID)

    assert _jumps(ctrl) == []
    ctrl.mass.streams.cancel_overlays.assert_not_called()


async def test_a_removed_player_does_not_keep_an_announcement_behind() -> None:
    """Removing a player ends its session instead of settling a jump, and keeps nothing back."""
    ctrl, _queue = _controller()
    ctrl._end_session = Mock()  # type: ignore[method-assign]

    ctrl.on_player_remove(QUEUE_ID, permanent=False)

    ctrl.mass.streams.cancel_overlays.assert_not_called()
    ctrl.mass.streams.cancel_overlays_of_a_jump.assert_not_called()
    # the queue is gone, so nothing is left that a later bus could speak for
    assert QUEUE_ID not in ctrl._queue_data


def test_the_reason_vocabulary_matches_the_design() -> None:
    """The two words a jump records are the design's, and nothing else is invented."""
    assert set(JUMP_ORIGINS.values()) <= DESIGN_DROP_REASONS
    assert ScheduleReason.SKIPPED in DESIGN_DROP_REASONS
    assert ScheduleReason.SEEKED in DESIGN_DROP_REASONS

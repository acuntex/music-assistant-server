"""Tests for dropping an announcement mixed into the stream when playback moves on."""

from __future__ import annotations

import asyncio
import struct
import time
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock

from music_assistant_models.enums import PlaybackState
from music_assistant_models.player_queue import PlayerQueue
from music_assistant_models.queue_item import QueueItem

from music_assistant.controllers.player_queues import PlayerQueuesController
from music_assistant.controllers.player_queues.state import PlayerQueueData
from music_assistant.controllers.streams import StreamsController
from music_assistant.controllers.streams.audio import _FlowProgramBus
from tests.controllers.streams.test_crossfade_transition import TEST_PCM_FORMAT

QUEUE_ID = "q1"


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
    ctrl.mass = MagicMock()
    ctrl.mass.players.get_player.return_value.offline_by_power_control = False
    ctrl.mass.players.get_player.return_value.extra_data = {}
    ctrl.mass.players._handle_cmd_stop = AsyncMock()
    ctrl.mass.players.get_group_and_player_lock = MagicMock()
    ctrl.signal_update = Mock()  # type: ignore[method-assign]
    ctrl.on_player_update = Mock()  # type: ignore[method-assign]
    return ctrl, queue


def _cancelled(ctrl: PlayerQueuesController) -> list[tuple[str, str]]:
    """Return the overlay cancels the controller asked for, with their reasons."""
    streams = cast("Any", ctrl.mass.streams)
    return [
        (call.args[0], call.kwargs["reason"]) for call in streams.cancel_overlays.call_args_list
    ]


async def test_next_drops_the_overlay() -> None:
    """Moving to the next item leaves no announcement behind on the stream."""
    ctrl, _queue = _controller()
    ctrl._get_next_index = Mock(return_value=1)  # type: ignore[method-assign]

    await ctrl.next(QUEUE_ID)

    assert _cancelled(ctrl) == [(QUEUE_ID, "next")]


async def test_previous_drops_the_overlay() -> None:
    """Going back restarts playback elsewhere, so the announcement is not continued."""
    ctrl, _queue = _controller()

    await ctrl.previous(QUEUE_ID)

    assert _cancelled(ctrl) == [(QUEUE_ID, "previous")]


async def test_skip_drops_the_overlay() -> None:
    """A jump within the item is no place for an announcement anchored elsewhere."""
    ctrl, _queue = _controller()
    ctrl._apply_pending_skip = AsyncMock()  # type: ignore[method-assign]

    await ctrl.skip(QUEUE_ID, 10)

    assert _cancelled(ctrl) == [(QUEUE_ID, "skip")]


async def test_skip_leaves_no_overlay_or_task_behind() -> None:
    """Dropping a live overlay is synchronous: nothing of it stays scheduled."""
    ctrl, _queue = _controller()
    ctrl._apply_pending_skip = AsyncMock()  # type: ignore[method-assign]
    streams: Any = StreamsController.__new__(StreamsController)
    streams._program_buses = {}
    streams.logger = MagicMock()
    ctrl.mass.streams = streams
    program_bus = _FlowProgramBus(TEST_PCM_FORMAT, cast("Any", MagicMock()))
    streams.register_program_bus(QUEUE_ID, program_bus)
    voice = struct.pack("<hh", 8, 8) * TEST_PCM_FORMAT.sample_rate * 3
    assert streams.schedule_overlay(QUEUE_ID, voice, label="announcement")
    before = asyncio.all_tasks()

    await ctrl.skip(QUEUE_ID, 10)

    assert program_bus.active is False
    assert asyncio.all_tasks() == before


async def test_play_index_drops_the_overlay() -> None:
    """Jumping to an item drops the announcement mixed into the previous one."""
    ctrl, _queue = _controller()
    ctrl._play_index = AsyncMock()  # type: ignore[method-assign]

    await ctrl.play_index(QUEUE_ID, 1)

    assert _cancelled(ctrl) == [(QUEUE_ID, "play_index")]


async def test_stop_leaves_the_overlay_to_the_ending_stream() -> None:
    """A stop ends the stream, which drops what it was still to play by itself."""
    ctrl, _queue = _controller()
    ctrl._end_session = Mock()  # type: ignore[method-assign]

    await ctrl._handle_stop(QUEUE_ID)

    assert _cancelled(ctrl) == []

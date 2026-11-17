"""Bare streams controllers and helpers for the program-bus tests."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from music_assistant_models.enums import PlaybackState

from music_assistant.controllers.streams import StreamsController
from music_assistant.controllers.streams.audio import FlowProgramBus
from music_assistant.controllers.streams.constants import LEAD_MIN_LEAD_SECONDS
from music_assistant.controllers.streams.program_bus_registry import ProgramBusRegistry
from music_assistant.controllers.streams.program_ledger import ProgramLedger
from music_assistant.controllers.streams.program_schedule import ProgramKind, ProgramPriority
from tests.controllers.streams.test_crossfade_transition import TEST_PCM_FORMAT

if TYPE_CHECKING:
    from music_assistant_models.media_items import AudioFormat

SECOND = TEST_PCM_FORMAT.sample_rate
LEAD_FRAMES = int(LEAD_MIN_LEAD_SECONDS * SECOND)
FRAME_BYTES = (TEST_PCM_FORMAT.bit_depth // 8) * TEST_PCM_FORMAT.channels
AUDIBLE_BEFORE = 1.5
PRODUCED_BEFORE = 3.0
QUEUE_ID = "queue-1"


def player_at(audible: float | None) -> MagicMock:
    """
    Return a player that reports the given stream position, if any.

    :param audible: Second of the stream the player has played, or None for no position.
    """
    player = MagicMock()
    player.state.playback_state = (
        PlaybackState.PLAYING if audible is not None else PlaybackState.IDLE
    )
    player.state.corrected_elapsed_time = audible
    return player


def bare_streams_controller(*, audible: float | None = None, enabled: bool = False) -> Any:
    """
    Return a streams controller whose program-bus state is ready to use.

    :param audible: Second of the stream the queue's player has heard, or None for a player
        that reports no position.
    :param enabled: Whether the program bus switch is on.
    """
    controller: Any = StreamsController.__new__(StreamsController)
    controller._overlays_without_position = set()
    controller._program_aired_listeners = []
    controller._flow_mode_owners = {}
    controller._boundary_mix_owners = {}
    controller._lead_hold_owners = {}
    controller.logger = MagicMock()
    controller.mass = MagicMock()
    controller.config = controller.mass.config
    controller.config.get_value.return_value = enabled
    controller.mass.players.get_player.return_value = player_at(audible)
    controller._program_ledger = ProgramLedger(
        on_dropped=controller._overlay_was_dropped,
        on_aired=controller._overlay_was_aired,
        logger=controller.logger,
    )
    controller._program_buses = ProgramBusRegistry(
        controller._program_ledger,
        heard_frame=controller._audible_frame,
        logger=controller.logger,
    )
    return controller


def register_bus(
    controller: Any,
    *,
    queue_id: str = QUEUE_ID,
    pcm_format: AudioFormat = TEST_PCM_FORMAT,
) -> FlowProgramBus:
    """Build a live flow session's bus and publish it for the queue."""
    program_bus = FlowProgramBus(pcm_format, MagicMock())
    controller.register_program_bus(queue_id, program_bus)
    return program_bus


def hear(controller: Any, seconds: float, *, paused: bool = False) -> None:
    """
    Move the queue's player to the given position in the stream.

    :param controller: The controller whose queue player to move.
    :param seconds: Second of the stream the player has played.
    :param paused: Whether the player is paused there instead of playing on.
    """
    player = controller.mass.players.get_player.return_value
    player.state.playback_state = PlaybackState.PAUSED if paused else PlaybackState.PLAYING
    player.state.corrected_elapsed_time = seconds


def voice(seconds: float, level: int = 8) -> bytes:
    """Build a voice at the given level for the given number of seconds."""
    return struct.pack("<hh", level, level) * int(seconds * SECOND)


def music(seconds: float, level: int = 100) -> bytes:
    """Build music at the given level for the given number of seconds."""
    return struct.pack("<hh", level, level) * int(seconds * SECOND)


def place(
    controller: Any,
    *,
    queue_id: str = QUEUE_ID,
    label: str = "term",
    start_in_seconds: float = 0.0,
    priority: ProgramPriority | None = None,
    kind: ProgramKind = ProgramKind.TERM,
    seconds: float = 1.0,
    head_seconds: float = 0.0,
) -> Any:
    """Place a program through the controller and return its result."""
    return controller.schedule_overlay(
        queue_id,
        voice(seconds),
        label=label,
        start_in_seconds=start_in_seconds,
        priority=priority,
        kind=kind,
        head_seconds=head_seconds,
    )

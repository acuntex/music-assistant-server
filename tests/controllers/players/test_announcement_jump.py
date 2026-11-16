"""A rendered announcement survives the jump in the queue that mixed it in."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from music_assistant.controllers.players.announcement_overlays import (
    OverlayAnnouncementRetention,
)
from music_assistant.controllers.streams.program_schedule import (
    ProgramKind,
    ScheduleReason,
)
from tests.controllers.streams.program_bus_helpers import (
    QUEUE_ID,
    bare_streams_controller,
    hear,
    music,
    register_bus,
    voice,
)

ANNOUNCEMENT = "announcement:one"


def _retention(streams: Any) -> OverlayAnnouncementRetention:
    """Subscribe a retention to the streams controller's bus changes."""
    return OverlayAnnouncementRetention(streams, MagicMock())


async def test_a_clip_that_was_not_heard_yet_is_anchored_again_after_a_skip() -> None:
    """The rendered audio is not lost with the session the jump ends."""
    streams = bare_streams_controller(audible=0.0)
    retention = _retention(streams)
    program_bus = register_bus(streams)
    await program_bus.feed(music(3.0))
    pcm = voice(1.0)
    placed = streams.schedule_overlay(
        QUEUE_ID, pcm, label=ANNOUNCEMENT, kind=ProgramKind.ANNOUNCEMENT
    )
    assert placed.scheduled
    retention.remember(QUEUE_ID, ANNOUNCEMENT, pcm)

    # what a skip does: the programs of the boundaries it passes go, the clip stays rendered
    streams.cancel_overlays_of_a_jump(QUEUE_ID, ScheduleReason.SEEKED, origin="skip")
    new_bus = register_bus(streams)

    states = [(state.label, state.state) for state in streams.overlay_state(QUEUE_ID)]
    # anchored again on the bus the jump opened, never dropped
    assert ("announcement:one", "scheduled") in states
    assert "dropped" not in {state for _label, state in states}
    assert new_bus.scheduled_voice_frames(ANNOUNCEMENT) is not None
    assert retention.retained_labels() == (ANNOUNCEMENT,)


async def test_a_clip_the_listener_was_hearing_is_released_instead_of_repeated() -> None:
    """What was already heard is cut with the session, not spoken a second time."""
    streams = bare_streams_controller(audible=0.0)
    retention = _retention(streams)
    program_bus = register_bus(streams)
    await program_bus.feed(music(3.0))
    pcm = voice(1.0)
    assert streams.schedule_overlay(
        QUEUE_ID, pcm, label=ANNOUNCEMENT, kind=ProgramKind.ANNOUNCEMENT
    ).scheduled
    retention.remember(QUEUE_ID, ANNOUNCEMENT, pcm)
    await program_bus.feed(music(4.0))
    hear(streams, 3.5)

    streams.cancel_overlays_of_a_jump(QUEUE_ID, ScheduleReason.SEEKED, origin="skip")
    new_bus = register_bus(streams)

    assert [(state.label, state.state) for state in streams.overlay_state(QUEUE_ID)] == [
        (ANNOUNCEMENT, "aired")
    ]
    assert new_bus.scheduled_voice_frames(ANNOUNCEMENT) is None
    assert retention.retained_labels() == ()

"""What a jump in the queue does to the programs placed on its stream."""

from __future__ import annotations

from typing import Any

import pytest

from music_assistant.controllers.streams.program_schedule import (
    BOUNDARY_KINDS,
    KINDS_A_JUMP_DROPS,
    ProgramKind,
    ScheduleReason,
)
from tests.controllers.streams.program_bus_helpers import (
    QUEUE_ID,
    bare_streams_controller,
    hear,
    music,
    place,
    register_bus,
)

ANNOUNCEMENT = "announcement:one"
# the two words a jump in the queue records, one for another item and one within the item
JUMP_REASONS = (ScheduleReason.SKIPPED, ScheduleReason.SEEKED)


def _controller() -> Any:
    """Return a streams controller whose listener is at the start of the stream."""
    return bare_streams_controller(audible=0.0)


def _term(controller: Any) -> Any:
    """Place a term for a boundary five seconds ahead of the listener."""
    return place(controller, label="term", kind=ProgramKind.TERM, start_in_seconds=5.0)


async def test_a_jump_drops_the_boundary_program_and_keeps_the_announcement() -> None:
    """A move in the queue leaves the boundary behind but not a rendered announcement."""
    controller = _controller()
    program_bus = register_bus(controller)
    await program_bus.feed(music(3.0))
    assert _term(controller).scheduled
    assert place(controller, label=ANNOUNCEMENT, kind=ProgramKind.ANNOUNCEMENT).scheduled

    controller.cancel_overlays_of_a_jump(QUEUE_ID, ScheduleReason.SKIPPED, origin="next")

    states = {
        state.label: (state.state, state.reason) for state in controller.overlay_state(QUEUE_ID)
    }
    assert states["term"] == ("dropped", "skipped")
    assert states[ANNOUNCEMENT] == ("scheduled", None)
    # the term is gone from the bus, the announcement is still there to be heard
    assert program_bus.scheduled_voice_frames("term") is None
    assert program_bus.scheduled_voice_frames(ANNOUNCEMENT) is not None


@pytest.mark.parametrize("kind", sorted(KINDS_A_JUMP_DROPS))
async def test_every_kind_a_jump_leaves_behind_is_dropped(kind: ProgramKind) -> None:
    """What a jump makes unreachable ends as skipped, whatever kind it is."""
    controller = _controller()
    program_bus = register_bus(controller)
    await program_bus.feed(music(3.0))
    # a program written for a boundary is placed ahead of it, an immediate one right away
    delay = 5.0 if kind in BOUNDARY_KINDS else 0.0
    assert place(controller, label="program", kind=kind, start_in_seconds=delay).scheduled

    controller.cancel_overlays_of_a_jump(QUEUE_ID, ScheduleReason.SKIPPED, origin="next")

    states = [
        (state.label, state.state, state.reason) for state in controller.overlay_state(QUEUE_ID)
    ]
    assert states == [("program", "dropped", "skipped")]


async def test_a_term_the_jump_passed_is_not_anchored_again_on_the_new_bus() -> None:
    """A skipped boundary is never made up for on the session the jump opens."""
    controller = _controller()
    program_bus = register_bus(controller)
    await program_bus.feed(music(3.0))
    assert _term(controller).scheduled
    controller.cancel_overlays_of_a_jump(QUEUE_ID, ScheduleReason.SKIPPED, origin="next")

    new_bus = register_bus(controller)

    assert new_bus.scheduled_voice_frames("term") is None
    states = {
        state.label: (state.state, state.reason) for state in controller.overlay_state(QUEUE_ID)
    }
    assert states["term"] == ("dropped", "skipped")


async def test_a_jump_across_a_running_announcement_closes_it_as_aired() -> None:
    """A clip the listener was hearing is cut, not repeated, so it counts as heard."""
    controller = _controller()
    program_bus = register_bus(controller)
    await program_bus.feed(music(3.0))
    assert place(
        controller,
        label=ANNOUNCEMENT,
        kind=ProgramKind.ANNOUNCEMENT,
        seconds=1.0,
    ).scheduled
    # the session produced on, and the listener is inside the clip
    await program_bus.feed(music(4.0))
    hear(controller, 3.5)
    aired: list[tuple[str, str]] = []
    controller.add_program_aired_listener(lambda queue_id, label: aired.append((queue_id, label)))

    controller.cancel_overlays_of_a_jump(QUEUE_ID, ScheduleReason.SEEKED, origin="skip")
    new_bus = register_bus(controller)

    assert [
        (state.label, state.state, state.reason) for state in controller.overlay_state(QUEUE_ID)
    ] == [(ANNOUNCEMENT, "aired", None)]
    # nothing of it airs a second time, and its audio is released through the ledger event
    assert new_bus.scheduled_voice_frames(ANNOUNCEMENT) is None
    assert aired == [(QUEUE_ID, ANNOUNCEMENT)]


async def test_a_queued_announcement_waits_for_the_bus_the_jump_opens() -> None:
    """A clip that was never heard keeps its non-terminal state for a re-anchor."""
    controller = _controller()
    program_bus = register_bus(controller)
    await program_bus.feed(music(3.0))
    assert place(controller, label=ANNOUNCEMENT, kind=ProgramKind.ANNOUNCEMENT).scheduled

    controller.cancel_overlays_of_a_jump(QUEUE_ID, ScheduleReason.SEEKED, origin="skip")
    register_bus(controller)

    assert [
        (state.label, state.state, state.reason) for state in controller.overlay_state(QUEUE_ID)
    ] == [(ANNOUNCEMENT, "orphaned", None)]


async def test_a_hard_abort_drops_the_announcement_too() -> None:
    """The unfiltered cancel is what a removed player or a deleted queue ends its programs with."""
    controller = _controller()
    program_bus = register_bus(controller)
    await program_bus.feed(music(3.0))
    assert place(controller, label=ANNOUNCEMENT, kind=ProgramKind.ANNOUNCEMENT).scheduled

    controller.cancel_overlays(QUEUE_ID, reason=ScheduleReason.FLOW_ENDED)

    assert program_bus.scheduled_voice_frames(ANNOUNCEMENT) is None
    assert [
        (state.label, state.state, state.reason) for state in controller.overlay_state(QUEUE_ID)
    ] == [(ANNOUNCEMENT, "dropped", "flow_ended")]


@pytest.mark.parametrize("reason", JUMP_REASONS)
async def test_the_reason_is_logged_together_with_its_origin(reason: ScheduleReason) -> None:
    """The ledger keeps the vocabulary word, while the log says which command asked for it."""
    controller = _controller()
    program_bus = register_bus(controller)
    await program_bus.feed(music(3.0))
    assert _term(controller).scheduled

    controller.cancel_overlays_of_a_jump(QUEUE_ID, reason, origin="play_index")

    logged = " ".join(
        str(arg) for call in controller.logger.debug.call_args_list for arg in call.args
    )
    assert "play_index" in logged
    assert [(state.state, state.reason) for state in controller.overlay_state(QUEUE_ID)] == [
        ("dropped", str(reason))
    ]

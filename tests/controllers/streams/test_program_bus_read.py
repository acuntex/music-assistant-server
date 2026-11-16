"""The program-bus read command: what the audio path is told about a queue's bus."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from music_assistant_models.auth import Scope

from music_assistant.controllers.streams import StreamsController
from music_assistant.controllers.streams.audio import FlowProgramBus
from music_assistant.controllers.streams.program_bus_view import (
    MAX_LABEL_LEN,
    ProgramBusClaims,
    build_program_bus_view,
)
from music_assistant.controllers.streams.program_ledger import ProgramLedger
from music_assistant.controllers.streams.program_schedule import (
    ProgramKind,
    ProgramPriority,
)
from tests.controllers.streams.program_bus_helpers import (
    AUDIBLE_BEFORE,
    PRODUCED_BEFORE,
    QUEUE_ID,
    SECOND,
    TEST_PCM_FORMAT,
    bare_streams_controller,
)
from tests.controllers.streams.program_bus_helpers import (
    hear as _hear,
)
from tests.controllers.streams.program_bus_helpers import (
    music as _music,
)
from tests.controllers.streams.program_bus_helpers import (
    place as _schedule,
)
from tests.controllers.streams.program_bus_helpers import (
    register_bus as _register,
)
from tests.controllers.streams.program_bus_helpers import (
    voice as _voice,
)


def _controller(*, audible: float | None = 0.0, enabled: bool = True) -> Any:
    """Return a bare controller whose bus is switched on unless told otherwise."""
    return bare_streams_controller(audible=audible, enabled=enabled)


async def _scheduled(
    controller: Any,
    *,
    label: str = "term",
    start: float = 5.0,
    feed: float = 1.0,
    head: float = 0.0,
) -> Any:
    """Place one program and return the queue's reading afterwards."""
    program_bus = _register(controller)
    await program_bus.feed(_music(feed))
    _schedule(controller, label=label, start_in_seconds=start, head_seconds=head)
    return controller.program_bus(QUEUE_ID)


def _reading(
    ledger: ProgramLedger,
    program_bus: FlowProgramBus | None,
    position: int | None,
) -> Any:
    """Read a ledger and a bus the way the command does, without a queue around them."""
    return build_program_bus_view(
        enabled=True,
        queue_id=QUEUE_ID,
        ledger=ledger,
        program_bus=program_bus,
        claims=ProgramBusClaims(flow=False, boundary=False),
        position=position,
    )


def test_the_reading_is_empty_while_the_bus_is_switched_off() -> None:
    """With the switch off nothing is mixed, so the panel shows the empty state."""
    controller = _controller(enabled=False)
    _register(controller)

    assert controller.program_bus(QUEUE_ID).to_dict() == {
        "enabled": False,
        "session_active": False,
        "no_audible_position": False,
        "format": None,
        "claims": {"flow": False, "boundary": False},
        "overlays": [],
        "last": [],
    }


def test_an_active_bus_reports_its_session_and_format_without_programs() -> None:
    """A live flow session is reported with the format a program has to come in."""
    controller = _controller()
    _register(controller)

    view = controller.program_bus(QUEUE_ID)

    assert view.enabled is True
    assert view.session_active is True
    assert view.format is not None
    assert view.format.to_dict() == {
        "sample_rate": 8000,
        "bit_depth": 16,
        "channels": 2,
    }
    assert view.overlays == []
    assert view.last == []


def test_the_claims_the_registry_owners_hold_are_reported() -> None:
    """The flow and boundary claims come from the owners, not from the bus."""
    controller = _controller()
    _register(controller)
    controller.set_flow_mode_required(QUEUE_ID, "ai_radio", True)
    controller.set_boundary_mix_required(QUEUE_ID, "ai_radio", True)

    claims = controller.program_bus(QUEUE_ID).claims

    assert (claims.flow, claims.boundary) == (True, True)


def test_the_claims_of_another_queue_are_not_reported() -> None:
    """A claim belongs to the queue it was taken for."""
    controller = _controller()
    _register(controller)
    controller.set_flow_mode_required("queue-2", "ai_radio", True)

    claims = controller.program_bus(QUEUE_ID).claims

    assert (claims.flow, claims.boundary) == (False, False)


async def test_a_planned_program_shows_its_label_state_and_seconds_to_go() -> None:
    """A program the bus holds is listed with what is still to come."""
    controller = _controller()

    view = await _scheduled(controller, label="term", start=5.0, head=4.0)

    assert [overlay.to_dict() for overlay in view.overlays] == [
        {
            "label": "term",
            "state": "scheduled",
            "start_in_seconds": 5.0,
            "frames": SECOND,
            "head_seconds": 4.0,
            "late_by_seconds": 0.0,
        }
    ]
    assert view.last == []


async def test_a_program_the_bus_placed_late_says_how_late() -> None:
    """The head is what was planned; lateness is what the bus had to do to place it."""
    controller = _controller()
    program_bus = _register(controller)
    # the producer is half a second past the boundary before the program is offered
    await program_bus.feed(_music(5.5))

    _schedule(controller, label="term", start_in_seconds=5.0)

    overlay = controller.program_bus(QUEUE_ID).overlays[0]
    assert overlay.start_in_seconds == 5.5
    assert overlay.late_by_seconds == 0.5


async def test_a_program_without_a_head_reports_zero() -> None:
    """An announcement has no boundary before it, so its head is nothing."""
    controller = _controller()
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="announcement", kind=ProgramKind.ANNOUNCEMENT)

    overlay = controller.program_bus(QUEUE_ID).overlays[0]
    assert overlay.head_seconds == 0.0
    # an announcement has no boundary, so nothing can make it late for one
    assert overlay.late_by_seconds == 0.0


async def test_the_seconds_to_go_follow_the_heard_position() -> None:
    """The countdown is relative to what the listener hears, not to the produced frames."""
    controller = _controller()
    view = await _scheduled(controller, start=11.0, feed=10.0)
    assert view.overlays[0].start_in_seconds == 11.0

    _hear(controller, 9.0)

    assert controller.program_bus(QUEUE_ID).overlays[0].start_in_seconds == 2.0


async def test_a_paused_queue_keeps_showing_the_seconds_to_go() -> None:
    """A pause holds the deadline; the panel still says how long the wait is."""
    controller = _controller()
    await _scheduled(controller, start=11.0, feed=10.0)

    _hear(controller, 9.0, paused=True)

    assert controller.program_bus(QUEUE_ID).overlays[0].start_in_seconds == 2.0


async def test_a_program_starting_right_now_reads_zero_not_unknown() -> None:
    """A wait of nothing is a number the panel shows, not a missing position."""
    controller = _controller()
    program_bus = _register(controller)
    await program_bus.feed(_music(10.0))
    _schedule(
        controller,
        label="announcement",
        kind=ProgramKind.ANNOUNCEMENT,
    )
    _hear(controller, 10.0)

    assert controller.program_bus(QUEUE_ID).overlays[0].start_in_seconds == 0.0


async def test_a_dropped_program_is_listed_with_the_reason_it_was_not_heard() -> None:
    """The reason is the whole point of the list: the panel has to show it."""
    controller = _controller()
    await _scheduled(controller, label="term", start=5.0)

    controller.cancel_overlays(QUEUE_ID, reason="skipped")

    view = controller.program_bus(QUEUE_ID)
    assert view.overlays == []
    assert [ended.to_dict() for ended in view.last] == [
        {
            "label": "term",
            "state": "dropped",
            "reason": "skipped",
            "frames": SECOND,
        }
    ]


async def test_an_aired_program_moves_to_the_last_list_without_a_reason() -> None:
    """What was heard to its end is history, and it is not a failure."""
    controller = _controller()
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="term", start_in_seconds=5.0)
    await program_bus.feed(_music(10.0))

    _hear(controller, 6.0)

    view = controller.program_bus(QUEUE_ID)
    assert view.overlays == []
    assert [(ended.label, ended.state, ended.reason) for ended in view.last] == [
        ("term", "aired", None)
    ]


async def test_an_announcement_that_lost_its_bus_stays_pending() -> None:
    """A recoverable announcement is still to be heard, so it is not in the last list."""
    controller = _controller()
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(
        controller,
        label="announcement",
        start_in_seconds=0.0,
        kind=ProgramKind.ANNOUNCEMENT,
    )

    _register(controller)

    view = controller.program_bus(QUEUE_ID)
    assert [(overlay.label, overlay.state) for overlay in view.overlays] == [
        ("announcement", "orphaned")
    ]
    assert view.last == []


async def test_the_lead_is_the_distance_between_the_produced_and_the_heard_frame() -> None:
    """What the producer is ahead of the listener is the lead a program has to respect."""
    controller = _controller(audible=AUDIBLE_BEFORE)
    program_bus = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))

    lead = controller.program_bus_lead_seconds(QUEUE_ID)

    assert lead is not None
    assert lead == pytest.approx(PRODUCED_BEFORE - AUDIBLE_BEFORE)


async def test_a_lead_of_nothing_is_a_number_not_an_unknown() -> None:
    """A bus that produced exactly what was heard reports zero, which is a lead too."""
    controller = _controller(audible=AUDIBLE_BEFORE)
    program_bus = _register(controller)
    await program_bus.feed(_music(AUDIBLE_BEFORE))

    assert controller.program_bus_lead_seconds(QUEUE_ID) == 0.0


def test_a_queue_without_a_bus_has_no_lead() -> None:
    """Nothing is produced for a queue the stream does not mix, so there is no lead."""
    controller = _controller(audible=AUDIBLE_BEFORE)

    assert controller.program_bus_lead_seconds(QUEUE_ID) is None


def test_a_bus_without_a_heard_position_has_no_lead() -> None:
    """Without a position the listener has reached, nothing says how far ahead the bus is."""
    controller = _controller(audible=None)
    _register(controller)

    assert controller.program_bus_lead_seconds(QUEUE_ID) is None


def test_a_bus_without_a_heard_position_says_placement_is_paused() -> None:
    """The multiroom case: a program ahead can not be placed without a position."""
    controller = _controller(audible=None)
    _register(controller)

    view = controller.program_bus(QUEUE_ID)

    assert view.session_active is True
    assert view.no_audible_position is True


def test_a_heard_position_means_placement_is_not_paused() -> None:
    """A position the panel can show is also a position the bus can place on."""
    controller = _controller(audible=1.0)
    _register(controller)

    assert controller.program_bus(QUEUE_ID).no_audible_position is False


async def test_the_readout_carries_no_program_audio() -> None:
    """The rows describe the bus; the audio never leaves the stream."""
    controller = _controller()
    await _scheduled(controller, start=5.0)
    controller.cancel_overlays(QUEUE_ID, reason="too_late")
    await _scheduled(controller, label="next", start=9.0)

    payload = controller.program_bus(QUEUE_ID).to_dict()

    assert set(payload) == {
        "enabled",
        "session_active",
        "no_audible_position",
        "format",
        "claims",
        "overlays",
        "last",
    }
    assert set(payload["format"]) == {"sample_rate", "bit_depth", "channels"}
    assert set(payload["claims"]) == {"flow", "boundary"}
    assert set(payload["overlays"][0]) == {
        "label",
        "state",
        "start_in_seconds",
        "frames",
        "head_seconds",
        "late_by_seconds",
    }
    assert set(payload["last"][0]) == {"label", "state", "reason", "frames"}


async def test_a_long_label_is_cut_to_what_the_panel_can_show() -> None:
    """A program label is not a prompt: the reading carries a short name only."""
    controller = _controller()

    view = await _scheduled(controller, label="s" * (MAX_LABEL_LEN + 20), start=5.0)

    assert len(view.overlays[0].label) == MAX_LABEL_LEN
    assert view.overlays[0].label == "s" * MAX_LABEL_LEN


def test_a_label_is_cut_between_graphemes_not_inside_one() -> None:
    """A cut label never ends in half an emoji, a lone accent or half a flag."""
    ledger = ProgramLedger(logger=MagicMock())
    program_bus = FlowProgramBus(TEST_PCM_FORMAT, MagicMock())
    # each of these is one grapheme, so a cut between characters must keep it whole
    composed = f"{'a' * (MAX_LABEL_LEN - 1)}👩‍👩‍👧‍👦tail"
    accented = f"{'a' * (MAX_LABEL_LEN - 1)}e\u0301tail"
    flagged = f"{'a' * (MAX_LABEL_LEN - 1)}🇩🇪tail"
    for label, kept in (
        (composed, "👩‍👩‍👧‍👦"),
        (accented, "e\u0301"),
        (flagged, "🇩🇪"),
    ):
        ledger.record_scheduled(
            QUEUE_ID,
            label=label,
            start_frame=0,
            frames=SECOND,
            priority=ProgramPriority.TERM,
            kind=ProgramKind.TERM,
            reason=None,
            bus=program_bus,
        )

        short = _reading(ledger, program_bus, 0).overlays[-1].label

        assert short == f"{'a' * (MAX_LABEL_LEN - 1)}{kept}"


def test_a_queue_without_a_bus_reads_as_no_bus() -> None:
    """An unknown queue is an empty readout, not an error."""
    controller = _controller()

    view = controller.program_bus("queue-unknown")

    assert view.enabled is True
    assert view.session_active is False
    assert view.no_audible_position is False
    assert view.format is None
    assert view.overlays == []
    assert view.last == []


def test_program_bus_is_a_queue_scoped_api_command() -> None:
    """The bus belongs to a player's queue, so reading it needs the queue read scope."""
    assert StreamsController.program_bus.api_cmd == "streams/program_bus"
    assert StreamsController.program_bus.api_required_scope is Scope.QUEUES_READ


def test_a_start_the_listener_passed_reads_as_now_not_as_negative() -> None:
    """A clock that moved past the anchor shows no countdown, never a negative one."""
    ledger = ProgramLedger(logger=MagicMock())
    program_bus = FlowProgramBus(TEST_PCM_FORMAT, MagicMock())
    ledger.record_scheduled(
        QUEUE_ID,
        label="term",
        start_frame=0,
        frames=2 * SECOND,
        priority=ProgramPriority.TERM,
        kind=ProgramKind.TERM,
        reason=None,
        bus=program_bus,
    )

    overlay = _reading(ledger, program_bus, SECOND).overlays[0]

    assert overlay.start_in_seconds == 0.0


def test_an_overlay_of_a_replaced_bus_has_no_countdown() -> None:
    """Frames belong to the session they were counted in, so an old bus has no position."""
    ledger = ProgramLedger(logger=MagicMock())
    old_bus = FlowProgramBus(TEST_PCM_FORMAT, MagicMock())
    new_bus = FlowProgramBus(TEST_PCM_FORMAT, MagicMock())
    ledger.record_scheduled(
        QUEUE_ID,
        label="announcement",
        start_frame=5 * SECOND,
        frames=SECOND,
        priority=ProgramPriority.ANNOUNCEMENT,
        kind=ProgramKind.ANNOUNCEMENT,
        reason=None,
        bus=old_bus,
    )

    overlay = _reading(ledger, new_bus, SECOND).overlays[0]

    assert overlay.state == "scheduled"
    assert overlay.start_in_seconds is None


async def test_a_program_heard_to_its_end_before_a_pause_is_aired_while_paused() -> None:
    """A frozen position still says what was already heard, so the reading may say aired."""
    controller = _controller()
    program_bus = _register(controller)
    await program_bus.feed(_music(10.0))
    _schedule(controller, label="announcement", kind=ProgramKind.ANNOUNCEMENT)
    # the session produced past the program, so the frozen position can be past its end
    await program_bus.feed(_music(3.0))

    _hear(controller, 11.0, paused=True)

    view = controller.program_bus(QUEUE_ID)
    assert view.overlays == []
    assert [(ended.label, ended.state) for ended in view.last] == [("announcement", "aired")]


async def test_the_reading_asks_for_the_heard_position_once(monkeypatch: Any) -> None:
    """One reading is one position: two reads could disagree within the same answer."""
    controller = _controller()
    await _scheduled(controller, start=5.0, feed=1.0)
    await _scheduled(controller, label="next", start=9.0, feed=1.0)
    reads = []
    original = controller._audible_frame

    def _counted(queue_id: str, program_bus: Any) -> int | None:
        reads.append(program_bus)
        return original(queue_id, program_bus)

    monkeypatch.setattr(controller, "_audible_frame", _counted)
    controller.program_bus(QUEUE_ID)

    assert len(reads) == 1


async def test_a_shortened_program_is_read_with_the_frames_it_plays() -> None:
    """The ui is told the audio that is left of a program, not the length it was offered."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(3.0))
    # five seconds are offered at one second, but the bus already produced three of them, so
    # the placement it takes is three seconds long
    planned = program_bus.schedule(_voice(5.0), 1 * SECOND, label="term")
    assert planned == 3 * SECOND
    assert program_bus.scheduled_voice_frames("term") == 3 * SECOND
    controller._program_ledger.record_scheduled(
        QUEUE_ID,
        label="term",
        start_frame=planned,
        frames=3 * SECOND,
        priority=ProgramPriority.TERM,
        kind=ProgramKind.TERM,
        reason=None,
        bus=program_bus,
    )

    states = controller.overlay_state(QUEUE_ID)
    assert [state.frames for state in states] == [3 * SECOND]
    view = controller.program_bus(QUEUE_ID)
    assert [overlay.frames for overlay in view.overlays] == [3 * SECOND]

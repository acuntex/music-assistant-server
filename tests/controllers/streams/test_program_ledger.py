"""The per-queue program ledger: states, deadlines and the answer overlay_state gives."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import MagicMock

import pytest
from music_assistant_models.auth import Scope

from music_assistant.controllers.streams import StreamsController
from music_assistant.controllers.streams.audio import FlowProgramBus
from music_assistant.controllers.streams.program_ledger import ProgramLedger
from music_assistant.controllers.streams.program_schedule import ProgramKind, ProgramPriority
from tests.controllers.streams.program_bus_helpers import (
    QUEUE_ID,
    SECOND,
    TEST_PCM_FORMAT,
)
from tests.controllers.streams.program_bus_helpers import (
    bare_streams_controller as _controller,
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


async def test_a_program_the_listener_is_inside_reads_as_playing() -> None:
    """A program is playing while the listener is inside its window, and only then."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(5.0))
    _schedule(controller, label="term", start_in_seconds=5.0)
    await program_bus.feed(_music(5.0))

    _hear(controller, 4.5)
    before = controller.overlay_state(QUEUE_ID)[0].playing
    _hear(controller, 5.5)
    inside = controller.overlay_state(QUEUE_ID)[0].playing
    _hear(controller, 6.0)
    aired = controller.overlay_state(QUEUE_ID)[0]

    assert (before, inside) == (False, True)
    # a program that was heard to its end is no longer playing, it is over
    assert (aired.state, aired.playing) == ("aired", False)


async def test_the_ledger_records_a_program_from_scheduled_to_aired() -> None:
    """A program is aired once the listener reached its end, not when the bus produced it."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="term", start_in_seconds=5.0)

    scheduled = controller.overlay_state(QUEUE_ID)
    await program_bus.feed(_music(7.0))
    still_scheduled = controller.overlay_state(QUEUE_ID)
    _hear(controller, 6.0)
    aired = controller.overlay_state(QUEUE_ID)

    assert [(state.label, state.state) for state in scheduled] == [("term", "scheduled")]
    # the bus is far ahead, but the listener has not heard it yet
    assert [(state.label, state.state) for state in still_scheduled] == [("term", "scheduled")]
    assert [(state.label, state.state) for state in aired] == [("term", "aired")]
    assert aired[0].start_frame == 5 * SECOND
    assert aired[0].frames == SECOND


async def test_a_program_is_aired_exactly_on_its_deadline_not_before() -> None:
    """The frame before the deadline is still scheduled, the deadline itself is aired."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="term", start_in_seconds=5.0)

    await program_bus.feed(_music(10.0))
    _hear(controller, 6.0 - 0.001)
    assert [state.state for state in controller.overlay_state(QUEUE_ID)] == ["scheduled"]

    _hear(controller, 6.0)
    assert [state.state for state in controller.overlay_state(QUEUE_ID)] == ["aired"]


async def test_the_ledger_records_a_dropped_program_with_its_reason() -> None:
    """A program that lost its window is in the ledger as dropped, with why."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(
        controller,
        label="announcement",
        kind=ProgramKind.ANNOUNCEMENT,
    )
    _schedule(controller, label="term", start_in_seconds=2.0)

    term = next(state for state in controller.overlay_state(QUEUE_ID) if state.label == "term")

    assert term.state == "dropped"
    assert term.reason == "window_taken"


async def test_the_ledger_keeps_the_last_sixteen_programs() -> None:
    """The per-queue ledger is bounded, so a long session does not grow it."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))

    for index in range(19):
        _schedule(controller, label=f"program-{index}", start_in_seconds=10.0 + index * 2)

    labels = [state.label for state in controller.overlay_state(QUEUE_ID)]
    assert len(labels) == 16
    assert labels == [f"program-{index}" for index in range(3, 19)]
    assert {state.state for state in controller.overlay_state(QUEUE_ID)} == {"scheduled"}


async def test_the_ledger_holds_no_pcm() -> None:
    """Only the description of a program is kept, never its audio."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="term", start_in_seconds=5.0)
    _schedule(controller, label="announcement", start_in_seconds=5.0)

    entries = list(controller._program_ledger.entries(QUEUE_ID))
    assert entries
    for entry in entries:
        for value in vars(entry).values():
            assert not isinstance(value, (bytes, bytearray, memoryview))


async def test_overlay_state_returns_a_copy_of_the_ledger() -> None:
    """A caller mutating the answer can not change the ledger or another caller's answer."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="term", start_in_seconds=5.0)
    first = controller.overlay_state(QUEUE_ID)

    first.append(first[0])
    with pytest.raises(FrozenInstanceError):
        first[0].label = "changed"  # type: ignore[misc]

    second = controller.overlay_state(QUEUE_ID)
    assert [(state.label, state.state) for state in second] == [("term", "scheduled")]


async def test_a_paused_program_keeps_its_deadline() -> None:
    """A pause stops the stream-time deadline: what was scheduled stays scheduled."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="term", start_in_seconds=5.0)
    # the pause is past the deadline, so only the pause keeps it from airing
    _hear(controller, 30.0, paused=True)

    await program_bus.feed(_music(30.0))

    assert [state.state for state in controller.overlay_state(QUEUE_ID)] == ["scheduled"]


async def test_a_removed_queue_leaves_no_program_behind() -> None:
    """A queue that is gone takes its ledger with it, so a new one under the id starts empty."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="term", start_in_seconds=5.0)
    controller._overlays_without_position.add(QUEUE_ID)

    controller.on_queue_removed(QUEUE_ID)

    assert controller.overlay_state(QUEUE_ID) == []
    assert controller._overlays_without_position == set()

    again = _register(controller)
    await again.feed(_music(1.0))
    _schedule(controller, label="announcement", kind=ProgramKind.ANNOUNCEMENT)

    assert [state.label for state in controller.overlay_state(QUEUE_ID)] == ["announcement"]


async def test_a_program_is_not_aired_from_the_bus_clock_alone() -> None:
    """The bus producing far ahead says nothing about what the listener has heard."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="term", start_in_seconds=5.0)

    await program_bus.feed(_music(30.0))

    assert [state.state for state in controller.overlay_state(QUEUE_ID)] == ["scheduled"]


async def test_a_program_the_bus_shortened_airs_where_its_audio_ends() -> None:
    """A program whose head the bus dropped keeps no deadline past the audio that is left."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(3.0))

    # five seconds are offered a second into the stream, but the bus already produced two of
    # them: a fifth of the delay plus the produced frames are gone, 40 % of the program stays
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

    assert controller._program_ledger.entries(QUEUE_ID)[0].deadline_frame == 6 * SECOND

    # the listener reaches the end of what is left at six seconds, not at eight
    await program_bus.feed(_music(5.0))
    _hear(controller, 5.5)
    assert [state.state for state in controller.overlay_state(QUEUE_ID)] == ["scheduled"]
    _hear(controller, 6.0)
    assert [state.state for state in controller.overlay_state(QUEUE_ID)] == ["aired"]


async def test_two_programs_under_one_label_keep_their_own_row() -> None:
    """Two placements of one label are told apart by their anchor and their own length."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="transition", start_in_seconds=5.0, seconds=2.0)
    _schedule(controller, label="transition", start_in_seconds=20.0, seconds=4.0)

    rows = [state for state in controller.overlay_state(QUEUE_ID) if state.label == "transition"]

    assert [(row.start_frame, row.frames) for row in rows] == [
        (5 * SECOND, 2 * SECOND),
        (20 * SECOND, 4 * SECOND),
    ]


async def test_a_window_drop_touches_only_the_program_at_that_anchor() -> None:
    """Two programs sharing a label are told apart by their anchor when one gives way."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="term", start_in_seconds=2.0)
    _schedule(controller, label="term", start_in_seconds=13.0)

    _schedule(
        controller,
        label="announcement",
        kind=ProgramKind.ANNOUNCEMENT,
    )

    terms = [state for state in controller.overlay_state(QUEUE_ID) if state.label == "term"]
    assert [(state.start_frame, state.state) for state in terms] == [
        (2 * SECOND, "dropped"),
        (13 * SECOND, "scheduled"),
    ]
    assert program_bus.scheduled_voice_frames("term") is not None


def test_overlay_state_is_a_queue_scoped_api_command() -> None:
    """The ledger is readable through the api under the queue read scope."""
    assert StreamsController.overlay_state.api_cmd == "streams/overlay_state"
    assert StreamsController.overlay_state.api_required_scope is Scope.QUEUES_READ


async def test_overlay_state_rows_serialize_without_pcm() -> None:
    """A row handed to the api carries the state description only, never the audio."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="term", start_in_seconds=5.0)

    payload = [state.to_dict() for state in controller.overlay_state(QUEUE_ID)]

    assert payload == [
        {
            "label": "term",
            "state": "scheduled",
            "start_frame": 5 * SECOND,
            "frames": SECOND,
            "reason": None,
            "priority": 2,
            "playing": False,
        }
    ]


async def test_the_ledger_ends_the_oldest_program_when_it_overflows() -> None:
    """An overflowing ledger does not silently forget a program that is still to be heard."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    for index in range(16):
        _schedule(controller, label=f"program-{index}", start_in_seconds=5.0 + index * 2)
    assert program_bus.scheduled_voice_frames("program-0") is not None

    _schedule(controller, label="program-16", start_in_seconds=37.0)

    labels = [state.label for state in controller.overlay_state(QUEUE_ID)]
    assert len(labels) == 16
    assert "program-0" not in labels
    # the evicted program is stopped on the bus and the eviction is reported
    assert program_bus.scheduled_voice_frames("program-0") is None
    assert any(
        "ledger_evicted" in str(call.args[0]) or "ledger_evicted" in str(call.args[1:])
        for call in controller.logger.warning.call_args_list
    )


def _ledger_with_bus(max_entries: int) -> tuple[Any, Any]:
    """Build a ledger and the bus its entries refer to."""
    program_bus = FlowProgramBus(TEST_PCM_FORMAT, MagicMock())
    return ProgramLedger(max_entries=max_entries, logger=MagicMock()), program_bus


def _record(ledger: Any, bus: Any, label: str, kind: Any = ProgramKind.TERM) -> None:
    """Record a placed program in the ledger."""
    ledger.record_scheduled(
        QUEUE_ID,
        label=label,
        start_frame=5 * SECOND,
        frames=SECOND,
        priority=ProgramPriority.TERM,
        kind=kind,
        reason=None,
        bus=bus,
    )


def test_two_programs_under_one_label_each_keep_their_own_length() -> None:
    """A label is no identity: each row keeps the length of the placement it belongs to."""
    ledger, bus = _ledger_with_bus(4)

    # the first placement was offered five seconds and the bus plays three of them, the
    # second one plays whole; a label shared by both can tell neither length from the other
    ledger.record_scheduled(
        QUEUE_ID,
        label="transition",
        start_frame=0,
        frames=3 * SECOND,
        priority=ProgramPriority.TERM,
        kind=ProgramKind.TERM,
        reason=None,
        bus=bus,
    )
    ledger.record_scheduled(
        QUEUE_ID,
        label="transition",
        start_frame=20 * SECOND,
        frames=SECOND,
        priority=ProgramPriority.TERM,
        kind=ProgramKind.TERM,
        reason=None,
        bus=bus,
    )

    assert [(entry.start_frame, entry.frames) for entry in ledger.entries(QUEUE_ID)] == [
        (0, 3 * SECOND),
        (20 * SECOND, SECOND),
    ]


def test_the_ledger_gives_up_an_ended_program_first() -> None:
    """Overflow removes a program that already ended before a queued one is touched."""
    ledger, bus = _ledger_with_bus(2)
    _record(ledger, bus, "ended")
    ledger.mark_dropped(QUEUE_ID, "ended", "test")
    _record(ledger, bus, "queued")

    _record(ledger, bus, "new")

    assert [entry.label for entry in ledger.entries(QUEUE_ID)] == ["queued", "new"]


def test_the_ledger_keeps_a_queued_announcement_visible_as_orphaned() -> None:
    """A queued announcement is the last thing to give way, and it stays visible."""
    ledger, bus = _ledger_with_bus(1)
    _record(ledger, bus, "announcement:one", kind=ProgramKind.ANNOUNCEMENT)

    _record(ledger, bus, "announcement:two", kind=ProgramKind.ANNOUNCEMENT)

    entries = {entry.label: entry for entry in ledger.entries(QUEUE_ID)}
    assert entries["announcement:one"].state == "orphaned"
    # the placement was given up, so it must not be anchored again
    assert entries["announcement:one"].reason == "ledger_evicted"
    assert entries["announcement:two"].state == "scheduled"


@pytest.mark.parametrize("max_entries", [0, 1])
def test_a_tiny_ledger_limit_does_not_raise(max_entries: int) -> None:
    """A ledger of zero or one still accepts a program instead of popping an empty one."""
    ledger, bus = _ledger_with_bus(max_entries)
    _record(ledger, bus, "first")

    _record(ledger, bus, "second")

    assert [entry.label for entry in ledger.entries(QUEUE_ID)] == ["second"]


def test_the_ledger_reclaims_a_visible_orphaned_announcement_on_the_next_program() -> None:
    """The kept-back announcement is stopped, reported and replaced, and the bound holds."""
    program_bus = FlowProgramBus(TEST_PCM_FORMAT, MagicMock())
    logger = MagicMock()
    dropped = MagicMock()
    ledger = ProgramLedger(max_entries=1, logger=logger, on_dropped=dropped)
    _record(ledger, program_bus, "announcement:one", kind=ProgramKind.ANNOUNCEMENT)
    _record(ledger, program_bus, "announcement:two", kind=ProgramKind.ANNOUNCEMENT)
    assert [entry.label for entry in ledger.entries(QUEUE_ID)] == [
        "announcement:one",
        "announcement:two",
    ]

    # keeping it back means stopping it: it can not air as an "orphaned" program
    dropped.assert_called_once()
    assert dropped.call_args.args[0:2] == (QUEUE_ID, "announcement:one")
    assert any("announcement:one" in str(call.args) for call in logger.warning.call_args_list)

    _record(ledger, program_bus, "term")

    labels = [entry.label for entry in ledger.entries(QUEUE_ID)]
    assert labels == ["announcement:two", "term"]
    # reclaiming its line does not cancel it a second time
    assert [call.args[1] for call in dropped.call_args_list] == ["announcement:one"]

    _record(ledger, program_bus, "another-term")

    # the fourth record evicts the term, which is cancelled once; no program twice
    assert sorted(call.args[1] for call in dropped.call_args_list) == [
        "announcement:one",
        "term",
    ]
    assert dropped.call_count == 2
    assert len(ledger.entries(QUEUE_ID)) == 2


async def test_a_kept_back_announcement_is_cancelled_on_the_bus() -> None:
    """The last-resort announcement is stopped on the real bus, not only marked."""
    controller = _controller(audible=0.0)
    controller._program_ledger = ProgramLedger(
        max_entries=1,
        on_dropped=controller._overlay_was_dropped,
        on_aired=controller._overlay_was_aired,
        logger=controller.logger,
    )
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="announcement:one", kind=ProgramKind.ANNOUNCEMENT)

    _schedule(controller, label="announcement:two", kind=ProgramKind.ANNOUNCEMENT)

    assert program_bus.scheduled_voice_frames("announcement:one") is None
    assert program_bus.scheduled_voice_frames("announcement:two") is not None


async def test_cancelling_one_ledger_row_hits_only_its_bus_window() -> None:
    """Two rows sharing a label are told apart by their anchor on the real bus."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="same", start_in_seconds=5.0)
    _schedule(controller, label="same", start_in_seconds=13.0)

    controller._program_ledger.mark_dropped(QUEUE_ID, "same", "test", start_frame=5 * SECOND)

    assert [(window.label, window.start_frame) for window in program_bus.scheduled_windows()] == [
        ("same", 13 * SECOND)
    ]

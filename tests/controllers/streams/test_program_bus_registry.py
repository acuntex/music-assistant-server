"""The queue program bus registry: registration, replacement, release and events."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from music_assistant.controllers.streams.audio import FlowProgramBus
from music_assistant.controllers.streams.program_levels import ProgramLevels
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


async def test_a_replaced_bus_orphans_what_it_still_had_to_play() -> None:
    """A session that is superseded leaves its programs orphaned, not dropped."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(
        controller,
        label="announcement",
        kind=ProgramKind.ANNOUNCEMENT,
    )

    _register(controller)

    orphaned = controller.overlay_state(QUEUE_ID)
    assert [(state.label, state.state, state.reason) for state in orphaned] == [
        ("announcement", "orphaned", None)
    ]


async def test_a_released_bus_drops_a_term_that_was_never_played() -> None:
    """A session that ends can not air its terms, so they are dropped as flow ended."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="term", start_in_seconds=5.0)

    controller.release_program_bus(QUEUE_ID, program_bus)

    assert [
        (state.label, state.state, state.reason) for state in controller.overlay_state(QUEUE_ID)
    ] == [("term", "dropped", "flow_ended")]


async def test_a_released_bus_orphans_an_announcement() -> None:
    """An announcement keeps its non-terminal state so a new bus can anchor it again."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(
        controller,
        label="announcement",
        kind=ProgramKind.ANNOUNCEMENT,
    )

    controller.release_program_bus(QUEUE_ID, program_bus)

    assert [
        (state.label, state.state, state.reason) for state in controller.overlay_state(QUEUE_ID)
    ] == [("announcement", "orphaned", None)]


async def test_a_replaced_bus_drops_a_term_as_bus_replaced() -> None:
    """A term belongs to the session that took it, so a new bus ends it."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="term", start_in_seconds=5.0)

    _register(controller)

    assert [
        (state.label, state.state, state.reason) for state in controller.overlay_state(QUEUE_ID)
    ] == [("term", "dropped", "bus_replaced")]


async def test_a_replaced_bus_orphans_only_its_announcements() -> None:
    """A replacement orphans announcements and drops everything else."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="term", start_in_seconds=13.0)
    _schedule(
        controller,
        label="announcement",
        kind=ProgramKind.ANNOUNCEMENT,
    )

    _register(controller)

    assert [
        (state.label, state.state, state.reason) for state in controller.overlay_state(QUEUE_ID)
    ] == [
        ("term", "dropped", "bus_replaced"),
        ("announcement", "orphaned", None),
    ]


async def test_an_aired_announcement_is_not_orphaned_by_a_new_bus() -> None:
    """A program the listener already heard must not be anchored a second time."""
    controller = _controller(audible=0.0)
    first_bus = _register(controller)
    await first_bus.feed(_music(1.0))
    _schedule(
        controller,
        label="announcement:one",
        kind=ProgramKind.ANNOUNCEMENT,
    )
    # the listener reached its end, but nobody read the ledger since
    await first_bus.feed(_music(10.0))
    _hear(controller, 6.0)

    _register(controller)

    assert [(state.label, state.state) for state in controller.overlay_state(QUEUE_ID)] == [
        ("announcement:one", "aired")
    ]


async def test_a_high_ranking_lead_is_dropped_when_the_bus_is_replaced() -> None:
    """Whether a program may be anchored again follows its kind, not its numeric rank."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="lead", start_in_seconds=5.0, kind=ProgramKind.LEAD)

    _register(controller)

    assert [
        (state.label, state.state, state.reason) for state in controller.overlay_state(QUEUE_ID)
    ] == [("lead", "dropped", "bus_replaced")]


def test_an_unknown_queue_reports_no_bus() -> None:
    """A queue this controller never streamed has no bus, and no ledger is invented for it."""
    controller = _controller(audible=0.0)
    _register(controller)

    result = controller.schedule_overlay("unknown-queue", _voice(1.0), label="term")

    assert result.scheduled is False
    assert result.reason == "no_bus"
    assert result.frames == 0
    assert controller.overlay_state("unknown-queue") == []


async def test_scheduling_after_the_bus_was_released_keeps_the_ledger() -> None:
    """A program offered after its session ended changes nothing that was placed before."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="before", start_in_seconds=5.0)
    controller.release_program_bus(QUEUE_ID, program_bus)
    after_release = controller.overlay_state(QUEUE_ID)

    result = _schedule(controller, label="after")

    assert result.scheduled is False
    assert result.reason == "no_bus"
    assert controller.overlay_state(QUEUE_ID) == after_release
    # the session that ended drops what it never played
    assert [(state.label, state.state, state.reason) for state in after_release] == [
        ("before", "dropped", "flow_ended")
    ]


async def test_two_queues_keep_their_own_bus_and_ledger() -> None:
    """A program on one queue is independent of the bus and ledger of the other."""
    controller = _controller(audible=0.0)
    first_bus = _register(controller, queue_id="queue-1")
    second_bus = _register(controller, queue_id="queue-2")
    await first_bus.feed(_music(1.0))
    await second_bus.feed(_music(1.0))
    _schedule(controller, label="first", start_in_seconds=5.0)
    controller.schedule_overlay("queue-2", _voice(1.0), label="second", start_in_seconds=5.0)

    controller.cancel_overlays("queue-1", reason="next")

    assert first_bus.active is False
    assert second_bus.active is True
    assert [state.label for state in controller.overlay_state("queue-1")] == ["first"]
    assert [state.label for state in controller.overlay_state("queue-2")] == ["second"]
    assert [state.state for state in controller.overlay_state("queue-1")] == ["dropped"]
    assert [state.state for state in controller.overlay_state("queue-2")] == ["scheduled"]


def test_registered_replaced_and_released_buses_are_announced_to_a_listener() -> None:
    """A listener hears every bus that starts, supersedes another and ends."""
    controller = _controller(audible=0.0)
    events: list[Any] = []
    controller.add_program_bus_listener(events.append)
    first = _register(controller)
    second = _register(controller)
    # the superseded session does not take the new bus with it
    controller.release_program_bus(QUEUE_ID, first)
    controller.release_program_bus(QUEUE_ID, second)

    assert [(event.queue_id, event.change) for event in events] == [
        (QUEUE_ID, "registered"),
        (QUEUE_ID, "replaced"),
        (QUEUE_ID, "released"),
    ]


def test_releasing_the_published_bus_is_announced() -> None:
    """The session that ends withdraws its own bus, and that is announced."""
    controller = _controller(audible=0.0)
    events: list[Any] = []
    controller.add_program_bus_listener(events.append)
    program_bus = _register(controller)

    controller.release_program_bus(QUEUE_ID, program_bus)

    assert [event.change for event in events] == ["registered", "released"]


def test_a_listener_can_be_removed_again() -> None:
    """Unsubscribing stops the events for that listener."""
    controller = _controller(audible=0.0)
    events: list[Any] = []
    remove = controller.add_program_bus_listener(events.append)
    remove()

    _register(controller)

    assert events == []


async def test_a_replaced_bus_is_closed_before_the_new_one_is_published() -> None:
    """The old bus gives its programs away while it is still the published one."""
    controller = _controller(audible=0.0)
    old_bus = _register(controller)
    await old_bus.feed(_music(1.0))
    _schedule(controller, label="term", start_in_seconds=5.0)
    new_bus = FlowProgramBus(TEST_PCM_FORMAT, MagicMock())
    assert (
        new_bus.schedule(_voice(1.0), int(5 * SECOND), label="term", priority=ProgramPriority.TERM)
        is not None
    )

    controller.register_program_bus(QUEUE_ID, new_bus)

    # the term of the old session is cancelled on the old bus ...
    assert old_bus.scheduled_voice_frames("term") is None
    # ... and a same-named program of the new session is untouched
    assert new_bus.scheduled_voice_frames("term") is not None


async def test_a_bus_change_keeps_an_aired_announcement_aired_and_orphans_the_queued_one() -> None:
    """One queue, one bus change: what was heard stays heard, what was not is recoverable."""
    controller = _controller(audible=0.0)
    first_bus = _register(controller)
    await first_bus.feed(_music(1.0))
    _schedule(controller, label="announcement:aired", kind=ProgramKind.ANNOUNCEMENT)
    _schedule(controller, label="announcement:queued", kind=ProgramKind.ANNOUNCEMENT)
    # the first is heard to its end, the second is still to come
    await first_bus.feed(_music(10.0))
    _hear(controller, 3.0)

    _register(controller)

    states = {
        state.label: (state.state, state.reason) for state in controller.overlay_state(QUEUE_ID)
    }
    assert states["announcement:aired"] == ("aired", None)
    assert states["announcement:queued"] == ("orphaned", None)


async def test_releasing_a_bus_cancels_its_programs_before_it_is_withdrawn() -> None:
    """The ended bus is still the published one while its programs are cancelled."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="term", start_in_seconds=5.0)

    controller.release_program_bus(QUEUE_ID, program_bus)

    assert program_bus.scheduled_voice_frames("term") is None
    assert controller.program_bus_active(QUEUE_ID) is False


async def test_a_removed_queue_ends_its_bus_and_forgets_its_level() -> None:
    """A queue that is gone leaves no live bus and no level a new queue could inherit."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    controller.set_program_levels(QUEUE_ID, voice_db=-6, duck_db=-12)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="term", start_in_seconds=5.0)

    controller.on_queue_removed(QUEUE_ID)

    # what the removed queue still had is stopped on its bus, not left to air
    assert program_bus.scheduled_voice_frames("term") is None
    assert controller.program_bus_active(QUEUE_ID) is False
    assert controller.program_levels(QUEUE_ID) == ProgramLevels()
    assert controller.overlay_state(QUEUE_ID) == []


async def test_a_queue_that_comes_back_with_the_same_id_starts_empty() -> None:
    """Nothing of the queue that was removed is read back by the queue that follows it."""
    controller = _controller(audible=0.0)
    _register(controller)
    controller.set_program_levels(QUEUE_ID, voice_db=-6, duck_db=-12)
    controller.on_queue_removed(QUEUE_ID)

    program_bus = _register(controller)

    assert controller._program_buses.get(QUEUE_ID) is program_bus
    assert controller.program_bus_active(QUEUE_ID) is True
    # the level belonged to the queue that is gone, so the new one mixes as a bus always did
    assert controller.program_levels(QUEUE_ID) == ProgramLevels()
    assert controller.overlay_state(QUEUE_ID) == []

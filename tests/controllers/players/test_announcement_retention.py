"""Tests for keeping rendered announcements until they aired and re-anchoring them."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

from music_assistant.controllers.players.announcement_overlays import (
    RETENTION_SECONDS,
    OverlayAnnouncementRetention,
    new_announcement_label,
)
from music_assistant.controllers.streams.program_bus_registry import (
    ProgramBusChange,
    ProgramBusEvent,
)
from music_assistant.controllers.streams.program_schedule import (
    ScheduleReason,
    ScheduleResult,
)

QUEUE_ID = "queue-1"


class _FakeStreams:
    """A streams controller that records placements and reports a ledger."""

    def __init__(self, *, scheduled: bool = True) -> None:
        self.placed: list[tuple[str, str]] = []
        self.states: dict[str, list[SimpleNamespace]] = {}
        self.listeners: list[Any] = []
        self.aired_listeners: list[Any] = []
        self.scheduled = scheduled

    def add_program_bus_listener(self, listener: Any) -> Any:
        """Record a bus listener and hand back its remover."""
        self.listeners.append(listener)
        return lambda: self.listeners.remove(listener)

    def add_program_aired_listener(self, listener: Any) -> Any:
        """Record an aired listener and hand back its remover."""
        self.aired_listeners.append(listener)
        return lambda: self.aired_listeners.remove(listener)

    def overlay_state(self, queue_id: str) -> list[SimpleNamespace]:
        """Return the ledger rows of the queue, with the fields the read always carries."""
        return [
            SimpleNamespace(
                label=row.label,
                state=row.state,
                reason=getattr(row, "reason", None),
                playing=getattr(row, "playing", False),
            )
            for row in self.states.get(queue_id, ())
        ]

    def schedule_overlay(
        self,
        queue_id: str,
        pcm: bytes,
        *,
        label: str,
        start_in_seconds: float = 0.0,
        priority: Any = None,
        kind: Any = None,
    ) -> ScheduleResult:
        """Record a placement and report it as scheduled."""
        self.placed.append((queue_id, label))
        if not self.scheduled:
            return ScheduleResult(False, 0, 0, 0.0, 0.0, ScheduleReason.NO_BUS)
        self.states.setdefault(queue_id, []).append(SimpleNamespace(label=label, state="scheduled"))
        return ScheduleResult(True, 0, 0, 0.0, 0.0, None)

    def emit(self, change: ProgramBusChange, queue_id: str = QUEUE_ID) -> None:
        """Announce a bus change to every listener."""
        for listener in list(self.listeners):
            listener(ProgramBusEvent(queue_id=queue_id, change=change))

    def emit_aired(self, label: str, queue_id: str = QUEUE_ID) -> None:
        """Tell the listeners that the ledger aired the clip."""
        for listener in list(self.aired_listeners):
            listener(queue_id, label)


class _Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _retention(
    streams: _FakeStreams,
    clock: _Clock | None = None,
    logger: MagicMock | None = None,
    **options: Any,
) -> OverlayAnnouncementRetention:
    """Build a retention on the fake streams and subscribe it."""
    return OverlayAnnouncementRetention(
        cast("Any", streams), logger or MagicMock(), clock or _Clock(), **options
    )


def test_an_orphaned_announcement_is_anchored_again_exactly_once() -> None:
    """A replaced bus hands an orphaned announcement to the new one, and only once."""
    streams = _FakeStreams()
    retention = _retention(streams)
    label = new_announcement_label()
    retention.remember(QUEUE_ID, label, b"voice")
    streams.states[QUEUE_ID] = [SimpleNamespace(label=label, state="orphaned")]

    streams.emit(ProgramBusChange.REPLACED)
    streams.emit(ProgramBusChange.REPLACED)

    assert streams.placed == [(QUEUE_ID, label)]
    assert retention.retained_labels() == (label,)


def test_an_aired_announcement_is_forgotten_not_anchored_again() -> None:
    """What the listener already heard is never played a second time."""
    streams = _FakeStreams()
    retention = _retention(streams)
    label = new_announcement_label()
    retention.remember(QUEUE_ID, label, b"voice")
    streams.states[QUEUE_ID] = [SimpleNamespace(label=label, state="aired")]

    streams.emit(ProgramBusChange.REPLACED)

    assert streams.placed == []
    assert retention.retained_labels() == ()


def test_an_announcement_past_its_retention_is_not_anchored_again() -> None:
    """A clip that was rendered too long ago is dropped instead of spoken late."""
    streams = _FakeStreams()
    clock = _Clock()
    retention = _retention(streams, clock)
    label = new_announcement_label()
    retention.remember(QUEUE_ID, label, b"voice")
    streams.states[QUEUE_ID] = [SimpleNamespace(label=label, state="orphaned")]
    clock.now = 301.0

    streams.emit(ProgramBusChange.REPLACED)

    assert streams.placed == []
    assert retention.retained_labels() == ()


def test_a_session_that_ends_and_one_that_starts_anchor_the_announcement_once() -> None:
    """The release itself has no bus to anchor on; the next registration does."""
    streams = _FakeStreams()
    retention = _retention(streams)
    label = new_announcement_label()
    retention.remember(QUEUE_ID, label, b"voice")

    streams.emit(ProgramBusChange.RELEASED)
    assert streams.placed == []

    streams.states[QUEUE_ID] = [SimpleNamespace(label=label, state="orphaned")]
    streams.emit(ProgramBusChange.REGISTERED)

    assert streams.placed == [(QUEUE_ID, label)]


def test_a_placement_without_a_bus_is_kept_for_the_next_one() -> None:
    """A bus that vanished between the event and the placement does not lose the clip."""
    streams = _FakeStreams(scheduled=False)
    retention = _retention(streams)
    label = new_announcement_label()
    retention.remember(QUEUE_ID, label, b"voice")
    streams.states[QUEUE_ID] = [SimpleNamespace(label=label, state="orphaned")]

    streams.emit(ProgramBusChange.REPLACED)

    assert retention.retained_labels() == (label,)


def test_every_placement_gets_its_own_label() -> None:
    """Two announcements are told apart by their labels, so both can be tracked."""
    first = new_announcement_label()
    second = new_announcement_label()

    assert first != second
    assert first.startswith("announcement:")
    assert second.startswith("announcement:")


def test_the_retention_listens_to_the_bus_changes() -> None:
    """The retention subscribes once and can stop again."""
    streams = _FakeStreams()
    retention = _retention(streams)

    assert len(streams.listeners) == 1
    retention.close()

    assert streams.listeners == []


def test_an_aired_announcement_is_forgotten_without_a_bus_change() -> None:
    """The retention frees the PCM as soon as the ledger reports the clip was heard."""
    streams = _FakeStreams()
    retention = _retention(streams)
    label = new_announcement_label()
    retention.remember(QUEUE_ID, label, b"voice")
    streams.states[QUEUE_ID] = [SimpleNamespace(label=label, state="aired")]

    assert retention.retained_labels() == ()


async def test_a_clip_past_its_retention_is_released_without_a_new_placement() -> None:
    """A queue that renders a clip it never hears must not keep the bytes until the next one."""
    streams = _FakeStreams()
    clock = _Clock()
    # the tick waits SWEEP_SECONDS on its own, so the test drives the loop, not the clock
    retention = OverlayAnnouncementRetention(streams, MagicMock(), clock, sweep_seconds=0.0)
    retention.remember(QUEUE_ID, new_announcement_label(), b"voice")

    clock.now = RETENTION_SECONDS + 1.0
    # no second remember: the retention's own tick is what has to free the clip
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert retention._retained == {}
    assert retention._sweep_handle is None


async def test_the_sweep_stays_armed_while_a_clip_is_retained() -> None:
    """A clip inside its window keeps the tick going, and closing stops it."""
    streams = _FakeStreams()
    retention = _retention(streams)
    retention.remember(QUEUE_ID, new_announcement_label(), b"voice")

    retention._sweep()

    assert retention._sweep_handle is not None
    retention.close()
    assert retention._sweep_handle is None


def test_an_announcement_past_its_retention_is_forgotten_on_access() -> None:
    """A clip kept longer than the retention window is released, bus change or not."""
    streams = _FakeStreams()
    clock = _Clock()
    retention = _retention(streams, clock)
    label = new_announcement_label()
    retention.remember(QUEUE_ID, label, b"voice")
    clock.now = 301.0

    assert retention.retained_labels() == ()


def test_unheard_announcements_are_kept_above_the_soft_limit() -> None:
    """A burst of unheard clips is not silently discarded, and the other queue is untouched."""
    streams = _FakeStreams()
    retention = _retention(streams)
    labels = [new_announcement_label() for _index in range(17)]
    other = new_announcement_label()

    for label in labels:
        retention.remember(QUEUE_ID, label, b"voice")
    retention.remember("queue-2", other, b"voice")

    retained = retention.retained_labels()
    assert len(retained) == 18
    assert labels[0] in retained
    assert other in retained


def test_a_runaway_queue_is_bounded_by_the_hard_ceiling() -> None:
    """Only the last resort drops clips, and it says so."""
    streams = _FakeStreams()
    logger = MagicMock()
    retention = _retention(streams, logger=logger)

    for _index in range(65):
        retention.remember(QUEUE_ID, new_announcement_label(), b"voice")

    assert len(retention.retained_labels()) == 64
    assert any("not kept" in str(call.args[0]) for call in logger.warning.call_args_list)


def test_a_queue_that_outgrows_the_byte_budget_gives_up_its_oldest_clips() -> None:
    """The clip ceiling counts clips; the bytes a queue may hold are bounded on their own."""
    streams = _FakeStreams()
    logger = MagicMock()
    retention = _retention(streams, logger=logger, bytes_per_queue=100)
    labels = [new_announcement_label() for _index in range(12)]

    for label in labels:
        retention.remember(QUEUE_ID, label, b"0123456789")

    retained = retention.retained_labels()
    # the two oldest clips went, and what is left fits the budget
    assert retained == tuple(labels[2:])
    assert sum(len(b"0123456789") for _label in retained) <= 100
    assert any("bytes" in str(call.args[0]) for call in logger.warning.call_args_list)


def test_an_announcement_the_listener_hears_is_never_given_up_for_its_bytes() -> None:
    """A clip that is playing right now is the one a re-anchor exists for."""
    streams = _FakeStreams()
    clock = _Clock()
    retention = _retention(streams, clock, bytes_per_queue=15)
    playing = new_announcement_label()
    retention.remember(QUEUE_ID, playing, b"0123456789")
    streams.states[QUEUE_ID] = [SimpleNamespace(label=playing, state="scheduled", playing=True)]
    clock.now = 1.0
    waiting = new_announcement_label()
    retention.remember(QUEUE_ID, waiting, b"0123456789")

    # the budget is over with the second clip, and the older one is the one being heard
    assert retention.retained_labels() == (playing,)


def test_an_aired_announcement_is_freed_by_the_ledger_event() -> None:
    """The ledger says aired without a bus change, and the bytes go right away."""
    streams = _FakeStreams()
    retention = _retention(streams)
    label = new_announcement_label()
    retention.remember(QUEUE_ID, label, b"voice")

    streams.emit_aired(label)

    assert retention.retained_labels() == ()


def test_a_scheduled_clip_past_its_retention_is_forgotten_on_a_bus_change() -> None:
    """A pause that left the clip scheduled must not keep it forever across bus changes."""
    streams = _FakeStreams()
    clock = _Clock()
    retention = _retention(streams, clock)
    label = new_announcement_label()
    retention.remember(QUEUE_ID, label, b"voice")
    streams.states[QUEUE_ID] = [SimpleNamespace(label=label, state="scheduled")]
    clock.now = 301.0

    streams.emit(ProgramBusChange.REPLACED)

    assert retention.retained_labels() == ()


def test_forget_queue_releases_only_that_queue() -> None:
    """A queue that is gone releases its clips, the other queue keeps its own."""
    streams = _FakeStreams()
    retention = _retention(streams)
    first = new_announcement_label()
    second = new_announcement_label()
    retention.remember(QUEUE_ID, first, b"voice")
    retention.remember("queue-2", second, b"voice")

    retention.forget_queue(QUEUE_ID)

    assert retention.retained_labels() == (second,)


def test_close_drops_every_retained_announcement_and_the_listener() -> None:
    """The controller's shutdown releases the retained audio and unsubscribes."""
    streams = _FakeStreams()
    retention = _retention(streams)
    retention.remember(QUEUE_ID, new_announcement_label(), b"voice")

    retention.close()

    assert retention.retained_labels() == ()
    assert streams.listeners == []


def test_a_later_announcement_releases_a_clip_that_already_aired() -> None:
    """A new placement reads the ledger, so an earlier clip that aired is freed right then."""
    streams = _FakeStreams()
    retention = _retention(streams)
    first = new_announcement_label()
    retention.remember(QUEUE_ID, first, b"voice")
    # the ledger will report the first as aired when the retention next reads it
    streams.states[QUEUE_ID] = [SimpleNamespace(label=first, state="aired")]

    retention.remember(QUEUE_ID, new_announcement_label(), b"voice")

    assert first not in retention.retained_labels()


def test_a_bus_event_for_one_queue_leaves_the_other_queue_alone() -> None:
    """The bus of one queue changing does not touch the clips retained for another."""
    streams = _FakeStreams()
    retention = _retention(streams)
    first = new_announcement_label()
    second = new_announcement_label()
    retention.remember("queue-1", first, b"voice")
    retention.remember("queue-2", second, b"voice")
    streams.states["queue-1"] = [SimpleNamespace(label=first, state="orphaned")]
    streams.states["queue-2"] = [SimpleNamespace(label=second, state="orphaned")]

    streams.emit(ProgramBusChange.REPLACED, queue_id="queue-1")

    assert streams.placed == [("queue-1", first)]
    assert second in retention.retained_labels()


def test_an_evicted_announcement_is_not_anchored_again() -> None:
    """A placement the ledger gave up must not play a second time on the new bus."""
    streams = _FakeStreams()
    retention = _retention(streams)
    label = new_announcement_label()
    retention.remember(QUEUE_ID, label, b"voice")
    streams.states[QUEUE_ID] = [
        SimpleNamespace(label=label, state="orphaned", reason="ledger_evicted")
    ]

    streams.emit(ProgramBusChange.REPLACED)

    assert streams.placed == []
    assert retention.retained_labels() == ()


def test_an_orphaned_announcement_that_was_heard_is_not_anchored_again() -> None:
    """An older orphaned row does not hide that the clip already aired."""
    streams = _FakeStreams()
    retention = _retention(streams)
    label = new_announcement_label()
    retention.remember(QUEUE_ID, label, b"voice")
    streams.states[QUEUE_ID] = [
        SimpleNamespace(label=label, state="orphaned", reason=None),
        SimpleNamespace(label=label, state="aired", reason=None),
    ]

    streams.emit(ProgramBusChange.REPLACED)

    assert streams.placed == []
    assert retention.retained_labels() == ()

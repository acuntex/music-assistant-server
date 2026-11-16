"""Per-queue ledger of the programs placed on a queue's live flow stream."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from mashumaro import DataClassDictMixin

from music_assistant.controllers.streams.program_mixer import FlowProgramBus
from music_assistant.controllers.streams.program_schedule import (
    ProgramKind,
    ProgramPriority,
    ScheduleReason,
)

# how many programs the ledger remembers per queue
ENTRIES_PER_QUEUE = 16


class OverlayProgramState(StrEnum):
    """State of one program in a queue's program ledger."""

    SCHEDULED = "scheduled"
    AIRED = "aired"
    # the bus was replaced before the program was heard; it may still be anchored again
    ORPHANED = "orphaned"
    DROPPED = "dropped"


@dataclass(frozen=True)
class OverlayState(DataClassDictMixin):
    """One program in a queue's ledger, as reported by overlay_state."""

    label: str
    state: OverlayProgramState
    start_frame: int
    frames: int
    reason: str | None
    priority: ProgramPriority
    # the listener is inside this program's window at the moment of the read, so it is heard
    # right now and must not be given up: it is the one a re-anchor exists for
    playing: bool = False


@dataclass
class LedgerEntry:
    """A program the streams controller placed or refused, kept for the queue's ledger."""

    label: str
    state: OverlayProgramState
    start_frame: int
    frames: int
    reason: str | None
    priority: ProgramPriority
    kind: ProgramKind
    bus: FlowProgramBus | None
    deadline_frame: int
    # seconds the program was planned to start before the boundary it belongs to
    head_seconds: float = 0.0
    # seconds the bus had to move it past the boundary it was planned for
    late_by_seconds: float = 0.0
    # whether the last read found the listener inside this program's window
    playing: bool = False


class ProgramLedger:
    """
    Remember the last programs of every queue and what became of them.

    The ledger is the single source of ``overlay_state`` and the only place that decides
    when a program has aired: its deadline is a frame of the session's stream, so a pause
    holds it and a bus that produced ahead says nothing about what was heard.
    """

    def __init__(
        self,
        max_entries: int = ENTRIES_PER_QUEUE,
        on_dropped: Callable[[str, str, int, str], None] | None = None,
        on_aired: Callable[[str, str, int], None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the ledger.

        :param max_entries: How many programs to remember per queue.
        :param on_dropped: Called with (queue_id, label, start_frame, reason) when a program
            that was still to be heard is dropped, so a live bus can stop exactly that one.
        :param on_aired: Called with (queue_id, label, start_frame) when a program was heard
            to its end, so whoever holds its audio can release it.
        :param logger: Logger for a program the bounded history had to give up.
        """
        # a ledger of zero would have nowhere to keep a program
        self._max_entries = max(1, max_entries)
        self._on_dropped = on_dropped
        self._on_aired = on_aired
        self._logger = logger
        self._queues: dict[str, deque[LedgerEntry]] = {}

    def entries(self, queue_id: str) -> tuple[LedgerEntry, ...]:
        """
        Return the recorded programs of the queue, oldest first.

        :param queue_id: Queue whose entries to read.
        """
        return tuple(self._queues.get(queue_id, ()))

    def forget_queue(self, queue_id: str) -> None:
        """
        Forget everything the ledger kept for a queue that is gone.

        :param queue_id: Queue that was removed.
        """
        self._queues.pop(queue_id, None)

    def record_scheduled(
        self,
        queue_id: str,
        *,
        label: str,
        start_frame: int,
        frames: int,
        priority: ProgramPriority,
        kind: ProgramKind,
        reason: str | None,
        bus: FlowProgramBus,
        head_seconds: float = 0.0,
        late_by_seconds: float = 0.0,
    ) -> None:
        """
        Record a program that was placed on the queue's stream.

        :param queue_id: Queue the program was placed on.
        :param label: Label the program was scheduled with.
        :param start_frame: Frame the program was anchored at.
        :param frames: Frames the bus really plays of it, which is fewer than the program was
            offered with when it had to drop the front of it. The caller takes it from its own
            placement, so a second program under the same label can not stand in for it.
        :param priority: Priority the program claimed.
        :param kind: What the program is for.
        :param reason: Why it was moved, if it was.
        :param bus: The bus it was placed on.
        :param head_seconds: Seconds the program was planned to start before its boundary.
        :param late_by_seconds: Seconds the bus had to move it past that boundary.
        """
        # a bus that had to move the anchor forward dropped the front of the program, so it
        # is aired where the audio that is left ends, and that is the length it keeps in the
        # ledger and in the view as well
        self._append(
            queue_id,
            LedgerEntry(
                label=label,
                state=OverlayProgramState.SCHEDULED,
                start_frame=start_frame,
                frames=frames,
                reason=reason,
                priority=priority,
                kind=kind,
                bus=bus,
                deadline_frame=start_frame + frames,
                head_seconds=head_seconds,
                late_by_seconds=late_by_seconds,
            ),
        )

    def record_dropped(
        self,
        queue_id: str,
        *,
        label: str,
        frames: int,
        priority: ProgramPriority,
        kind: ProgramKind,
        reason: str,
        start_frame: int = 0,
        bus: FlowProgramBus | None = None,
    ) -> None:
        """
        Record a program that was not placed on the queue's stream.

        :param queue_id: Queue the program was offered to.
        :param label: Label the program was offered with.
        :param frames: Number of frames the program has.
        :param priority: Priority the program claimed.
        :param kind: What the program is for.
        :param reason: Why it was not placed.
        :param start_frame: Frame it would have been anchored at, if one was planned.
        :param bus: The bus it was offered to, if there was one.
        """
        self._append(
            queue_id,
            LedgerEntry(
                label=label,
                state=OverlayProgramState.DROPPED,
                start_frame=start_frame,
                frames=frames,
                reason=reason,
                priority=priority,
                kind=kind,
                bus=bus,
                deadline_frame=start_frame + frames,
            ),
        )

    def states(
        self, queue_id: str, heard_frame: Callable[[FlowProgramBus], int | None]
    ) -> list[OverlayState]:
        """
        Return the programs of the queue after bringing their deadlines up to date.

        :param queue_id: Queue whose entries to read.
        :param heard_frame: Reader for the frame the listener has heard on a program's bus.
        """
        self.refresh(queue_id, heard_frame)
        return [
            OverlayState(
                label=entry.label,
                state=entry.state,
                start_frame=entry.start_frame,
                frames=entry.frames,
                reason=entry.reason,
                priority=entry.priority,
                playing=entry.playing,
            )
            for entry in self._queues.get(queue_id, ())
        ]

    def refresh(self, queue_id: str, heard_frame: Callable[[FlowProgramBus], int | None]) -> None:
        """
        Mark the programs the listener has heard to their end as aired.

        A position that can not be told leaves the deadline where it is, so a pause pauses
        it and a session without a heard position never turns a program into aired.

        :param queue_id: Queue whose entries to update.
        :param heard_frame: Reader for the frame the listener has heard on a program's bus.
        """
        for entry in self._queues.get(queue_id, ()):
            if entry.state is not OverlayProgramState.SCHEDULED or entry.bus is None:
                entry.playing = False
                continue
            heard = heard_frame(entry.bus)
            # the program is being heard while the position is inside its window, not only
            # once it is over: a bus that is replaced right then cuts it, it does not
            # replay it
            entry.playing = heard is not None and entry.start_frame <= heard < entry.deadline_frame
            if heard is not None and heard >= entry.deadline_frame:
                entry.state = OverlayProgramState.AIRED
                entry.playing = False
                if self._on_aired is not None:
                    self._on_aired(queue_id, entry.label, entry.start_frame)

    def mark_dropped(
        self, queue_id: str, label: str, reason: str, *, start_frame: int | None = None
    ) -> bool:
        """
        Mark the still-scheduled programs with the label (and anchor) as dropped.

        :param queue_id: Queue whose entries to update.
        :param label: Label of the programs to drop.
        :param reason: Why they are dropped.
        :param start_frame: Anchor to match, or None for every program with the label.
        :return: Whether any program was still scheduled.
        """
        dropped: list[LedgerEntry] = []
        for entry in self._queues.get(queue_id, ()):
            if entry.label != label or entry.state is not OverlayProgramState.SCHEDULED:
                continue
            if start_frame is not None and entry.start_frame != start_frame:
                continue
            entry.state = OverlayProgramState.DROPPED
            entry.reason = reason
            dropped.append(entry)
        for entry in dropped:
            self._report_dropped(queue_id, entry.label, entry.start_frame, reason)
        return bool(dropped)

    def drop_queue(
        self,
        queue_id: str,
        reason: str,
        *,
        kinds: frozenset[ProgramKind] | None = None,
    ) -> None:
        """
        Mark every program of the queue that is still to be heard as dropped.

        :param queue_id: Queue whose programs are dropped.
        :param reason: Why they are dropped.
        :param kinds: Only programs of these kinds are dropped, or None for all of them.
        """
        for entry in self._queues.get(queue_id, ()):
            if entry.state is not OverlayProgramState.SCHEDULED:
                continue
            if kinds is not None and entry.kind not in kinds:
                continue
            entry.state = OverlayProgramState.DROPPED
            entry.reason = reason
            self._report_dropped(queue_id, entry.label, entry.start_frame, reason)

    def close_bus(
        self,
        queue_id: str,
        bus: FlowProgramBus,
        *,
        terminal_state: OverlayProgramState,
        terminal_reason: str,
        keep_announcements: bool = False,
        heard_frame: Callable[[FlowProgramBus], int | None] | None = None,
    ) -> None:
        """
        End what a bus still had to play, keeping announcements recoverable if asked.

        An announcement the listener was already hearing when the bus went away ends as
        aired: its audio was cut at that point and is never spoken again, so the ledger says
        what was heard instead of reporting it as dropped or of letting it be anchored again
        on the next bus.

        :param queue_id: Queue the bus belonged to.
        :param bus: The bus that ended or was replaced.
        :param terminal_state: State for programs that may not be anchored again.
        :param terminal_reason: Why they end.
        :param keep_announcements: Whether announcements stay non-terminal (orphaned).
        :param heard_frame: Reader for the frame the listener has heard on a bus, or None
            when no position can be told; without it nothing counts as already heard.
        """
        for entry in self._queues.get(queue_id, ()):
            if entry.state is not OverlayProgramState.SCHEDULED or entry.bus is not bus:
                continue
            if entry.kind == ProgramKind.ANNOUNCEMENT and heard_frame is not None:
                heard = heard_frame(entry.bus)
                # at least one frame of it was heard, so the clip was playing when the bus
                # went away and the jump cuts it: it aired, and its audio may be released
                if heard is not None and heard > entry.start_frame:
                    entry.state = OverlayProgramState.AIRED
                    entry.reason = None
                    if self._on_aired is not None:
                        self._on_aired(queue_id, entry.label, entry.start_frame)
                    continue
            if keep_announcements and entry.kind == ProgramKind.ANNOUNCEMENT:
                entry.state = OverlayProgramState.ORPHANED
                entry.reason = None
                continue
            entry.state = terminal_state
            entry.reason = terminal_reason
            if terminal_state is OverlayProgramState.DROPPED:
                self._report_dropped(queue_id, entry.label, entry.start_frame, terminal_reason)

    def _report_dropped(self, queue_id: str, label: str, start_frame: int, reason: str) -> None:
        """Tell the hook that a program the queue was going to hear is dropped."""
        if self._on_dropped is not None:
            self._on_dropped(queue_id, label, start_frame, reason)

    def _queue(self, queue_id: str) -> deque[LedgerEntry]:
        """Return the queue's entries, creating the ledger on first use."""
        ledger = self._queues.get(queue_id)
        if ledger is None:
            ledger = deque()
            self._queues[queue_id] = ledger
        return ledger

    def _append(self, queue_id: str, entry: LedgerEntry) -> None:
        """Remember a program, giving up the oldest one when the history is full."""
        ledger = self._queue(queue_id)
        if len(ledger) >= self._max_entries:
            self._make_room(queue_id, ledger)
        ledger.append(entry)

    def _make_room(self, queue_id: str, ledger: deque[LedgerEntry]) -> None:
        """Give up one program so a new one fits, without losing a queued announcement."""
        # a program that already ended is the cheapest to give up; one that may still be
        # heard on a live bus is stopped and reported before its line goes
        for index, recorded in enumerate(ledger):
            if recorded.state is not OverlayProgramState.SCHEDULED:
                if (
                    recorded.state is OverlayProgramState.ORPHANED
                    and recorded.reason != ScheduleReason.LEDGER_EVICTED
                ):
                    # an announcement that lost its bus may still be heard on a live one;
                    # one that was already cancelled at an overflow must not be again
                    self._report_dropped(
                        queue_id,
                        recorded.label,
                        recorded.start_frame,
                        ScheduleReason.LEDGER_EVICTED,
                    )
                    self._log_eviction(
                        queue_id, recorded, "orphaned", ScheduleReason.LEDGER_EVICTED
                    )
                del ledger[index]
                return
        # then a program that is not an announcement: its bus stops it through the hook
        for index, recorded in enumerate(ledger):
            if recorded.kind != ProgramKind.ANNOUNCEMENT:
                del ledger[index]
                recorded.state = OverlayProgramState.DROPPED
                recorded.reason = ScheduleReason.LEDGER_EVICTED
                self._report_dropped(
                    queue_id, recorded.label, recorded.start_frame, recorded.reason
                )
                self._log_eviction(queue_id, recorded, "dropped", recorded.reason)
                return
        # only queued announcements are left: the oldest stays visible as orphaned, so a
        # re-anchor can still find it; the next program reclaims it as an ended entry
        oldest = ledger[0]
        # stopping it now keeps the ledger honest: it will not air as an "orphaned" program,
        # and the retention releases its audio through the ledger_evicted guard
        self._report_dropped(
            queue_id, oldest.label, oldest.start_frame, ScheduleReason.LEDGER_EVICTED
        )
        oldest.state = OverlayProgramState.ORPHANED
        # this one was never heard, so it must not be anchored a second time
        oldest.reason = ScheduleReason.LEDGER_EVICTED
        self._log_eviction(
            queue_id, oldest, "orphaned", "cancelled at overflow, not anchored again"
        )

    def _log_eviction(self, queue_id: str, entry: LedgerEntry, state: str, reason: str) -> None:
        """Report a program the bounded history had to give up."""
        if self._logger is None:
            return
        self._logger.warning(
            "PROGRAM-BUS %s on queue %s ended as %s (%s): the ledger is full",
            entry.label,
            queue_id,
            state,
            reason,
        )

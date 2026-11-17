"""The terms a queue's AI DJ plans, and the store that keeps them for the session."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from music_assistant.controllers.streams.program_schedule import ScheduleReason

if TYPE_CHECKING:
    from music_assistant_models.media_items import AudioFormat

    from .models import PlannedSection

# how many terms one queue remembers. a term is a boundary of the running session, so the
# bound only has to hold a long show, not the history: what aired is the ledger's business
TERMS_PER_QUEUE = 16

# the boundary of an item: the item itself and what follows it, or None for the queue's end
BoundaryKey = tuple[str, str | None]


class TermState(StrEnum):
    """Where a term is in its life, from a decision to a boundary that passed."""

    DECIDED = "decided"
    RENDERING = "rendering"
    SCHEDULED = "scheduled"
    AIRED = "aired"
    DROPPED = "dropped"


class BoundaryState(StrEnum):
    """What the store settled about one boundary, the only owner of that answer."""

    # nobody has looked at it yet, or its term was merely missed and may be decided again
    OPEN = "open"
    # the store decided it: a term speaks there, or a pass found the host with nothing to say
    DECIDED = "decided"
    # no pass can ever speak it, for the reason the store keeps
    UNREACHABLE = "unreachable"


class TermDropReason(StrEnum):
    """Why a term never reached the listener, as the provider knows it."""

    TOO_LATE = "too_late"
    SKIPPED = "skipped"
    SEEKED = "seeked"
    FLOW_ENDED = "flow_ended"
    BUS_REPLACED = "bus_replaced"
    NO_BUS = "no_bus"
    UNKNOWN_BOUNDARY = "unknown_boundary"
    NO_DURATION = "no_duration"
    NO_SUCCESSOR = "no_successor"
    RENDER_FAILED = "render_failed"


TERMINAL_STATES = frozenset({TermState.AIRED, TermState.DROPPED})

# the reasons that describe the boundary itself rather than the moment it was tried: a
# boundary that fell for one of them falls for it again, so it is closed instead of retried
PERMANENT_DROP_REASONS = frozenset(
    {
        str(TermDropReason.NO_DURATION),
        str(TermDropReason.UNKNOWN_BOUNDARY),
        str(TermDropReason.NO_SUCCESSOR),
        str(ScheduleReason.FORMAT),
    }
)


def term_key(queue_item_id: str, successor_id: str | None) -> BoundaryKey:
    """
    Return the stable identity of a term's boundary.

    :param queue_item_id: The item the term speaks over.
    :param successor_id: The item after it, or None for the end of the queue.
    """
    return (queue_item_id, successor_id)


@dataclass
class ProgramTerm:
    """One spoken term planned for the end of a queue item."""

    queue_item_id: str
    successor_id: str | None
    section_ids: tuple[str, ...]
    # seconds of the term that play before its boundary: the host's lead, cut to the clip,
    # or the whole clip when it may not carry over the boundary
    head_seconds: float
    # the section may start over the outgoing item's tail
    allow_pre: bool = False
    # the clip may run on over the next item. without it the term has to end before the
    # boundary, and a boundary too close for the whole clip is no boundary for it
    allow_post: bool = False
    state: TermState = TermState.DECIDED
    dropped_reason: str | None = None
    # frame the streams controller anchored the term at, once it was placed
    scheduled_frame: int | None = None
    # size of the rendered clip, never its audio
    pcm_len: int = 0
    # seconds from the listener to the term's start, as planned
    delay_seconds: float | None = None
    # seconds after the scheduled start by which the term has been heard to its end
    seconds_left: float | None = None
    # the position in the item the timing was last computed from, to spot a seek
    last_elapsed: float | None = None
    last_seen_at: float | None = None
    # a bus was live for this term at least once, so losing it is a replacement
    bus_seen: bool = False
    # loop time the term first looked for a bus it never saw, so it can wait out its grace
    bus_missing_since: float | None = None
    # a clip written before it could be placed, held so a resume does not pay for it again
    rendered_pcm: bytes | None = field(default=None, repr=False, compare=False)
    # the flow format the held clip was rendered in
    rendered_format: AudioFormat | None = field(default=None, repr=False, compare=False)
    # the sections this term renders, kept out of the identity and out of the comparison
    sections: tuple[PlannedSection, ...] = field(default=(), compare=False, repr=False)

    @property
    def key(self) -> BoundaryKey:
        """Return the boundary this term belongs to."""
        return term_key(self.queue_item_id, self.successor_id)

    @property
    def is_terminal(self) -> bool:
        """Return whether the term has ended, one way or the other."""
        return self.state in TERMINAL_STATES

    def mark_rendering(self) -> None:
        """Record that the term's clip is being rendered."""
        self.state = TermState.RENDERING

    def mark_decided(self) -> None:
        """Return the term to deciding, for a render that landed where nothing is heard."""
        if self.is_terminal:
            return
        self.state = TermState.DECIDED

    def keep_rendered(self, pcm: bytes, pcm_format: AudioFormat | None = None) -> None:
        """
        Hold a rendered clip that could not be placed yet.

        :param pcm: The rendered clip in the flow's format.
        :param pcm_format: The flow format the clip was rendered in.
        """
        if self.is_terminal:
            return
        self.rendered_pcm = pcm
        self.rendered_format = pcm_format
        self.state = TermState.DECIDED

    def mark_scheduled(
        self,
        *,
        start_frame: int,
        pcm_len: int,
        delay_seconds: float,
        seconds_left: float,
    ) -> None:
        """
        Record that the streams controller took the term.

        :param start_frame: Frame the term was anchored at.
        :param pcm_len: Size of the rendered clip in bytes.
        :param delay_seconds: Seconds from the listener to the term's start.
        :param seconds_left: Seconds of stream until the term has been heard.
        """
        self.state = TermState.SCHEDULED
        self.scheduled_frame = start_frame
        self.pcm_len = pcm_len
        self.rendered_pcm = None
        self.rendered_format = None
        self.delay_seconds = delay_seconds
        self.seconds_left = max(0.0, seconds_left)

    def mark_aired(self) -> None:
        """Record that the listener heard the term to its end."""
        if self.is_terminal:
            return
        self.state = TermState.AIRED
        self.dropped_reason = None
        self.seconds_left = 0.0
        self.rendered_pcm = None
        self.rendered_format = None

    def mark_dropped(self, reason: str | ScheduleReason) -> None:
        """
        Record why the term never aired.

        :param reason: Why the term ended without being heard.
        """
        if self.is_terminal:
            return
        self.state = TermState.DROPPED
        self.dropped_reason = str(reason)
        self.seconds_left = None
        self.rendered_pcm = None
        self.rendered_format = None


@dataclass
class BoundaryRecord:
    """The one entry a store keeps for a boundary, whatever was decided to speak there."""

    queue_item_id: str
    successor_id: str | None
    state: BoundaryState = BoundaryState.OPEN
    # why no pass may speak here, for a boundary that was found unservable
    reason: str | None = None
    # the term that speaks (or last spoke) at this boundary, None for a boundary a pass
    # looked at and left silent
    term: ProgramTerm | None = None
    # where the boundary plays, the one order a bounded store gives its terms up in
    position: int = 0

    @property
    def key(self) -> BoundaryKey:
        """Return the boundary this record belongs to."""
        return term_key(self.queue_item_id, self.successor_id)


class TermStore:
    """
    The boundaries of one queue's DJ session, bounded like the streams ledger.

    The store is the only place that answers what a boundary's state is: a boundary it
    never saw is open, one it decided holds a term (or the host's silence), and one that
    can not be spoken is unreachable with the reason. Nothing else books that.
    """

    def __init__(
        self,
        queue_id: str,
        max_terms: int = TERMS_PER_QUEUE,
        on_evicted: Callable[[ProgramTerm], None] | None = None,
    ) -> None:
        """
        Initialize the store of one queue.

        :param queue_id: The queue whose boundaries these terms belong to.
        :param max_terms: How many terms to remember at most.
        :param on_evicted: Called with a term the bounded store had to give up, so a
            placed one can be taken back from the bus.
        """
        self._on_evicted = on_evicted
        self.queue_id = queue_id
        # a store of zero would have nowhere to keep a term
        self._max_terms = max(1, max_terms)
        self._boundaries: dict[BoundaryKey, BoundaryRecord] = {}
        # counts the boundaries added without a play position, so they keep the order they
        # were added in; a caller that knows where a boundary plays always says so
        self._added = 0
        # the boundary the item that plays now ends at: due next, so never given up for room
        self._running: BoundaryKey | None = None
        # the station and host the queue's terms are rendered with
        self.program: dict[str, Any] | None = None
        # rendering takes seconds, so a pass that renders must not overlap another one
        self.lock = asyncio.Lock()

    def __len__(self) -> int:
        """Return how many boundaries the store holds an entry for."""
        return len(self._boundaries)

    def add(self, term: ProgramTerm, *, position: int | None = None) -> ProgramTerm | None:
        """
        Decide the given term's boundary, keeping one term per boundary that is alive.

        :param term: The term a planning pass decided on.
        :param position: Where the boundary plays, in the queue's playing order. A boundary
            decided again after it was settled keeps the place it plays at instead of
            moving to the end of the store.
        :return: The term now held for that boundary, or None when the boundary is closed.
        """
        self._added += 1
        order = self._added if position is None else position
        record = self._boundaries.get(term.key)
        if record is not None:
            if record.state is BoundaryState.UNREACHABLE:
                # nothing reopens a boundary that no pass can speak
                return None
            if record.term is not None and not record.term.is_terminal:
                return record.term
            # the same boundary came back, e.g. a repeat-one replay or a term that was
            # dropped: it is a new term, decided in place so its order keeps its place
            record.term = term
            record.state = BoundaryState.DECIDED
            record.reason = None
            record.position = order
            return term
        if self._held_terms() >= self._max_terms:
            self._make_room()
        self._boundaries[term.key] = BoundaryRecord(
            queue_item_id=term.queue_item_id,
            successor_id=term.successor_id,
            state=BoundaryState.DECIDED,
            term=term,
            position=order,
        )
        return term

    def settle(self, boundaries: Iterable[BoundaryKey]) -> None:
        """
        Record boundaries a pass looked at and found nothing to say at.

        A pass that rolled its chance and planned no break there settled the boundary as
        surely as one that decided a term: rolling it again on every pass would speak a
        break where one roll already said no. A boundary that already has an entry keeps it.

        :param boundaries: The boundaries the pass evaluated.
        """
        for key in boundaries:
            record = self._boundaries.get(key)
            if record is None:
                self._boundaries[key] = BoundaryRecord(
                    queue_item_id=key[0], successor_id=key[1], state=BoundaryState.DECIDED
                )
                continue
            if record.state is not BoundaryState.OPEN:
                continue
            # a boundary whose last term was dropped is settled by the pass that found
            # nothing to say there; what that term was is no longer this entry's business
            record.term = None
            record.state = BoundaryState.DECIDED

    def reopen(self, key: BoundaryKey) -> None:
        """
        Open a boundary a pass settled without a term, so it may be decided again.

        :param key: The boundary to open.
        """
        record = self._boundaries.get(key)
        if record is not None and record.term is None and record.state is BoundaryState.DECIDED:
            del self._boundaries[key]

    def reopen_settled(self) -> None:
        """Open every boundary a pass settled without a term, as a switched host does."""
        for key, record in list(self._boundaries.items()):
            if record.term is None and record.state is BoundaryState.DECIDED:
                del self._boundaries[key]

    def set_running(self, key: BoundaryKey | None) -> None:
        """
        Record the boundary the item that plays now ends at.

        :param key: The running boundary, or None while no item plays.
        """
        self._running = key

    def state(self, key: BoundaryKey) -> BoundaryState:
        """
        Return what the store settled about the given boundary.

        :param key: The boundary to look up.
        """
        record = self._boundaries.get(key)
        return BoundaryState.OPEN if record is None else record.state

    def reason(self, key: BoundaryKey) -> str | None:
        """
        Return why the given boundary can not be spoken, if it was found unservable.

        :param key: The boundary to look up.
        """
        record = self._boundaries.get(key)
        return None if record is None else record.reason

    def is_settled(self, key: BoundaryKey) -> bool:
        """
        Return whether the store has decided the given boundary, one way or the other.

        :param key: The boundary to look up.
        """
        return self.state(key) is not BoundaryState.OPEN

    def term_of(self, key: BoundaryKey) -> ProgramTerm | None:
        """
        Return the term that speaks at the given boundary, if one was decided there.

        :param key: The boundary to look up.
        """
        record = self._boundaries.get(key)
        return None if record is None else record.term

    def boundaries(self) -> tuple[BoundaryKey, ...]:
        """Return every boundary the store holds an entry for, oldest first."""
        return tuple(self._boundaries)

    def get(self, key: BoundaryKey) -> ProgramTerm | None:
        """
        Return the term of the given boundary, if it is still remembered.

        :param key: The boundary to look up.
        """
        return self.term_of(key)

    def terms(self) -> tuple[ProgramTerm, ...]:
        """Return every term the store holds, oldest first."""
        return tuple(record.term for record in self._boundaries.values() if record.term is not None)

    def active(self) -> tuple[ProgramTerm, ...]:
        """Return the terms that are still to be heard."""
        return tuple(term for term in self.terms() if not term.is_terminal)

    def room(self, reserve: int = 0) -> int:
        """
        Return how many more terms the store takes without giving one up.

        Counted on the terms still to be heard, never on the records: a term that aired holds
        its boundary (so a re-roll can not open it again) but a new term takes its record over
        for free, so it is no reason to leave a boundary undecided. The bound is the store's
        own, never one a caller restates.

        :param reserve: Store slots the caller has to keep free, e.g. for a show's outro.
        """
        return max(0, self._max_terms - max(0, reserve) - len(self.active()))

    def replaceable_slots(self) -> list[int]:
        """
        Return where the terms a nearer boundary may take the slot of play, nearest first.

        A boundary the store can not hold is given up for one that comes first, and what it
        gives up is the term that is due last. The end of the show is no such term: nothing
        speaks after it, so it keeps the slot it holds however far out it is.
        """
        positions = [
            record.position
            for key, record in self._boundaries.items()
            if record.term is not None
            and not record.term.is_terminal
            and record.term.successor_id is not None
            and key != self._running
        ]
        return sorted(positions)

    def drop(self, key: BoundaryKey, reason: str | ScheduleReason) -> ProgramTerm | None:
        """
        End the term of the given boundary and record what that means for the boundary.

        :param key: The boundary whose term ends.
        :param reason: Why the term never aired.
        :return: The dropped term, or None when it had already ended or was unknown.
        """
        record = self._boundaries.get(key)
        term = None if record is None else record.term
        if record is None or term is None or term.is_terminal:
            return None
        term.mark_dropped(reason)
        if str(reason) in PERMANENT_DROP_REASONS:
            # the boundary itself is the reason, so no pass would get a different answer
            record.state = BoundaryState.UNREACHABLE
            record.reason = str(reason)
        else:
            # it was only this moment that was missed: a later pass may speak it
            record.state = BoundaryState.OPEN
            record.reason = None
        return term

    def drop_all(self, reason: str | ScheduleReason) -> tuple[ProgramTerm, ...]:
        """
        Drop every term that is still to be heard.

        :param reason: Why the terms never aired.
        :return: The terms that were dropped by this call.
        """
        dropped = self.active()
        for term in dropped:
            self.drop(term.key, reason)
        return dropped

    def retain(self, item_ids: Iterable[str]) -> None:
        """
        Forget the boundaries of items the queue no longer holds.

        A term still to be heard is kept: the pump still has to drive it, and dropping it
        here would leave a placed one on the bus.

        :param item_ids: The queue item ids the queue holds now.
        """
        held = set(item_ids)
        for key, record in list(self._boundaries.items()):
            if key[0] in held:
                continue
            if record.term is not None and not record.term.is_terminal:
                continue
            del self._boundaries[key]

    def _held_terms(self) -> int:
        """Return how many boundaries hold a term, the measure the store's size is bounded by."""
        return sum(
            1
            for record in self._boundaries.values()
            if record.term is not None and record.state is not BoundaryState.UNREACHABLE
        )

    def _make_room(self) -> None:
        """
        Give up one boundary so a new term fits.

        A boundary that already ended is cheaper to forget than one still to come, and a
        closed one holds no room at all.
        """
        # the record to give up is found first and removed after the walk: a dict may not be
        # resized while it is iterated, and that must not hinge on a return right after
        ended = next(
            (
                key
                for key, record in self._boundaries.items()
                if record.state is not BoundaryState.UNREACHABLE
                and (record.term is None or record.term.is_terminal)
            ),
            None,
        )
        if ended is not None:
            del self._boundaries[ended]
            return
        # every term is still live, so the farthest one goes. the position a boundary plays
        # at is what orders them, never the order they were added in: one that was settled
        # and decided again is added last, and it is not the boundary that is due last. the
        # boundary the running item ends at is due next, so it is never the one given up
        victims = [
            (record.position, index, key)
            for index, (key, record) in enumerate(self._boundaries.items())
            if record.state is not BoundaryState.UNREACHABLE
            and record.term is not None
            and key != self._running
        ]
        if not victims:
            return
        # the break at the end of the show is never given up for room: nothing speaks after
        # it, so no boundary that comes before it takes the slot it holds
        replaceable = [
            victim
            for victim in victims
            if (term := self._boundaries[victim[2]].term) is not None and term.successor_id
        ]
        _position, _index, farthest_key = max(replaceable or victims)
        farthest = self._boundaries[farthest_key].term
        if farthest is None:
            return
        del self._boundaries[farthest_key]
        farthest.mark_dropped(ScheduleReason.LEDGER_EVICTED)
        if self._on_evicted is not None:
            self._on_evicted(farthest)

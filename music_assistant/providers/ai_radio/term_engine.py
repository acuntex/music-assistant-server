"""
The life of a queue DJ's program terms.

A term is decided while a queue is replanned, rendered only once its boundary is close
enough to reach, placed on the queue's program bus and then followed until the listener
heard it or the boundary was lost. Nothing here writes a queue item: the queue stays the
user's, and the term is the show that runs over it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from music_assistant_models.enums import MediaType, PlaybackState

from music_assistant.controllers.streams.constants import LEAD_MIN_LEAD_SECONDS
from music_assistant.controllers.streams.program_ledger import OverlayProgramState, OverlayState
from music_assistant.controllers.streams.program_schedule import (
    ProgramKind,
    ScheduleReason,
    ScheduleResult,
)

from .constants import (
    ATTR_QUEUE_DJ,
    OVERLAY_LABEL_TRANSITION,
    TERM_HEAD_DEFAULT_SECONDS,
)
from .contribution import cap_contribution_pcm, contribution_seconds
from .helpers import coerce_float
from .term_timing import (
    TERM_SCHEDULE_TOLERANCE_SECONDS,
    RenderEstimate,
    TermTiming,
    boundary_remaining,
    term_timing_for_remaining,
)
from .terms import (
    PERMANENT_DROP_REASONS,
    BoundaryKey,
    BoundaryState,
    ProgramTerm,
    TermDropReason,
    TermState,
    TermStore,
)

if TYPE_CHECKING:
    import logging

    from music_assistant_models.media_items import AudioFormat
    from music_assistant_models.player_queue import PlayerQueue
    from music_assistant_models.queue_item import QueueItem

    from music_assistant.mass import MusicAssistant

    from .models import DJQueueState, PlannedSection

# how far a position may drift before a backward move is read as a seek
SEEK_BACK_TOLERANCE_SECONDS = 0.5
# how far a position may run ahead of the wall time it had before it is read as a seek;
# a player reports its position in steps, so a small overshoot is normal
SEEK_FORWARD_TOLERANCE_SECONDS = 2.0
# a term whose boundary is closer than this is not worth a pass of its own
MIN_TERM_PUMP_SECONDS = 0.5
# a flow session registers a moment after the play it belongs to, so a term that never saw
# a bus waits this long before it is written off instead of losing the boundary to a race
BUS_GRACE_SECONDS = 1.5
# while nothing plays a queue's terms are only looked at this often, so a bus that appears
# in the pause is picked up even when no queue event announces it
PAUSED_TERM_PUMP_SECONDS = 5.0
# a term far from its boundary is looked at again in steps rather than once: the bus lead is
# the producer's buffer, which grows while it fills, so a pass booked a whole window ahead
# would render against the lead that was true when it was booked and could miss the boundary
TERM_PUMP_MAX_SECONDS = 15.0

TermPosition = Literal["current", "ahead", "behind"]


class TermReopenCause(StrEnum):
    """What makes a pump open a boundary an earlier pass had settled."""

    # the queue plays again after a pause it spent settled on that boundary
    RESUME = "resume"
    # the host or its sections were switched, so its silence says nothing about this one
    PROGRAM = "program"


@dataclass(slots=True)
class TermReopenMark:
    """What the last pump of a queue saw, so a settled boundary opens only on a change."""

    playing: bool
    # the host and the sections its program was built from, as a comparable snapshot
    program: tuple[str | None, tuple[Any, ...] | None]


def _program_signature(program: dict[str, Any] | None) -> tuple[Any, ...] | None:
    """
    Return the sections a program plans with, as a snapshot that compares by content.

    :param program: The station and host a queue's terms are rendered with, if known.
    """
    if program is None:
        return None
    sections = program.get("sections")
    return (
        tuple(
            sorted(
                str(section.get("id", ""))
                for section in (sections if isinstance(sections, list) else [])
                if isinstance(section, dict)
            )
        ),
        str(program.get("section_order", [])),
    )


@dataclass(frozen=True)
class TermBoundaryPlan:
    """One boundary a planning pass wants a term spoken at."""

    queue_item_id: str
    successor_id: str | None
    sections: tuple[PlannedSection, ...]
    # where the boundary plays, in the queue's playing order, so a bounded store knows which
    # term is the one furthest out instead of trusting the order it added them in
    index: int | None = None


@dataclass(frozen=True)
class RunningBoundary:
    """The boundary the item that plays now ends at."""

    # queue index of the running item, so a pass can start its window there
    index: int
    item_id: str
    successor_id: str | None


def running_boundary(items: list[QueueItem], current_item_id: str | None) -> RunningBoundary | None:
    """
    Return the boundary the item that plays now ends at.

    :param items: All items of the queue, in playing order.
    :param current_item_id: The queue item the player stands in, if any.
    """
    if current_item_id is None:
        return None
    for index, item in enumerate(items):
        if item.queue_item_id != current_item_id:
            continue
        following = next(
            (item for item in items[index + 1 :] if not item.extra_attributes.get(ATTR_QUEUE_DJ)),
            None,
        )
        return RunningBoundary(
            index=index,
            item_id=item.queue_item_id,
            successor_id=None if following is None else following.queue_item_id,
        )
    return None


def gap_can_air(window_tracks: list[dict[str, Any]], insert_at_index: int) -> bool:
    """
    Return whether the gap a section is planned for can air at all.

    Both sides of the boundary have to be able to play: a break written for a track that
    never started, or one that announces a track that cannot play, is written and paid for
    and then dropped.

    :param window_tracks: The tracks of the planning window, in playing order.
    :param insert_at_index: The window index the clip is planned in front of, which is also
        how a section names its gap; the track before it is the one the break follows.
    """
    if not window_tracks:
        return False
    if insert_at_index >= len(window_tracks):
        return bool(window_tracks[-1].get("available", True))
    if insert_at_index <= 0:
        return bool(window_tracks[0].get("available", True))
    return bool(
        window_tracks[insert_at_index - 1].get("available", True)
        and window_tracks[insert_at_index].get("available", True)
    )


def section_boundary(
    section: PlannedSection,
    window_tracks: list[dict[str, Any]],
    outro_target: dict[str, Any] | None,
) -> tuple[dict[str, Any], int, bool]:
    """
    Return the track a section speaks around, the boundary it guards and whether it is an outro.

    :param section: The planned section.
    :param window_tracks: The tracks of the planning window, in playing order.
    :param outro_target: The window track the show's outro belongs after, if one is offered.
    """
    # an end_of_playlist section targets a slot past the last window track;
    # _plan_sections numbers it len(window_tracks), one past the last valid index
    if section.insert_at_index != len(window_tracks):
        return window_tracks[section.insert_at_index], section.insert_at_index, False
    # the outro belongs after the show's true final track, which is not necessarily the
    # queue tail (a user may have appended tracks behind it)
    target = outro_target if outro_target is not None else window_tracks[-1]
    guard_at = section.insert_at_index
    if target in window_tracks:
        # an outro may sit before tracks appended behind the show, so its boundary is the
        # target's, not the window tail's
        guard_at = window_tracks.index(target) + 1
    return target, guard_at, True


def term_boundary_plans(
    planned: list[PlannedSection],
    window_tracks: list[dict[str, Any]],
    outro_target: dict[str, Any] | None,
    history: dict[str, list[tuple[int, float]]],
    *,
    index_base: int = 0,
) -> list[TermBoundaryPlan]:
    """
    Resolve the planned sections to the boundaries their terms speak at.

    :param planned: The sections this pass planned.
    :param window_tracks: The tracks of the planning window, in playing order.
    :param outro_target: The window track the show's outro belongs after, if one is offered.
    :param history: The guard history of the pass, so a section that can never air is released.
    :param index_base: Songs that play before the window, so a boundary carries its place in
        the queue rather than its place in this pass's window.
    """
    grouped: dict[tuple[str, str | None], list[PlannedSection]] = {}
    index_of: dict[tuple[str, str | None], int] = {}
    for section in planned:
        target, guard_at, after_target = section_boundary(section, window_tracks, outro_target)
        if not gap_can_air(window_tracks, guard_at):
            # a boundary that cannot play is nothing to speak over, and its gap stays
            # decided, exactly as the item path keeps it
            _drop_planned_history(section, history)
            continue
        key: tuple[str, str | None]
        if after_target:
            key = (str(target["item_id"]), None)
            index = window_tracks.index(target)
        elif guard_at > 0:
            key = (str(window_tracks[guard_at - 1]["item_id"]), str(target["item_id"]))
            index = guard_at - 1
        else:
            # a break planned before the first track of the window has no item to speak
            # over; the item path can splice it, a term can not
            _drop_planned_history(section, history)
            continue
        index_of.setdefault(key, index_base + index)
        grouped.setdefault(key, []).append(section)
    return [
        TermBoundaryPlan(
            queue_item_id=item_id,
            successor_id=successor_id,
            sections=tuple(sections),
            index=index_of[(item_id, successor_id)],
        )
        for (item_id, successor_id), sections in grouped.items()
    ]


def _drop_planned_history(
    section: PlannedSection, history: dict[str, list[tuple[int, float]]]
) -> None:
    """Forget the guard events of a section that was never decided."""
    for section_id, event in section.history_events:
        if (events := history.get(section_id)) and event in events:
            events.remove(event)


def term_plans_in_horizon(
    plans: list[TermBoundaryPlan],
    horizon: set[BoundaryKey],
    history: dict[str, list[tuple[int, float]]],
) -> list[TermBoundaryPlan]:
    """
    Keep the boundaries a planning pass may decide now, and release the rest.

    A boundary past the horizon stays open: the playback rolls it into reach, and it is
    decided then. Deciding it now would put a term in the bounded store that is given up
    long before it is due, taking its gap with it. The guard event of a deferred section
    is released, so it holds no later break back while it waits.

    :param plans: The boundaries this pass planned.
    :param horizon: The boundaries this pass may decide.
    :param history: The guard history of the pass, so a deferred plan leaves nothing behind.
    """
    kept: list[TermBoundaryPlan] = []
    for plan in plans:
        if plan.successor_id is None or (plan.queue_item_id, plan.successor_id) in horizon:
            # the outro announces no successor, so no horizon is in front of it
            kept.append(plan)
            continue
        for section in plan.sections:
            _drop_planned_history(section, history)
    return kept


class AIRadioTermEngineMixin:
    """Mixin that turns a queue's planned boundaries into terms on the program bus."""

    if TYPE_CHECKING:
        mass: MusicAssistant
        logger: logging.Logger
        _dj_queues: dict[str, DJQueueState]

        async def _render_mixed_program(
            self,
            queue_id: str,
            program: dict[str, Any],
            sections: list[PlannedSection],
            pcm_format: AudioFormat,
        ) -> bytes:
            """Render a program for the queue's stream."""

        def _dj_queue_items(self, queue_id: str) -> list[QueueItem]:
            """Return all items of a queue."""

        @staticmethod
        def _queue_plays(queue: PlayerQueue) -> bool:
            """Return whether the queue is playing."""

        def _schedule_replan(self, queue_id: str) -> None:
            """Request a replan pass for the given queue."""

    _term_stores: dict[str, TermStore]
    _term_estimates: dict[str, RenderEstimate]

    def _term_store(self, queue_id: str) -> TermStore:
        """
        Return the term store of a queue, creating it on first use.

        :param queue_id: The queue whose terms are kept.
        """
        stores: dict[str, TermStore] = getattr(self, "_term_stores", {})
        if not hasattr(self, "_term_stores"):
            self._term_stores = stores
        store = stores.get(queue_id)
        if store is None:
            store = stores[queue_id] = TermStore(
                queue_id,
                on_evicted=lambda term: self._term_evicted(queue_id, term),
            )
        return store

    def _decide_terms(
        self,
        queue_id: str,
        program: dict[str, Any],
        plans: list[TermBoundaryPlan],
        *,
        history: dict[str, list[tuple[int, float]]] | None = None,
        horizon: set[BoundaryKey] | None = None,
    ) -> None:
        """
        Decide which sections speak at which boundary, without rendering anything.

        :param queue_id: The queue the terms belong to.
        :param program: The station and host the sections are rendered with.
        :param plans: The boundaries this planning pass wants served.
        :param history: The pass's guard history, so a boundary that can not be served
            releases the weight its plan claimed.
        :param horizon: Every boundary the pass evaluated, so the ones it left silent are
            settled and are not rolled again by the next pass.
        """
        store = self._term_store(queue_id)
        store.program = program
        # the boundary that is due next is never the one the bounded store gives up
        store.set_running(self._term_running_key(queue_id))
        # the host owns how far a break starts before its boundary, so the decision and the
        # placement are made on the same reading of it
        lead = self._term_head_lead(program)
        for plan in plans:
            allow_pre = any(section.allow_pre for section in plan.sections)
            # a merged break carries over only when every section of it allows the post: one
            # of them ending before the boundary ends the whole break there
            allow_post = all(section.allow_post for section in plan.sections)
            term = store.add(
                ProgramTerm(
                    queue_item_id=plan.queue_item_id,
                    successor_id=plan.successor_id,
                    section_ids=tuple(section.section_id for section in plan.sections),
                    # the clip is unknown until it is rendered, so the decision plans with the
                    # lead a break may start before the boundary; the placement replaces it
                    # with the head the clip really has
                    head_seconds=lead if allow_pre else 0.0,
                    allow_pre=allow_pre,
                    allow_post=allow_post,
                    sections=plan.sections,
                ),
                position=plan.index,
            )
            if term is not None or history is None:
                continue
            # the store closed this boundary for good, so nothing here will ever be spoken
            for section in plan.sections:
                _drop_planned_history(section, history)
        if horizon is not None:
            planned_keys = {(plan.queue_item_id, plan.successor_id) for plan in plans}
            # a boundary this pass looked at and planned nothing for is settled just as much
            silent = [key for key in horizon if key not in planned_keys]
            if silent:
                # the one thing no other line says: a boundary the host left without a
                # break. it is what a listener hears as a DJ that has gone quiet, so a
                # report of one can be told apart from a boundary that was never looked at
                self.logger.debug(
                    "Queue %s speaks no break at %d of the %d boundaries it served: %s",
                    queue_id,
                    len(silent),
                    len(horizon),
                    ", ".join(str(key[0]) for key in silent),
                )
            store.settle(silent)

    def _term_holds_boundary(self, queue_id: str, boundary: RunningBoundary) -> bool:
        """
        Return whether the store has already settled the given boundary.

        :param queue_id: The queue whose terms are looked at.
        :param boundary: The boundary a term would speak at.
        """
        return self._term_store(queue_id).is_settled((boundary.item_id, boundary.successor_id))

    def _term_running_key(self, queue_id: str) -> BoundaryKey | None:
        """
        Return the boundary the item that plays now ends at, as the store names it.

        :param queue_id: The queue whose running boundary is looked up.
        """
        queue = self.mass.player_queues.get(queue_id)
        current = queue.current_item if queue is not None else None
        if queue is None or current is None:
            return None
        boundary = running_boundary(self._dj_queue_items(queue_id), current.queue_item_id)
        if boundary is None:
            return None
        return (boundary.item_id, boundary.successor_id)

    def _term_decided_next_item_ids(self, queue_id: str) -> set[str]:
        """
        Return the successors of the boundaries the store already decided.

        The term path books its boundaries in the store instead of in the item path's
        ledger, and the planner has to know what is decided there: re-evaluating a settled
        boundary would roll its chance a second time and register its guard event again.

        :param queue_id: The queue whose boundaries are looked at.
        """
        store = self._term_store(queue_id)
        return {
            key[1] for key in store.boundaries() if key[1] is not None and store.is_settled(key)
        }

    def _term_gap_horizon(
        self,
        queue_id: str,
        window_tracks: list[dict[str, Any]],
        *,
        reserve: int = 0,
        index_base: int = 0,
    ) -> set[BoundaryKey]:
        """
        Return the boundaries a pass may decide: the nearest ones, however full the store is.

        What the store can not hold is given up for what comes first: a boundary that is due
        before every term the store holds takes the slot of the term that is due last. Only
        the boundaries behind those are left open until the playback has rolled them into
        reach, so a store full of far terms never leaves a near boundary silent. The end of
        the show is no term a nearer boundary takes the slot of - nothing speaks after it.

        :param queue_id: The queue whose terms are planned.
        :param window_tracks: The tracks of the planning window, in playing order.
        :param reserve: Store slots this pass has to keep free, e.g. for a show's outro.
        :param index_base: Songs that play before the window, so a boundary carries its place
            in the queue and compares with the place the store keeps its terms at.
        """
        store = self._term_store(queue_id)
        # the store answers what it still holds, so a queue whose store keeps fewer terms
        # than this module's default decides fewer boundaries
        room = store.room(reserve)
        open_boundaries: list[tuple[int, BoundaryKey]] = [
            (index_base + index, (str(window_tracks[index - 1]["item_id"]), str(track["item_id"])))
            for index, track in enumerate(window_tracks)
            if index > 0
            and not store.is_settled(
                (str(window_tracks[index - 1]["item_id"]), str(track["item_id"]))
            )
        ]
        horizon = [key for _position, key in open_boundaries[:room]]
        # a boundary the store has no free slot for still takes the slot of the term that is
        # due last: that term is given up long before it is due, and the nearer one comes
        # first. this is what keeps a store filled with far terms from going quiet
        replaceable = store.replaceable_slots()
        for position, key in open_boundaries[room:]:
            if not replaceable or position >= replaceable[-1]:
                # nothing the store holds is due after this boundary, so it waits its turn
                break
            replaceable.pop()
            horizon.append(key)
        return set(horizon)

    def _term_outro_reserve(self, queue_id: str, outro_target: dict[str, Any] | None) -> int:
        """
        Return the store room an outro this pass would decide has to keep free.

        The outro is the farthest boundary of a show, so the bounded store would give it up
        first. It is planned once per play and speaks the end of the show, so a slot is kept
        for it instead of deciding one boundary more today.

        The outro's boundary is the one without a successor, so no horizon of window pairs
        holds it and no pass settles it: a pass that rolls no outro for it rolls again on the
        next one, which is what the item path always did for the end of a show. That stays as
        it is - the roll is one per replan, and a show whose tail changes still has to be
        offered its outro again - so the settlement of the end of the queue is the one open
        boundary a pass may re-roll.

        :param queue_id: The queue whose terms are planned.
        :param outro_target: The window track the show's outro belongs after, if one is offered.
        """
        if outro_target is None:
            return 0
        store = self._term_store(queue_id)
        if any(term.successor_id is None for term in store.active()):
            # the queue already holds the outro, so this pass adds nothing for it
            return 0
        if store.is_settled((str(outro_target["item_id"]), None)):
            # the outro was decided already, or closed: either way nothing is owed for it
            return 0
        return 1

    def _drop_terms(self, queue_id: str, reason: str | ScheduleReason) -> tuple[ProgramTerm, ...]:
        """
        Drop every term of a queue that is still to be heard, cancelling placed ones.

        :param queue_id: The queue whose terms end.
        :param reason: Why the terms will not be heard.
        """
        stores: dict[str, TermStore] = getattr(self, "_term_stores", {})
        store = stores.get(queue_id)
        if store is None:
            return ()
        dropped = store.drop_all(reason)
        for term in dropped:
            # a term the bus already holds has to be taken back, or it would air after it
            # was dropped (a resume, for one, would otherwise still speak it)
            if term.scheduled_frame is not None:
                self.mass.streams.cancel_overlay(
                    queue_id,
                    OVERLAY_LABEL_TRANSITION,
                    str(reason),
                    start_frame=term.scheduled_frame,
                )
            self._forget_term_guard(queue_id, term)
        return dropped

    def _reset_terms(self, queue_id: str) -> None:
        """
        Forget a queue's terms, as a disarmed DJ or a restart does.

        :param queue_id: The queue whose terms are forgotten.
        """
        self._drop_terms(queue_id, TermDropReason.FLOW_ENDED)
        due_by_queue: dict[str, float] = getattr(self, "_term_pump_due", {})
        due_by_queue.pop(queue_id, None)
        # a re-armed DJ looks at its running boundary afresh instead of waiting out a period
        replan_due: dict[str, float] = getattr(self, "_term_boundary_replan_due", {})
        replan_due.pop(queue_id, None)
        # nothing of the old session is left to compare a pause or a host switch against
        marks: dict[str, TermReopenMark] = getattr(self, "_term_reopen_marks", {})
        marks.pop(queue_id, None)
        stores: dict[str, TermStore] = getattr(self, "_term_stores", {})
        stores.pop(queue_id, None)
        estimates: dict[str, RenderEstimate] = getattr(self, "_term_estimates", {})
        estimates.pop(queue_id, None)

    async def _pump_terms(self, queue_id: str) -> None:
        """
        Move every term of a queue one step forward.

        :param queue_id: The queue whose terms are driven.
        """
        # this pass serves the queue's booking, so a later term may book again
        due_by_queue: dict[str, float] = getattr(self, "_term_pump_due", {})
        due_by_queue.pop(queue_id, None)
        stores: dict[str, TermStore] = getattr(self, "_term_stores", {})
        store = stores.get(queue_id)
        if store is None:
            return
        if getattr(self, "_unloading", False):
            return
        if not self.mass.streams.program_bus_enabled():
            self._drop_terms(queue_id, TermDropReason.NO_BUS)
            return
        queue = self.mass.player_queues.get(queue_id)
        if queue is None:
            self._drop_terms(queue_id, TermDropReason.FLOW_ENDED)
            return
        self.logger.debug("TERM-PUMP queue=%s terms=%d", queue_id, len(store.active()))
        # read before anything is driven: a pause spent settled and the play that follows it
        # are the change that lets a decided boundary be decided again
        cause = self._term_reopen_cause(queue_id, store, queue)
        if cause is TermReopenCause.PROGRAM:
            # the new host speaks with its own sections, so every boundary it never looked
            # at is decided again instead of staying silent for the rest of the show
            store.reopen_settled()
        async with store.lock:
            items = self._dj_queue_items(queue_id)
            active = store.active()
            # one ledger read per pump, however many terms stand on it: every placed term
            # settles against the same answer, and a queue with sixteen of them is sixteen
            # reads less
            ledger: list[OverlayState] = (
                self.mass.streams.overlay_state(queue_id)
                if any(term.state is TermState.SCHEDULED for term in active)
                else []
            )
            for term in active:
                await self._pump_term(queue_id, store, term, queue, items, ledger=ledger)
            self._replan_missed_boundary(queue_id, queue, items, cause=cause)
            # a boundary whose item left the queue says nothing about the next pass, and the
            # store would otherwise carry a closed one for the rest of the session
            store.retain(item.queue_item_id for item in items)

    def _term_reopen_cause(
        self, queue_id: str, store: TermStore, queue: PlayerQueue
    ) -> TermReopenCause | None:
        """
        Return what this pump may open a settled boundary for, if anything.

        A boundary a pass settled without a break stays settled: it is the boundary that is
        due next, and opening it on every pump would make the host's chance meaningless
        there. Only a real change opens it again - the queue started playing after a pause,
        or the host or its sections were switched.

        :param queue_id: The queue the pump drives.
        :param store: The queue's term store, holding the program the last pass decided with.
        :param queue: The queue that is playing.
        """
        state = self._dj_queues.get(queue_id)
        program = (None if state is None else state.host_id, _program_signature(store.program))
        playing = self._queue_plays(queue)
        marks: dict[str, TermReopenMark] = getattr(self, "_term_reopen_marks", {})
        if not hasattr(self, "_term_reopen_marks"):
            self._term_reopen_marks = marks
        mark = marks.get(queue_id)
        marks[queue_id] = TermReopenMark(playing=playing, program=program)
        if mark is None:
            # nothing is known about this queue yet, so this pump is no change of anything
            return None
        if playing and not mark.playing:
            return TermReopenCause.RESUME
        if program[1] is not None and mark.program[1] is not None and program != mark.program:
            return TermReopenCause.PROGRAM
        return None

    def _replan_missed_boundary(
        self,
        queue_id: str,
        queue: PlayerQueue,
        items: list[QueueItem],
        *,
        cause: TermReopenCause | None = None,
    ) -> None:
        """
        Ask for a replan when the boundary the running item ends at has no term.

        The pump only drives the terms a queue already holds, so a boundary a pass settled
        without a break stays silent for as long as nothing else asks for a pass: a pause
        and resume of the same item raises no event of its own. A boundary a term was
        dropped from is open in the store already, and the one a pass settled is opened here
        on the cause the pump found. The pass that looks at it again is asked for at most
        once per pump period, so a planner that legitimately decides nothing there is not
        asked in a loop.

        Only an open boundary is woken: one that holds a term speaks already, and one the
        store closed for good would fail again for the same reason, so it is left alone.

        :param queue_id: The queue whose running boundary is looked at.
        :param queue: The queue that is playing.
        :param items: All items of the queue, in playing order.
        :param cause: What this pump may open a settled boundary for, if anything.
        """
        state = self._dj_queues.get(queue_id)
        if state is None:
            return
        if not self._queue_plays(queue) or not self.mass.streams.program_bus_active(queue_id):
            return
        now = asyncio.get_running_loop().time()
        if state.render_health.cooling_down(now):
            # the render that tripped the cool-down booked the pass that plans again
            return
        current = queue.current_item
        boundary = running_boundary(items, None if current is None else current.queue_item_id)
        if boundary is None:
            return
        store = self._term_store(queue_id)
        key = (boundary.item_id, boundary.successor_id)
        boundary_state = store.state(key)
        if boundary_state is BoundaryState.UNREACHABLE:
            # no pass can speak this boundary, so asking for one only repeats the failure
            return
        held = store.term_of(key)
        if held is not None and not held.is_terminal:
            # a term speaks the boundary already, whether it aired or is still to come
            return
        if boundary_state is BoundaryState.DECIDED:
            if held is not None:
                # the term that spoke here aired, and what the listener heard is decided
                return
            # settled without a break by an earlier pass: only a real change opens it again
            if cause is None:
                return
            store.reopen(key)
        if state.replan_pending:
            # a pass is already on its way, and it finds the boundary open
            return
        due_by_queue: dict[str, float] = getattr(self, "_term_boundary_replan_due", {})
        if not hasattr(self, "_term_boundary_replan_due"):
            self._term_boundary_replan_due = due_by_queue
        if now < due_by_queue.get(queue_id, 0.0):
            return
        due_by_queue[queue_id] = now + TERM_PUMP_MAX_SECONDS
        self._schedule_replan(queue_id)
        # no queue event follows a pause resumed on the same item, so the retry is booked
        # here instead of waiting for one
        self._book_term_pump(queue_id, TERM_PUMP_MAX_SECONDS)

    async def _pump_term(
        self,
        queue_id: str,
        store: TermStore,
        term: ProgramTerm,
        queue: PlayerQueue,
        items: list[QueueItem],
        *,
        ledger: list[OverlayState],
    ) -> None:
        """
        Move one term forward, from deciding to rendering, placing or ending.

        :param queue_id: The queue the term belongs to.
        :param store: The queue's term store.
        :param term: The term this step moves.
        :param queue: The queue that is playing.
        :param items: The queue's items, in playing order.
        :param ledger: The queue's program ledger as this pump read it.
        """
        now = asyncio.get_running_loop().time()
        if term.is_terminal:
            return
        if queue.ended or queue.state is PlaybackState.IDLE:
            # a stop and a played-to-the-end queue end their terms; a pause does not
            self._drop_term(queue_id, store, term, TermDropReason.FLOW_ENDED)
            return
        paused = not self._queue_plays(queue)
        if term.state is TermState.SCHEDULED:
            if not self.mass.streams.program_bus_active(queue_id):
                reason = ScheduleReason.BUS_REPLACED if term.bus_seen else TermDropReason.NO_BUS
                self._drop_term(queue_id, store, term, reason)
                return
            term.bus_seen = True
            self._settle_scheduled_term(
                queue_id, store, term, queue, now, paused=paused, ledger=ledger
            )
            return
        if paused:
            # nothing is heard, so nothing is rendered and no term is lost; a DJ switched
            # on into a pause waits for the play, bus or no bus
            term.last_seen_at = now
            # every play attempt gets its own grace: a pause (or a buffering player that
            # reports no playing state for a moment) must not spend the one the next play
            # needs to wait for its flow session
            term.bus_missing_since = None
            # the stream that carries the term opens with the play, and no queue event of
            # its own follows it, so the paused queue is looked at again on its own
            self._book_term_pump(queue_id, PAUSED_TERM_PUMP_SECONDS)
            return
        if not self.mass.streams.program_bus_active(queue_id):
            if term.bus_seen:
                # the bus this term was placed for is gone: a replacement would not air it
                self._drop_term(queue_id, store, term, ScheduleReason.BUS_REPLACED)
                return
            if not self._term_bus_grace_over(term, now):
                # the play's flow session may still be on its way, so the term waits out a
                # short grace instead of losing a boundary that a late bus can still take
                self._book_term_pump(queue_id, BUS_GRACE_SECONDS)
                return
            self._drop_term(queue_id, store, term, TermDropReason.NO_BUS)
            return
        term.bus_seen = True
        if term.rendered_pcm is not None:
            # the clip already exists, so only the placement is owed
            self._place_held_term(queue_id, store, term)
            return
        await self._advance_term(queue_id, store, term, queue, items, now)

    async def _advance_term(
        self,
        queue_id: str,
        store: TermStore,
        term: ProgramTerm,
        queue: PlayerQueue,
        items: list[QueueItem],
        now: float,
    ) -> None:
        """Decide whether a term is due, and render it when it is."""
        current = queue.current_item
        if current is None:
            self._drop_term(queue_id, store, term, TermDropReason.FLOW_ENDED)
            return
        position = self._term_position(items, term, current.queue_item_id)
        if position == "behind":
            self._drop_term(queue_id, store, term, TermDropReason.SKIPPED)
            return
        if not self._term_successor_follows(items, term):
            # a reordered queue made this boundary a different one
            self._drop_term(queue_id, store, term, TermDropReason.NO_SUCCESSOR)
            return
        speed = float(getattr(queue, "playback_speed", 1.0) or 1.0)
        seeked = False
        if position == "current":
            elapsed = float(queue.corrected_elapsed_time or 0.0)
            seeked = self._term_saw_a_seek(term, elapsed, now, speed)
            term.last_elapsed = elapsed
            term.last_seen_at = now
        boundary_item = self._term_boundary_item(items, term, queue)
        if boundary_item is None:
            self._drop_term(queue_id, store, term, TermDropReason.SKIPPED)
            return
        fade_seconds = self._term_fade_seconds(boundary_item)
        if fade_seconds is None:
            self._drop_term(queue_id, store, term, TermDropReason.UNKNOWN_BOUNDARY)
            return
        remaining = self._term_seconds_to_boundary(items, queue, term, position, speed)
        if remaining is None:
            if position == "current":
                self._drop_term(queue_id, store, term, TermDropReason.NO_DURATION)
            # an item before the boundary has no duration, so its end is not timed yet
            return
        estimate = self._term_estimate(queue_id).seconds
        render_lead = self._term_render_lead(queue_id)
        timing = term_timing_for_remaining(
            remaining=remaining,
            clip_seconds=term.head_seconds,
            allow_pre=term.allow_pre,
            allow_post=term.allow_post,
            lead_seconds=self._term_head_lead(store.program),
            render_estimate=estimate,
            render_lead=render_lead,
        )
        self.logger.debug(
            "TERM queue=%s item=%s state=%s pos=%s remaining=%.2f head=%.2f estimate=%.2f "
            "lead=%.2f due=%s late=%s",
            queue_id,
            term.queue_item_id,
            term.state,
            position,
            remaining,
            timing.head_seconds,
            estimate,
            render_lead,
            timing.render_due,
            timing.too_late,
        )
        if timing.too_late:
            self._drop_term(
                queue_id,
                store,
                term,
                TermDropReason.SEEKED if seeked else ScheduleReason.TOO_LATE,
            )
            return
        if not timing.render_due:
            # the pass is booked from the window the decision was made on, so it lands due;
            # a window further out than the cap is only booked that far, and re-read then
            self._book_term_pump(
                queue_id,
                timing.render_in_seconds or 0.0,
                cap_seconds=TERM_PUMP_MAX_SECONDS,
            )
            return
        await self._render_and_place_term(queue_id, store, term)

    def _settle_scheduled_term(
        self,
        queue_id: str,
        store: TermStore,
        term: ProgramTerm,
        queue: PlayerQueue,
        now: float,
        *,
        paused: bool,
        ledger: list[OverlayState],
    ) -> None:
        """
        Follow a placed term until the listener heard it or its boundary was lost.

        :param queue_id: The queue the term was placed on.
        :param store: The queue's term store.
        :param term: The placed term.
        :param queue: The queue that is playing.
        :param now: Loop time of this pass.
        :param paused: Whether the queue is paused, so the deadline stands still.
        :param ledger: The queue's program ledger, read once for the whole pump.
        """
        for state in ledger:
            if state.label != OVERLAY_LABEL_TRANSITION:
                continue
            if term.scheduled_frame is not None and state.start_frame != term.scheduled_frame:
                continue
            if state.state == OverlayProgramState.AIRED:
                term.mark_aired()
                return
            if state.state == OverlayProgramState.DROPPED:
                self._drop_term(queue_id, store, term, state.reason or TermDropReason.FLOW_ENDED)
                return
        if not paused and term.seconds_left is not None:
            # the deadline is stream time: it stands still while nothing is heard, so only
            # the media the item really advanced counts, divided by its playback speed
            term.seconds_left -= self._term_heard_seconds(queue, term, now)
        term.last_seen_at = now
        current = queue.current_item
        if current is not None:
            term.last_elapsed = float(queue.corrected_elapsed_time or 0.0)
        if term.seconds_left is not None and term.seconds_left <= 0:
            term.mark_aired()
            return
        current = queue.current_item
        if current is None:
            return
        if current.queue_item_id == term.queue_item_id:
            return
        if term.successor_id is not None and current.queue_item_id == term.successor_id:
            # the boundary was crossed normally; the ledger decides when it aired
            return
        self._drop_term(queue_id, store, term, TermDropReason.SKIPPED)

    async def _render_and_place_term(
        self, queue_id: str, store: TermStore, term: ProgramTerm
    ) -> None:
        """Render a term that is due and place it on the queue's program bus."""
        pcm_format = self.mass.streams.program_bus_format(queue_id)
        if pcm_format is None:
            self._drop_term(queue_id, store, term, ScheduleReason.BUS_REPLACED)
            return
        dj_state = self._dj_queues.get(queue_id)
        now = asyncio.get_running_loop().time()
        if dj_state is not None and dj_state.render_health.cooling_down(now):
            # the pass after the cool-down was booked by the render that tripped it
            self.logger.debug("Queue %s renders no term while its DJ is paused", queue_id)
            return
        term.mark_rendering()
        started = now
        try:
            pcm = await self._render_mixed_program(
                queue_id, store.program or {}, list(term.sections), pcm_format
            )
        except Exception:
            # the engine is paid for once per boundary, never retried into the same gap
            self.logger.exception("Queue %s could not speak its term", queue_id)
            self._drop_term(queue_id, store, term, TermDropReason.RENDER_FAILED)
            return
        if pcm:
            # only a render that produced audio says how long a render takes: a failed one
            # returns at once, and folding that in would pull the estimate towards zero
            self._term_estimate(queue_id).record(asyncio.get_running_loop().time() - started)
        if term.is_terminal:
            # a stop, a disarm or a replaced bus ended the term while it rendered
            return
        if not pcm:
            self._drop_term(queue_id, store, term, TermDropReason.RENDER_FAILED)
            return
        pcm = self._cap_contribution(queue_id, store.program or {}, term, pcm, pcm_format)
        self._place_rendered_term(queue_id, store, term, pcm, pcm_format)

    def _cap_contribution(
        self,
        queue_id: str,
        program: dict[str, Any],
        term: ProgramTerm,
        pcm: bytes,
        pcm_format: AudioFormat,
    ) -> bytes:
        """
        Cut a rendered term to the host's ceiling for one spoken contribution.

        :param queue_id: The queue the term is spoken on.
        :param program: The station and host the term belongs to.
        :param term: The term whose sections were spoken.
        :param pcm: The rendered contribution, in the queue stream's format.
        :param pcm_format: The PCM format the queue's stream produces.
        """
        max_seconds = contribution_seconds(program)
        capped = cap_contribution_pcm(pcm, pcm_format, max_seconds)
        if len(capped) >= len(pcm):
            return pcm
        self.logger.info(
            "AI Radio cuts the contribution of section %s on queue %s: %.1fs of speech over "
            "the host's %gs are cut",
            ", ".join(term.section_ids),
            queue_id,
            self._pcm_seconds(pcm, pcm_format),
            max_seconds,
        )
        return capped

    def _place_rendered_term(
        self,
        queue_id: str,
        store: TermStore,
        term: ProgramTerm,
        pcm: bytes,
        pcm_format: AudioFormat,
    ) -> None:
        """Place a rendered term, unless the playback moved away from its boundary."""
        queue = self.mass.player_queues.get(queue_id)
        if queue is None or queue.ended:
            self._drop_term(queue_id, store, term, TermDropReason.FLOW_ENDED)
            return
        if not self._queue_plays(queue):
            # the render landed in a pause: nothing was heard, so the clip is held for the
            # resume instead of paying the engine for the same boundary twice
            term.keep_rendered(pcm, pcm_format)
            return
        if not self.mass.streams.program_bus_active(queue_id):
            self._drop_term(queue_id, store, term, ScheduleReason.BUS_REPLACED)
            return
        placement = self._rendered_placement(queue_id, store, term, pcm, pcm_format, queue)
        if placement is None:
            return
        delay, timing, now = placement
        term.head_seconds = timing.head_seconds
        if timing.head_truncated:
            self.logger.warning(
                "Queue %s speaks its term with a head cut short: the boundary is closer "
                "than the %ss the break plans to start before it",
                queue_id,
                round(term.head_seconds, 1),
            )
        elif term.head_seconds > 0 and delay < LEAD_MIN_LEAD_SECONDS:
            # the bus raises the anchor to its own lead, so the head starts later than
            # planned and the break talks over the boundary, not before it
            self.logger.warning(
                "Queue %s starts its term %.2fs before the boundary, less than the %ss bus "
                "lead: the head is cut short",
                queue_id,
                round(delay, 2),
                LEAD_MIN_LEAD_SECONDS,
            )
        result = self.mass.streams.schedule_overlay(
            queue_id,
            pcm,
            label=OVERLAY_LABEL_TRANSITION,
            start_in_seconds=delay,
            kind=ProgramKind.TERM,
            head_seconds=term.head_seconds,
        )
        if not result.scheduled:
            self._drop_term(queue_id, store, term, result.reason or ScheduleReason.TOO_LATE)
            return
        if result.reason is ScheduleReason.WINDOW_TAKEN:
            # the bus moved the term past a window it could not take: it would no longer
            # sit on the boundary it was written for, so it is taken back instead
            self.logger.info(
                "Queue %s could not take its term (%s) - it is taken back",
                queue_id,
                result.reason,
            )
            self.mass.streams.cancel_overlay(
                queue_id,
                OVERLAY_LABEL_TRANSITION,
                str(result.reason),
                start_frame=result.anchor_frame,
            )
            self._drop_term(queue_id, store, term, result.reason)
            return
        if result.late_by_seconds > TERM_SCHEDULE_TOLERANCE_SECONDS:
            # the boundary was already produced past the tolerance: the api only takes a
            # term that still lands on it
            self.logger.info(
                "Queue %s is %.1fs late for its term - it is taken back",
                queue_id,
                result.late_by_seconds,
            )
            self.mass.streams.cancel_overlay(
                queue_id,
                OVERLAY_LABEL_TRANSITION,
                str(TermDropReason.TOO_LATE),
                start_frame=result.anchor_frame,
            )
            self._drop_term(queue_id, store, term, TermDropReason.TOO_LATE)
            return
        # the term ends after the audio the bus really took, at the frames it placed, not
        # after the delay that was planned for it
        clip_seconds = self._result_seconds(result, pcm, pcm_format)
        seconds_left = (
            delay
            + max(0.0, result.late_by_seconds)
            + max(0.0, result.lead_shortfall_seconds)
            + clip_seconds
            + TERM_SCHEDULE_TOLERANCE_SECONDS
        )
        term.mark_scheduled(
            start_frame=result.anchor_frame,
            pcm_len=len(pcm),
            delay_seconds=delay,
            seconds_left=seconds_left,
        )
        term.last_seen_at = now
        # the deadline is the term's own way out, even when no queue event follows it
        self._book_term_pump(queue_id, seconds_left)
        self.logger.debug(
            "Queue %s placed a term %.1fs from now (%s)",
            queue_id,
            delay,
            result.reason or "as planned",
        )

    def _rendered_placement(
        self,
        queue_id: str,
        store: TermStore,
        term: ProgramTerm,
        pcm: bytes,
        pcm_format: AudioFormat,
        queue: PlayerQueue,
    ) -> tuple[float, TermTiming, float] | None:
        """
        Return the start delay, the boundary timing and the pass time, or drop the term.

        The clip is rendered already, so no render time is left to spend: what decides is
        only whether its start still lies ahead of the frames the producer emitted.
        """
        current = queue.current_item
        if current is None:
            self._drop_term(queue_id, store, term, TermDropReason.FLOW_ENDED)
            return None
        items = self._dj_queue_items(queue_id)
        position = self._term_position(items, term, current.queue_item_id)
        if position == "behind":
            self._drop_term(queue_id, store, term, TermDropReason.SKIPPED)
            return None
        if not self._term_successor_follows(items, term):
            self._drop_term(queue_id, store, term, TermDropReason.NO_SUCCESSOR)
            return None
        speed = float(getattr(queue, "playback_speed", 1.0) or 1.0)
        now = asyncio.get_running_loop().time()
        if position == "current":
            elapsed = float(queue.corrected_elapsed_time or 0.0)
            if self._term_saw_a_seek(term, elapsed, now, speed):
                self._drop_term(queue_id, store, term, TermDropReason.SEEKED)
                return None
            term.last_elapsed = elapsed
            term.last_seen_at = now
        boundary_item = self._term_boundary_item(items, term, queue)
        if boundary_item is None:
            self._drop_term(queue_id, store, term, TermDropReason.SKIPPED)
            return None
        fade_seconds = self._term_fade_seconds(boundary_item)
        if fade_seconds is None:
            self._drop_term(queue_id, store, term, TermDropReason.UNKNOWN_BOUNDARY)
            return None
        remaining = self._term_seconds_to_boundary(items, queue, term, position, speed)
        if remaining is None:
            self._drop_term(queue_id, store, term, TermDropReason.NO_DURATION)
            return None
        timing = term_timing_for_remaining(
            remaining=remaining,
            clip_seconds=self._pcm_seconds(pcm, pcm_format),
            allow_pre=term.allow_pre,
            allow_post=term.allow_post,
            lead_seconds=self._term_head_lead(store.program),
            render_estimate=0.0,
            render_lead=self._term_render_lead(queue_id),
        )
        if timing.too_late:
            self._drop_term(queue_id, store, term, ScheduleReason.TOO_LATE)
            return None
        return max(0.0, timing.delay_seconds or 0.0), timing, now

    def _place_held_term(self, queue_id: str, store: TermStore, term: ProgramTerm) -> None:
        """Place a clip that was rendered before the playback could hear it."""
        pcm = term.rendered_pcm
        pcm_format = self.mass.streams.program_bus_format(queue_id)
        if pcm is None:
            return
        if pcm_format is None or pcm_format != term.rendered_format:
            # a clip rendered for another bus can not be mixed into this one
            self._drop_term(queue_id, store, term, ScheduleReason.BUS_REPLACED)
            return
        self._place_rendered_term(queue_id, store, term, pcm, pcm_format)

    def _term_estimate(self, queue_id: str) -> RenderEstimate:
        """Return the render estimate of a queue, creating it on first use."""
        estimates: dict[str, RenderEstimate] = getattr(self, "_term_estimates", {})
        if not hasattr(self, "_term_estimates"):
            self._term_estimates = estimates
        return estimates.setdefault(queue_id, RenderEstimate())

    @staticmethod
    def _term_head_lead(program: dict[str, Any] | None) -> float:
        """
        Return how far the program's host starts a break before its boundary.

        :param program: The station and host a queue's terms are rendered with, if known.
        """
        raw = (program or {}).get("term_head_seconds")
        return coerce_float(raw, TERM_HEAD_DEFAULT_SECONDS)

    def _term_render_lead(self, queue_id: str) -> float:
        """
        Return the seconds the queue's stream produced ahead of the listener.

        A lead the bus can not tell is taken as the bus's own minimum rather than as none,
        so a missing reading never moves a boundary into the frames the producer holds.

        :param queue_id: Queue whose stream to measure.
        """
        lead = self.mass.streams.program_bus_lead_seconds(queue_id)
        return max(LEAD_MIN_LEAD_SECONDS, lead or 0.0)

    def _term_fade_seconds(self, item: QueueItem) -> float | None:
        """
        Return how much earlier than the item's end the term's boundary lies, or None.

        The boundary is the item's end, so the answer is 0: a player may fade the handover,
        but a fade it only plans at the handover never reaches the stream, and a break timed
        to a fade that does not happen plays too early over the outgoing item. The host's
        lead is what makes a break overlap the tail, and it starts the clip that early
        whether the player fades or not.

        :param item: The item whose end the term speaks over.
        """
        media_type = getattr(item, "media_type", MediaType.TRACK)
        if media_type is MediaType.UNKNOWN:
            # the queue knows no media type here, so it says nothing about its boundary either
            return None
        return 0.0

    def _term_boundary_item(
        self, items: list[QueueItem], term: ProgramTerm, queue: PlayerQueue
    ) -> QueueItem | None:
        """Return the item whose end the term speaks over."""
        current = queue.current_item
        if current is not None and current.queue_item_id == term.queue_item_id:
            return current
        return next((item for item in items if item.queue_item_id == term.queue_item_id), None)

    def _result_seconds(self, result: ScheduleResult, pcm: bytes, pcm_format: AudioFormat) -> float:
        """Return the audio length the bus really took, in seconds."""
        if result.frames > 0:
            return result.frames / max(pcm_format.sample_rate, 1)
        return self._pcm_seconds(pcm, pcm_format)

    def _term_seconds_to_boundary(
        self,
        items: list[QueueItem],
        queue: PlayerQueue,
        term: ProgramTerm,
        position: TermPosition,
        playback_speed: float,
    ) -> float | None:
        """
        Return the seconds until the term's boundary, counting the items before it.

        The boundary is the end of the item the term speaks over: no fade shortens it, so
        the seconds left are the item's duration less the position in it.

        A term for a track further down the queue is timed from the rest of the running
        item plus every item in between, so a track shorter than a render is still served
        while its predecessor plays.
        """
        current = queue.current_item
        if current is None:
            return None
        remaining = boundary_remaining(
            duration=self._item_duration(current),
            elapsed=queue.corrected_elapsed_time,
            playback_speed=playback_speed,
        )
        if remaining is None:
            return None
        if position == "current":
            return remaining
        ids = [item.queue_item_id for item in items]
        if current.queue_item_id not in ids or term.queue_item_id not in ids:
            return None
        start = ids.index(current.queue_item_id) + 1
        end = ids.index(term.queue_item_id)
        if end < start:
            return None
        speed = playback_speed if playback_speed > 0 else 1.0
        for index in range(start, end):
            duration = self._item_duration(items[index])
            if duration is None:
                return None
            remaining += duration / speed
        term_item_duration = self._item_duration(items[end])
        if term_item_duration is None:
            return None
        term_item_remaining = boundary_remaining(
            duration=term_item_duration,
            elapsed=0.0,
            playback_speed=playback_speed,
        )
        if term_item_remaining is None:
            return None
        return remaining + term_item_remaining

    @staticmethod
    def _item_duration(item: QueueItem) -> float | None:
        """Return the duration of a queue item, from its stream or its metadata."""
        streamdetails = getattr(item, "streamdetails", None)
        duration = (getattr(streamdetails, "duration", None) if streamdetails else None) or getattr(
            item, "duration", None
        )
        return float(duration) if duration else None

    @staticmethod
    def _pcm_seconds(pcm: bytes, pcm_format: AudioFormat) -> float:
        """Return the length of the given PCM in seconds."""
        frame_size = max((pcm_format.bit_depth // 8) * pcm_format.channels, 1)
        return (len(pcm) // frame_size) / max(pcm_format.sample_rate, 1)

    def _term_position(
        self, items: list[QueueItem], term: ProgramTerm, current_item_id: str
    ) -> TermPosition:
        """Return whether the term's item is playing, still ahead, or already behind."""
        ids = [item.queue_item_id for item in items]
        if term.queue_item_id == current_item_id:
            return "current"
        if term.queue_item_id not in ids or current_item_id not in ids:
            # the item left the queue, so its boundary is gone; a playhead the queue does
            # not list is treated as current, since nothing can be said about its order
            return "behind" if term.queue_item_id not in ids else "current"
        if ids.index(term.queue_item_id) > ids.index(current_item_id):
            return "ahead"
        return "behind"

    @staticmethod
    def _term_successor_follows(items: list[QueueItem], term: ProgramTerm) -> bool:
        """Return whether the queue still holds the successor the term was decided for."""
        ids = [item.queue_item_id for item in items if not item.extra_attributes.get(ATTR_QUEUE_DJ)]
        if term.queue_item_id not in ids:
            return False
        following = ids[ids.index(term.queue_item_id) + 1 :]
        if term.successor_id is None:
            # the end of the queue is only a boundary while nothing follows the item
            return not following
        if term.successor_id == term.queue_item_id:
            # repeat-one: the same item follows itself, so the boundary stays its own
            return True
        return bool(following) and following[0] == term.successor_id

    @staticmethod
    def _term_saw_a_seek(
        term: ProgramTerm, elapsed: float, now: float, playback_speed: float
    ) -> bool:
        """Return whether the position moved against what plain playback explains."""
        if term.last_elapsed is None or term.last_seen_at is None:
            return False
        drift = elapsed - term.last_elapsed
        if drift < -SEEK_BACK_TOLERANCE_SECONDS:
            return True
        expected = max(0.0, now - term.last_seen_at) * playback_speed
        return drift > expected + SEEK_FORWARD_TOLERANCE_SECONDS

    def _book_term_pump(
        self, queue_id: str, delay: float, *, cap_seconds: float | None = None
    ) -> None:
        """
        Book the pass that serves the earliest due term of a queue.

        A queue keeps one pass that is no later than any boundary it holds: a term that is
        due further out does not move it, and a term that came sooner books its own. Every
        booking gets its own timer id, because a shared one would cancel the earlier timer
        and abort a pump that is already rendering.

        :param queue_id: The queue whose terms the pass serves.
        :param delay: Seconds from now the pass is wanted.
        :param cap_seconds: Furthest out the pass may be booked, for a wait that has to
            read its window again before it renders.
        """
        due_by_queue: dict[str, float] = getattr(self, "_term_pump_due", {})
        if not hasattr(self, "_term_pump_due"):
            self._term_pump_due = due_by_queue
        wait = delay if cap_seconds is None else min(delay, cap_seconds)
        wait = max(MIN_TERM_PUMP_SECONDS, wait)
        due = asyncio.get_running_loop().time() + wait
        booked = due_by_queue.get(queue_id)
        if booked is not None and booked <= due:
            # a pass that runs no later already covers this term
            return
        due_by_queue[queue_id] = due
        self.logger.debug("TERM-PUMP queue=%s books a pass in %.2fs", queue_id, wait)
        self.mass.call_later(
            wait,
            self._pump_terms,
            queue_id,
            task_id=f"ai_radio_term_pump_{queue_id}_{uuid4().hex[:8]}",
        )

    @staticmethod
    def _term_heard_seconds(queue: PlayerQueue, term: ProgramTerm, now: float) -> float:
        """
        Return how many stream seconds the term's item was heard since the last pass.

        The queue's position is media time, which a playback speed scales against the
        stream, and it stands still while the queue is paused - unlike the wall clock. A
        term whose own item already left the boundary is timed by the clock again.
        """
        if term.last_elapsed is None:
            return max(0.0, now - (term.last_seen_at or now))
        current = queue.current_item
        if current is None or current.queue_item_id != term.queue_item_id:
            return max(0.0, now - (term.last_seen_at or now))
        speed = float(getattr(queue, "playback_speed", 1.0) or 1.0)
        media = float(queue.corrected_elapsed_time or 0.0)
        return max(0.0, media - term.last_elapsed) / (speed if speed > 0 else 1.0)

    def _term_bus_grace_over(self, term: ProgramTerm, now: float) -> bool:
        """Return whether a term's wait for a bus it never saw has run out."""
        if term.bus_missing_since is None:
            term.bus_missing_since = now
            return False
        return now - term.bus_missing_since >= BUS_GRACE_SECONDS

    def _term_evicted(self, queue_id: str, term: ProgramTerm) -> None:
        """Stop a term the bounded store gave up and let go of what it had claimed."""
        self._cancel_term(queue_id, term, ScheduleReason.LEDGER_EVICTED)
        # the store handed the term back, so nothing else will release the clip a render left
        # in it; a term that never airs must not hold that audio for the rest of the session
        term.mark_dropped(ScheduleReason.LEDGER_EVICTED)
        self._forget_term_guard(queue_id, term)

    def _forget_term_guard(self, queue_id: str, term: ProgramTerm) -> None:
        """
        Let go of the guard weight a term that never airs claimed from the planner.

        What the term's boundary is afterwards is the store's answer, not this one's: a
        boundary that was merely missed is open there, one that can not be spoken is closed.

        :param queue_id: The queue the term belongs to.
        :param term: The term that never aired.
        """
        state = self._dj_queues.get(queue_id)
        if state is None:
            return
        for section in term.sections:
            for section_id, event in section.history_events:
                if (events := state.history.get(section_id)) and event in events:
                    # a break that never aired must not hold its successor back
                    events.remove(event)

    def _cancel_term(self, queue_id: str, term: ProgramTerm, reason: str | ScheduleReason) -> None:
        """Take a placed term back from the queue's program bus."""
        if term.scheduled_frame is None:
            return
        self.mass.streams.cancel_overlay(
            queue_id,
            OVERLAY_LABEL_TRANSITION,
            str(reason),
            start_frame=term.scheduled_frame,
        )

    def _drop_term(
        self, queue_id: str, store: TermStore, term: ProgramTerm, reason: str | ScheduleReason
    ) -> None:
        """End a term, taking a placed one back so it can not air after all."""
        if term.is_terminal:
            return
        self._cancel_term(queue_id, term, reason)
        store.drop(term.key, reason)
        self._forget_term_guard(queue_id, term)
        if str(reason) in PERMANENT_DROP_REASONS:
            # said once: the boundary is closed, so no later pass repeats the failure
            self.logger.debug(
                "Queue %s can not speak the boundary at %s (%s) - it stays closed",
                queue_id,
                term.successor_id or "the end of the queue",
                reason,
            )
            return
        self.logger.info(
            "Queue %s dropped the term at %s (%s)",
            queue_id,
            term.successor_id or "the end of the queue",
            reason,
        )

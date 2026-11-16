"""
Placement math and the answer a program's planner gets from the streams api.

The streams controller mixes a program into a live flow stream at a frame. Where that frame
lies follows from what the listener has heard, what the session already produced and the
minimum lead the bus needs; the arithmetic and the rules around it live here as pure
functions, so the controller only talks to the bus and the ledger.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING

from music_assistant.controllers.streams.constants import (
    LEAD_MIN_LEAD_SECONDS,
    MAX_OVERLAY_LATE_SECONDS,
)

if TYPE_CHECKING:
    from music_assistant.controllers.streams.audio import FlowProgramBus
    from music_assistant.controllers.streams.program_ledger import ProgramLedger


class ProgramPriority(IntEnum):
    """How strongly a program claims the frame window it is placed in."""

    CROSSPLAY = 1
    TERM = 2
    # a lead head is the front of a clip that continues outside the bus, so a term gives way
    LEAD = 3
    # an announcement is heard over everything else, even over a lead head
    ANNOUNCEMENT = 4


PRIORITY_CROSSPLAY = ProgramPriority.CROSSPLAY
PRIORITY_TERM = ProgramPriority.TERM
PRIORITY_LEAD = ProgramPriority.LEAD
PRIORITY_ANNOUNCEMENT = ProgramPriority.ANNOUNCEMENT


class ProgramKind(StrEnum):
    """What a program is for, so its rules do not have to be guessed from its rank."""

    CROSSPLAY = "crossplay"
    TERM = "term"
    # the head of a clip that continues outside the bus, ranked between term and announcement
    LEAD = "lead"
    # an announcement and an intro have no boundary they were written for, so a deep
    # buffer never drops them as late; an announcement may also be anchored again on the
    # bus that took over from a replaced one
    ANNOUNCEMENT = "announcement"
    INTRO = "intro"


# the rank each kind carries; a caller can not claim a rank its kind does not have
PRIORITY_BY_KIND: dict[ProgramKind, ProgramPriority] = {
    ProgramKind.CROSSPLAY: ProgramPriority.CROSSPLAY,
    ProgramKind.TERM: ProgramPriority.TERM,
    ProgramKind.LEAD: ProgramPriority.LEAD,
    ProgramKind.ANNOUNCEMENT: ProgramPriority.ANNOUNCEMENT,
    # an intro is heard over a term as well, so it shares the announcement's rank
    ProgramKind.INTRO: ProgramPriority.ANNOUNCEMENT,
}

# only a program written for a boundary can miss it
BOUNDARY_KINDS: frozenset[ProgramKind] = frozenset(
    {ProgramKind.CROSSPLAY, ProgramKind.TERM, ProgramKind.LEAD}
)

# what a jump in the queue leaves behind: every program that speaks for a moment or for a
# boundary the queue will not reach (next, previous, skip, a play of another item). An
# announcement is not one of them: it is heard over the stream, and the audio that was
# already rendered is anchored again on the bus the jump opens
KINDS_A_JUMP_DROPS: frozenset[ProgramKind] = frozenset(ProgramKind) - {ProgramKind.ANNOUNCEMENT}


def priority_for(kind: ProgramKind) -> ProgramPriority:
    """Return the rank a program of the given kind claims."""
    return PRIORITY_BY_KIND[kind]


class ScheduleReason(StrEnum):
    """Why a program was not placed on a queue's stream, or was moved aside for another."""

    NO_BUS = "no_bus"
    NO_AUDIBLE_POSITION = "no_audible_position"
    WINDOW_TAKEN = "window_taken"
    TOO_LATE = "too_late"
    FORMAT = "format"
    BUS_REPLACED = "bus_replaced"
    FLOW_ENDED = "flow_ended"
    HEAD_TRUNCATED = "head_truncated"
    UNKNOWN_BOUNDARY = "unknown_boundary"
    # the queue moved on to another item, so a boundary this program was written for is
    # never played; the command that asked for the move belongs in the log, not here
    SKIPPED = "skipped"
    # the queue jumped within the item or the stream, past where this program was anchored
    SEEKED = "seeked"
    # the ledger's bounded history had to give up a program that was still to be heard
    LEDGER_EVICTED = "ledger_evicted"
    # the item a lead's head was placed over ended before the head did, so what is left of it
    # would sound over the item after it
    HEAD_OUTLASTED = "head_outlasted"
    # the item's stream ended before the head it carried reached its anchor, so it is dropped
    ITEM_ABANDONED = "item_abandoned"


@dataclass(frozen=True)
class ScheduleResult:
    """What became of the program a caller asked to mix into a queue's stream."""

    scheduled: bool
    anchor_frame: int
    # frames really played, which is fewer than the program was offered with when the bus had
    # to drop its front; a program that was not placed reports the length it was offered with
    frames: int
    late_by_seconds: float
    lead_shortfall_seconds: float
    reason: ScheduleReason | None

    def __bool__(self) -> bool:
        """Return the outcome, so a caller that only tests the answer keeps working."""
        return self.scheduled


def placed(
    *,
    anchor_frame: int,
    frames: int,
    late_by_seconds: float,
    lead_shortfall_seconds: float,
    reason: ScheduleReason | None = None,
) -> ScheduleResult:
    """Return the answer for a program that was placed."""
    return ScheduleResult(
        scheduled=True,
        anchor_frame=anchor_frame,
        frames=frames,
        late_by_seconds=late_by_seconds,
        lead_shortfall_seconds=lead_shortfall_seconds,
        reason=reason,
    )


def refused(
    reason: ScheduleReason,
    *,
    frames: int = 0,
    anchor_frame: int = 0,
    late_by_seconds: float = 0.0,
    lead_shortfall_seconds: float = 0.0,
) -> ScheduleResult:
    """Return the answer for a program that was not placed."""
    return ScheduleResult(
        scheduled=False,
        anchor_frame=anchor_frame,
        frames=frames,
        late_by_seconds=late_by_seconds,
        lead_shortfall_seconds=lead_shortfall_seconds,
        reason=reason,
    )


@dataclass(frozen=True)
class PlacementPlan:
    """Where a program is anchored, and how far that is from the boundary it was planned for."""

    anchor_frame: int
    late_by_seconds: float
    lead_shortfall_seconds: float
    reason: ScheduleReason | None
    without_position: bool


def plan_placement(
    *,
    audible_frame: int | None,
    produced_frame: int,
    delay_frames: int,
    lead_frames: int,
    sample_rate: int,
) -> PlacementPlan:
    """
    Return the frame a program is placed at, and why it is not the planned one.

    Lateness is what the producer edge already emitted past the boundary; the minimum lead
    the bus needs is a placement rule and is reported separately, never as lateness. The two
    numbers do not add up to the anchor's offset from the boundary either: when the lead
    raises the anchor, it also overlaps the boundary the producer may already have passed.

    :param audible_frame: Frame the listener has reached, or None when that can not be told.
    :param produced_frame: Frame the session has produced so far.
    :param delay_frames: Frames from the listener to the boundary the program is planned for.
    :param lead_frames: Minimum lead the bus needs to mix a program in.
    :param sample_rate: Frames per second of the flow.
    """
    if audible_frame is None:
        if delay_frames > 0:
            # counting the delay from the producer's edge would miss the boundary by seconds
            return PlacementPlan(0, 0.0, 0.0, ScheduleReason.NO_AUDIBLE_POSITION, True)
        # an immediate program keeps the fallback the api always had: a whole lead after the
        # produced frames, which is the shortfall measured against the listener
        return PlacementPlan(
            produced_frame + lead_frames,
            0.0,
            lead_frames / sample_rate,
            None,
            True,
        )
    boundary = audible_frame + delay_frames
    clamped = max(produced_frame, boundary)
    return PlacementPlan(
        anchor_frame=max(clamped, audible_frame + lead_frames),
        late_by_seconds=(clamped - boundary) / sample_rate,
        lead_shortfall_seconds=max(0, lead_frames - delay_frames) / sample_rate,
        reason=None,
        without_position=False,
    )


class CollisionDecision(StrEnum):
    """What a placement does with the programs already in the window it wants."""

    PLACE = "place"
    SHIFT = "shift"
    EVICT = "evict"
    REFUSE = "refuse"


def settle_collisions(
    collisions: Sequence[ProgramPriority], priority: ProgramPriority
) -> CollisionDecision:
    """
    Return whether a program may take a window another program already occupies.

    :param collisions: Priorities of the programs whose window the placement overlaps.
    :param priority: Priority of the program that wants the window.
    """
    if not collisions:
        return CollisionDecision.PLACE
    if any(other > priority for other in collisions):
        return CollisionDecision.REFUSE
    if all(other < priority for other in collisions):
        return CollisionDecision.EVICT
    # same priority: the bus moves the new program past the window that is taken
    return CollisionDecision.SHIFT


@dataclass(frozen=True)
class ProgramWindow:
    """A program already scheduled on a bus, as the collision rules see it."""

    key: int
    label: str
    start_frame: int
    end_frame: int
    priority: ProgramPriority


def window_collisions(
    windows: Iterable[ProgramWindow], start_frame: int, frames: int
) -> list[ProgramWindow]:
    """
    Return the scheduled programs whose window a placement at the given frames overlaps.

    :param windows: The programs already scheduled on the bus.
    :param start_frame: Frame the new program would start at.
    :param frames: Number of frames the new program lasts.
    """
    return [
        window
        for window in windows
        if start_frame < window.end_frame and start_frame + frames > window.start_frame
    ]


def split_window(
    windows: Iterable[ProgramWindow], start_frame: int, frames: int, *, below: ProgramPriority
) -> tuple[list[ProgramWindow], list[ProgramWindow]]:
    """
    Split the scheduled programs into what stays and what gives way to the given priority.

    :param windows: The programs already scheduled on the bus.
    :param start_frame: Frame the program that claims the window starts at.
    :param frames: Number of frames the program claims.
    :param below: Only programs ranked below this priority give way.
    :return: The windows that were dropped, then the ones that stay.
    """
    colliding = window_collisions(windows, start_frame, frames)
    dropped = [window for window in colliding if window.priority < below]
    kept = [window for window in windows if window not in colliding or window.priority >= below]
    return dropped, kept


@dataclass(frozen=True)
class ProgramRequest:
    """What a caller wants mixed into a queue's stream."""

    queue_id: str
    label: str
    start_in_seconds: float
    priority: ProgramPriority
    kind: ProgramKind
    # seconds the program is meant to start before the boundary it belongs to
    head_seconds: float = 0.0


def place_program(
    program_bus: FlowProgramBus,
    pcm: bytes,
    *,
    request: ProgramRequest,
    audible_frame: int | None,
    ledger: ProgramLedger,
    logger: logging.Logger,
    report_no_position: Callable[[bool, float], None],
) -> ScheduleResult:
    """
    Place a program on a live flow session's bus and record what became of it.

    :param program_bus: The queue's live program bus.
    :param pcm: The program's audio, in the flow's PCM format.
    :param request: What the caller wants placed.
    :param audible_frame: Frame the listener has reached, or None when that can not be told.
    :param ledger: The queue's program ledger.
    :param logger: Logger of the streams controller.
    :param report_no_position: Called once with (ahead, seconds) when no position is known.
    """
    queue_id = request.queue_id
    label = request.label
    priority = request.priority
    kind = request.kind
    sample_rate = max(program_bus.pcm_format.sample_rate, 1)
    frame_size = program_bus.frame_size
    frames = len(pcm) // frame_size if frame_size else 0
    if not frames or not program_bus.can_mix:
        logger.debug(
            "PROGRAM-BUS %s can not be mixed into the %s of queue %s",
            label,
            program_bus.pcm_format.content_type,
            queue_id,
        )
        ledger.record_dropped(
            queue_id,
            label=label,
            frames=frames,
            priority=priority,
            kind=kind,
            reason=ScheduleReason.FORMAT,
            bus=program_bus,
        )
        return refused(ScheduleReason.FORMAT, frames=frames)
    delay_frames = round(max(0.0, request.start_in_seconds) * sample_rate)
    plan = plan_placement(
        audible_frame=audible_frame,
        produced_frame=program_bus.produced_frames,
        delay_frames=delay_frames,
        lead_frames=int(LEAD_MIN_LEAD_SECONDS * sample_rate),
        sample_rate=sample_rate,
    )
    if plan.without_position:
        report_no_position(
            delay_frames > 0,
            request.start_in_seconds if delay_frames > 0 else LEAD_MIN_LEAD_SECONDS,
        )
    if plan.reason is ScheduleReason.NO_AUDIBLE_POSITION:
        ledger.record_dropped(
            queue_id,
            label=label,
            frames=frames,
            priority=priority,
            kind=kind,
            reason=ScheduleReason.NO_AUDIBLE_POSITION,
            bus=program_bus,
        )
        return refused(ScheduleReason.NO_AUDIBLE_POSITION, frames=frames)
    if kind in BOUNDARY_KINDS and plan.late_by_seconds > MAX_OVERLAY_LATE_SECONDS:
        # a term belongs to a boundary, so a deep buffer already past it is lateness even
        # without a delay; an announcement or intro has no boundary to miss
        logger.debug(
            "PROGRAM-BUS %s is %.2fs late for queue %s - not placed",
            label,
            plan.late_by_seconds,
            queue_id,
        )
        ledger.record_dropped(
            queue_id,
            label=label,
            frames=frames,
            priority=priority,
            kind=kind,
            reason=ScheduleReason.TOO_LATE,
            start_frame=plan.anchor_frame,
            bus=program_bus,
        )
        return refused(
            ScheduleReason.TOO_LATE,
            frames=frames,
            anchor_frame=plan.anchor_frame,
            late_by_seconds=plan.late_by_seconds,
            lead_shortfall_seconds=plan.lead_shortfall_seconds,
        )
    collisions = window_collisions(program_bus.scheduled_windows(), plan.anchor_frame, frames)
    decision = settle_collisions([window.priority for window in collisions], priority)
    if decision is CollisionDecision.REFUSE:
        logger.debug(
            "PROGRAM-BUS %s keeps the window of queue %s it is ranked above",
            label,
            queue_id,
        )
        ledger.record_dropped(
            queue_id,
            label=label,
            frames=frames,
            priority=priority,
            kind=kind,
            reason=ScheduleReason.WINDOW_TAKEN,
            start_frame=plan.anchor_frame,
            bus=program_bus,
        )
        return refused(
            ScheduleReason.WINDOW_TAKEN,
            frames=frames,
            anchor_frame=plan.anchor_frame,
            late_by_seconds=plan.late_by_seconds,
            lead_shortfall_seconds=plan.lead_shortfall_seconds,
        )
    if decision is CollisionDecision.EVICT:
        for dropped_start, dropped_label in program_bus.drop_window(
            plan.anchor_frame, frames, below=priority
        ):
            ledger.mark_dropped(
                queue_id,
                dropped_label,
                ScheduleReason.WINDOW_TAKEN,
                start_frame=dropped_start,
            )
    placement = program_bus.place(pcm, plan.anchor_frame, label=label, priority=priority)
    if placement is None:
        # the bus knows whether its window was taken or the frame already went out
        reason = program_bus.last_refusal or ScheduleReason.TOO_LATE
        ledger.record_dropped(
            queue_id,
            label=label,
            frames=frames,
            priority=priority,
            kind=kind,
            reason=reason,
            start_frame=plan.anchor_frame,
            bus=program_bus,
        )
        return refused(
            reason,
            frames=frames,
            anchor_frame=plan.anchor_frame,
            late_by_seconds=plan.late_by_seconds,
            lead_shortfall_seconds=plan.lead_shortfall_seconds,
        )
    start_frame = placement.start_frame
    # the schedule moved the program past a window it could not take; that is the only reason
    # window_taken is reported here. The bus moves the anchor itself too - it cuts the front of
    # a program whose start was already produced and answers with what is left - but that
    # shows in the placement, not in this reason
    shift_reason: ScheduleReason | None = (
        ScheduleReason.WINDOW_TAKEN if decision is CollisionDecision.SHIFT else None
    )
    # a program without a boundary can not be late for one, however far the producer is;
    # the ledger and the answer to the caller say the same thing
    late_by_seconds = plan.late_by_seconds if kind in BOUNDARY_KINDS else 0.0
    ledger.record_scheduled(
        queue_id,
        label=label,
        start_frame=start_frame,
        frames=placement.frames,
        priority=priority,
        kind=kind,
        reason=shift_reason,
        bus=program_bus,
        head_seconds=request.head_seconds,
        late_by_seconds=late_by_seconds,
    )
    logger.debug(
        "PROGRAM-BUS placed %s on queue %s at frame %d (%.2fs late, %s)",
        label,
        queue_id,
        start_frame,
        late_by_seconds,
        shift_reason or "as planned",
    )
    return placed(
        anchor_frame=start_frame,
        # the placement is the answer: a bus that had to drop the front of the program
        # plays less of it than was offered, and a caller that ends a deadline after this
        # length must not count the lateness a second time
        frames=placement.frames,
        late_by_seconds=late_by_seconds,
        lead_shortfall_seconds=plan.lead_shortfall_seconds,
        reason=shift_reason,
    )

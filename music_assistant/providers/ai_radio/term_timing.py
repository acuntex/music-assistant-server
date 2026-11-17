"""
Boundary arithmetic for the AI DJ's program terms.

A term is planned for the end of a running item, and the only thing the streams api needs
from the provider is the delay until that end. Everything that turns a queue position and
a clip into that delay - the playback speed, the head of a break and the render budget -
is pure arithmetic, so it lives here and not in the queue wiring. A crossfade is only
subtracted where a caller knows one: a term takes the item's end, because a fade the player
plans at the handover is no boundary it could be timed to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from music_assistant.controllers.streams.constants import MAX_OVERLAY_LATE_SECONDS

from .constants import TALK_OVER_MAX_PRE, TERM_HEAD_DEFAULT_SECONDS

# how long a render is assumed to take: a starting point, measured renders replace it.
# the budget is what keeps an engine call from being paid for a boundary that is already
# too close to reach, so an estimate that is off only costs a term, never an item
RENDER_ESTIMATE_SECONDS = 6.0
MIN_RENDER_ESTIMATE_SECONDS = 3.0
MAX_RENDER_ESTIMATE_SECONDS = 20.0
# how much of a measured render replaces the estimate. a single slow call is usually the
# engine having a bad moment, a run of them is the new normal
RENDER_ESTIMATE_SMOOTHING = 0.3
# a term may miss its boundary by this much and still count as placed there; starting
# earlier is always fine, so the tolerance is one-sided
TERM_SCHEDULE_TOLERANCE_SECONDS = MAX_OVERLAY_LATE_SECONDS
# how much wider than the bare render time a term's window is. a pass comes due a whole
# pump period late at worst (TERM_PUMP_MAX_SECONDS, the pump's own booking cap), so a window
# only as wide as the render itself is skipped by a pass, or entered in its middle - and a
# render that starts in the middle of it lands after the boundary it was written for
TERM_RENDER_MARGIN_SECONDS = 15.0
# how much earlier than its head a term is placed: the bus needs its own lead on top of
# the head, and starting a moment early is always allowed, starting late is not
TERM_START_BIAS_SECONDS = 0.25


@dataclass(frozen=True)
class TermTiming:
    """Where a term's boundary lies, and whether it can still be reached."""

    # seconds from the listener to the boundary the term speaks at, or None when unknown
    remaining_seconds: float | None
    # seconds of the clip that play before the boundary: the host's lead, cut to the clip
    head_seconds: float
    # seconds from the listener to the term's start; negative only for a truncated head
    delay_seconds: float | None
    # seconds after the scheduled start by which the term has been heard to its end
    deadline_seconds: float | None
    # the boundary lies closer than the head, so the head is cut short (not clamped)
    head_truncated: bool
    # the boundary is close enough that rendering has to start now
    render_due: bool
    # the boundary is closer than a render takes, so nothing is rendered at all
    too_late: bool
    # seconds to wait before a pass can render this term, or None without a known boundary
    render_in_seconds: float | None
    # the boundary can not be estimated in phase 1 (an item whose media type is unknown)
    unknown_boundary: bool
    # the item reports no duration, so it has no boundary to time
    no_duration: bool


def boundary_remaining(
    *,
    duration: float | None,
    elapsed: float | None,
    playback_speed: float = 1.0,
    fade_seconds: float = 0.0,
) -> float | None:
    """
    Return the seconds until the next item starts, or None when that can not be told.

    The boundary is not the end of the running item: a known crossfade starts the next
    item that much earlier. Streams position is media time, which a playback speed scales
    against the stream, so the result is divided by that speed.

    :param duration: Duration of the running item in seconds, or None when it has none.
    :param elapsed: Position in the running item in seconds, or None when it is not known.
    :param playback_speed: Speed the item is played at, as a factor above 0.
    :param fade_seconds: Known length of the crossfade into the next item.
    """
    if not duration or elapsed is None:
        return None
    speed = playback_speed if playback_speed > 0 else 1.0
    return max(0.0, (float(duration) - max(0.0, float(elapsed)) - max(0.0, fade_seconds))) / speed


def term_head_seconds(
    *,
    clip_seconds: float,
    allow_pre: bool,
    lead_seconds: float = TERM_HEAD_DEFAULT_SECONDS,
    allow_post: bool = True,
    remaining_seconds: float | None = None,
) -> float:
    """
    Return how much of a term plays before its boundary.

    The host owns that share: the lead is a plain number of seconds, so a break starts as
    early as the host asked for and only a clip shorter than the lead starts as early as it
    lasts. A section that may not start over the outgoing tail has no head at all, and the
    lead ceiling holds whatever a caller hands in.

    A section that may not carry over the boundary is the other way round: its whole clip
    has to play before the boundary, so the clip is its own head. The room in front of the
    boundary is all there is, so the head never reaches further back than that; a clip that
    no longer fits leaves no head the boundary can hold and the term is too late.

    :param clip_seconds: Length of the rendered term in seconds.
    :param allow_pre: Whether the section may start over the outgoing item's tail.
    :param lead_seconds: Seconds before the boundary the host wants a break to start.
    :param allow_post: Whether the section may carry over the boundary.
    :param remaining_seconds: Seconds to the boundary, or None while it is not known.
    """
    clip = max(0.0, float(clip_seconds))
    if not allow_post:
        head = clip
        if remaining_seconds is not None:
            head = min(head, max(0.0, float(remaining_seconds)))
        return head
    if not allow_pre:
        return 0.0
    lead = min(max(0.0, float(lead_seconds)), TALK_OVER_MAX_PRE)
    return min(clip, lead)


def term_timing(
    *,
    duration: float | None,
    elapsed: float | None,
    fade_seconds: float | None,
    clip_seconds: float,
    allow_pre: bool,
    lead_seconds: float = TERM_HEAD_DEFAULT_SECONDS,
    allow_post: bool = True,
    render_estimate: float,
    render_lead: float = 0.0,
    tolerance: float = TERM_SCHEDULE_TOLERANCE_SECONDS,
) -> TermTiming:
    """
    Return everything the engine needs to decide about one boundary.

    :param duration: Duration of the item the term speaks over, or None.
    :param elapsed: Position in that item in seconds, or None.
    :param fade_seconds: Known crossfade length, or None when the boundary is not known.
    :param clip_seconds: Length of the rendered term in seconds, in which the head lies.
    :param allow_pre: Whether the section may start over the outgoing item's tail.
    :param lead_seconds: Seconds before the boundary the host wants a break to start.
    :param allow_post: Whether the section may carry over the boundary.
    :param render_estimate: Seconds a render is currently assumed to take.
    :param render_lead: Seconds the queue's producer already runs ahead of the listener.
    :param tolerance: How far a placed term may miss its boundary and still count as placed.
    """
    head = term_head_seconds(
        clip_seconds=clip_seconds,
        allow_pre=allow_pre,
        lead_seconds=lead_seconds,
        allow_post=allow_post,
    )
    if fade_seconds is None:
        return TermTiming(
            remaining_seconds=None,
            head_seconds=head,
            delay_seconds=None,
            deadline_seconds=None,
            head_truncated=False,
            render_due=False,
            too_late=False,
            render_in_seconds=None,
            unknown_boundary=True,
            no_duration=False,
        )
    remaining = boundary_remaining(duration=duration, elapsed=elapsed, fade_seconds=fade_seconds)
    if remaining is None:
        return TermTiming(
            remaining_seconds=None,
            head_seconds=head,
            delay_seconds=None,
            deadline_seconds=None,
            head_truncated=False,
            render_due=False,
            too_late=False,
            render_in_seconds=None,
            unknown_boundary=False,
            no_duration=True,
        )
    return term_timing_for_remaining(
        remaining=remaining,
        clip_seconds=clip_seconds,
        allow_pre=allow_pre,
        lead_seconds=lead_seconds,
        allow_post=allow_post,
        render_estimate=render_estimate,
        render_lead=render_lead,
        tolerance=tolerance,
    )


def term_timing_for_remaining(
    *,
    remaining: float,
    clip_seconds: float,
    allow_pre: bool,
    lead_seconds: float = TERM_HEAD_DEFAULT_SECONDS,
    allow_post: bool = True,
    render_estimate: float,
    render_lead: float = 0.0,
    tolerance: float = TERM_SCHEDULE_TOLERANCE_SECONDS,
) -> TermTiming:
    """
    Return the timing of a boundary that is the given seconds away.

    A boundary further down the queue counts the rest of the running item and every item
    before it, so a short track is rendered while its predecessor still plays.

    A stream produces ahead of what is heard, and the bus can not mix a program into frames
    it already emitted: that lead is as much part of the render window as the head is, or
    the term is rendered while the boundary it was written for is already produced. The
    window also carries a full pump period of reserve, so a pass that only comes due a
    period late still enters the window at its edge instead of in its middle.

    :param remaining: Seconds from the listener to the boundary.
    :param clip_seconds: Length of the rendered term in seconds, in which the head lies.
    :param allow_pre: Whether the section may start over the outgoing item's tail.
    :param lead_seconds: Seconds before the boundary the host wants a break to start.
    :param allow_post: Whether the section may carry over the boundary.
    :param render_estimate: Seconds a render is currently assumed to take.
    :param render_lead: Seconds the queue's producer already runs ahead of the listener.
    :param tolerance: How far a placed term may miss its boundary and still count as placed.
    """
    head = term_head_seconds(
        clip_seconds=clip_seconds,
        allow_pre=allow_pre,
        lead_seconds=lead_seconds,
        allow_post=allow_post,
        remaining_seconds=remaining,
    )
    # a section that may not carry over the boundary is placed so that it ends on it: the
    # bias that starts a breaking term a moment early would push its end past the boundary,
    # and a negative delay is clamped to zero, which moves the whole clip that much later
    bias = 0.0 if not allow_post else TERM_START_BIAS_SECONDS
    delay = remaining - head - bias
    # a negative delay is the truncated head: the api clamps it to zero rather than the
    # term pretending that a head longer than the boundary still fits
    # a render may start within the lateness a placed term is still allowed, so a pass
    # that lands a moment after its budget window does not give up a reachable boundary.
    # the head is part of the budget: a term that starts that much before the boundary has
    # that much less time to be rendered in
    too_late = remaining < head + render_lead + render_estimate - tolerance
    window = head + render_lead + render_estimate + tolerance + TERM_RENDER_MARGIN_SECONDS
    return TermTiming(
        remaining_seconds=remaining,
        head_seconds=head,
        delay_seconds=delay,
        deadline_seconds=max(0.0, delay) + max(0.0, float(clip_seconds)) + tolerance,
        head_truncated=head > remaining,
        render_due=not too_late and remaining <= window,
        too_late=too_late,
        # the booking is the window itself, so the next pass lands exactly where it is due
        render_in_seconds=max(0.0, remaining - window),
        unknown_boundary=False,
        no_duration=False,
    )


@dataclass
class RenderEstimate:
    """An exponentially smoothed guess at how long a term render takes."""

    seconds: float = RENDER_ESTIMATE_SECONDS
    samples: int = field(default=0)

    def record(self, measured_seconds: float) -> None:
        """
        Fold one measured render into the estimate.

        :param measured_seconds: How long the render that just finished took.
        """
        measured = max(0.0, float(measured_seconds))
        smoothed = render_estimate_step(self.seconds, measured)
        self.seconds = smoothed
        self.samples += 1


def render_estimate_step(current: float, measured: float) -> float:
    """
    Return the estimate after one measured render.

    :param current: The estimate in use before the measurement.
    :param measured: How long the render took.
    """
    smoothed = (1.0 - RENDER_ESTIMATE_SMOOTHING) * current + RENDER_ESTIMATE_SMOOTHING * measured
    return min(MAX_RENDER_ESTIMATE_SECONDS, max(MIN_RENDER_ESTIMATE_SECONDS, smoothed))

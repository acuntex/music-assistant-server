"""Unit tests for the boundary arithmetic behind the AI DJ's program terms."""

from __future__ import annotations

import pytest

from music_assistant.providers.ai_radio.constants import (
    TALK_OVER_MAX_PRE,
    TERM_HEAD_DEFAULT_SECONDS,
    TERM_HEAD_MAX_SECONDS,
)
from music_assistant.providers.ai_radio.term_engine import TERM_PUMP_MAX_SECONDS
from music_assistant.providers.ai_radio.term_timing import (
    MAX_RENDER_ESTIMATE_SECONDS,
    MIN_RENDER_ESTIMATE_SECONDS,
    RENDER_ESTIMATE_SECONDS,
    TERM_RENDER_MARGIN_SECONDS,
    TERM_SCHEDULE_TOLERANCE_SECONDS,
    TERM_START_BIAS_SECONDS,
    RenderEstimate,
    boundary_remaining,
    term_head_seconds,
    term_timing,
    term_timing_for_remaining,
)

# the lead a live queue reported while its producer ran ahead of the listener
PRODUCER_LEAD_SECONDS = 12.34
# the live boundary that was lost: 45.61s to the boundary, a 2s head, a 5.68s render and a
# producer that already led by 37.01s
LIVE_REMAINING_SECONDS = 45.61
LIVE_ESTIMATE_SECONDS = 5.68
LIVE_LEAD_SECONDS = 37.01
# the render the pass started took this long, and the buffer grew by half a second in it
LIVE_RENDER_SECONDS = 5.81
LIVE_LEAD_GROWTH_SECONDS = 0.5


def test_remaining_is_the_media_time_left() -> None:
    """The rest of a running item is its duration minus what was played."""
    assert boundary_remaining(duration=200.0, elapsed=120.0) == pytest.approx(80.0)


def test_remaining_is_divided_by_the_playback_speed() -> None:
    """A track played faster reaches its boundary sooner."""
    assert boundary_remaining(duration=200.0, elapsed=120.0, playback_speed=1.25) == pytest.approx(
        64.0
    )


def test_remaining_subtracts_the_known_crossfade() -> None:
    """The next item starts a known fade before the running one ends."""
    assert boundary_remaining(duration=200.0, elapsed=120.0, fade_seconds=8.0) == pytest.approx(
        72.0
    )


def test_remaining_is_never_negative() -> None:
    """An item already past its end has no boundary left."""
    assert boundary_remaining(duration=200.0, elapsed=250.0, fade_seconds=8.0) == 0.0


def test_remaining_without_a_duration_is_unknown() -> None:
    """A dynamic source that reports no duration can not be timed."""
    assert boundary_remaining(duration=None, elapsed=12.0) is None


def test_remaining_without_a_playhead_is_unknown() -> None:
    """A queue that has not reported its position can not be timed either."""
    assert boundary_remaining(duration=200.0, elapsed=None) is None


def test_head_is_zero_without_allow_pre() -> None:
    """A break that may not start over the outgoing tail has no head."""
    assert term_head_seconds(clip_seconds=30.0, allow_pre=False) == 0.0


def test_the_default_head_is_two_seconds() -> None:
    """A host that never set a lead starts its breaks two seconds before the boundary."""
    assert TERM_HEAD_DEFAULT_SECONDS == 2.0
    assert term_head_seconds(clip_seconds=8.0, allow_pre=True) == pytest.approx(2.0)


def test_the_head_is_the_lead_the_host_set() -> None:
    """The configured lead is the head, however long the clip is."""
    assert term_head_seconds(clip_seconds=8.0, allow_pre=True, lead_seconds=5.0) == pytest.approx(
        5.0
    )


def test_a_lead_of_zero_starts_the_break_on_the_boundary() -> None:
    """A host that wants no lead lets the break start with the crossfade itself."""
    assert term_head_seconds(clip_seconds=8.0, allow_pre=True, lead_seconds=0.0) == 0.0


def test_a_clip_shorter_than_the_lead_starts_as_early_as_it_lasts() -> None:
    """A break shorter than its lead can not start before it does."""
    assert term_head_seconds(clip_seconds=3.0, allow_pre=True, lead_seconds=5.0) == pytest.approx(
        3.0
    )


def test_the_head_is_capped_at_the_pre_ceiling() -> None:
    """The ceiling holds even when a caller hands in a longer lead."""
    assert term_head_seconds(
        clip_seconds=30.0, allow_pre=True, lead_seconds=TERM_HEAD_MAX_SECONDS
    ) == pytest.approx(TALK_OVER_MAX_PRE)
    assert term_head_seconds(
        clip_seconds=30.0, allow_pre=True, lead_seconds=TERM_HEAD_MAX_SECONDS * 4
    ) == pytest.approx(TALK_OVER_MAX_PRE)


def test_without_allow_pre_the_lead_is_ignored() -> None:
    """A section that may not start over the outgoing tail has no head for any lead."""
    assert term_head_seconds(clip_seconds=8.0, allow_pre=False, lead_seconds=12.0) == 0.0


def test_without_allow_post_the_whole_clip_is_the_head() -> None:
    """A break that must end before the boundary starts its own length before it."""
    assert term_head_seconds(
        clip_seconds=8.0, allow_pre=True, lead_seconds=2.0, allow_post=False
    ) == pytest.approx(8.0)
    # the host's lead is no head for a break that has to be done at the boundary
    assert term_head_seconds(
        clip_seconds=8.0, allow_pre=False, lead_seconds=12.0, allow_post=False
    ) == pytest.approx(8.0)


def test_a_head_that_may_not_post_never_reaches_behind_the_boundary() -> None:
    """The room in front of the boundary caps the head, so a start never lands in the past."""
    assert term_head_seconds(
        clip_seconds=20.0,
        allow_pre=False,
        allow_post=False,
        remaining_seconds=5.0,
    ) == pytest.approx(5.0)


def test_a_negative_clip_length_has_no_head() -> None:
    """A clip of no audio has no head to play before the boundary."""
    assert term_head_seconds(clip_seconds=-3.0, allow_pre=True) == 0.0


def test_a_negative_lead_has_no_head() -> None:
    """A lead below zero is no lead, never a head behind the boundary."""
    assert term_head_seconds(clip_seconds=8.0, allow_pre=True, lead_seconds=-3.0) == 0.0


def test_the_delay_is_the_remaining_time_minus_the_head() -> None:
    """The api counts its delay from the boundary, so the head comes off it."""
    timing = term_timing(
        duration=100.0,
        elapsed=60.0,
        fade_seconds=0.0,
        clip_seconds=20.0,
        allow_pre=True,
        render_estimate=RENDER_ESTIMATE_SECONDS,
    )
    assert timing.remaining_seconds == pytest.approx(40.0)
    assert timing.head_seconds == pytest.approx(TERM_HEAD_DEFAULT_SECONDS)
    assert timing.delay_seconds == pytest.approx(
        40.0 - TERM_HEAD_DEFAULT_SECONDS - TERM_START_BIAS_SECONDS
    )
    assert timing.head_truncated is False
    assert timing.unknown_boundary is False


def test_a_break_starts_the_lead_the_host_set_before_its_boundary() -> None:
    """The delay of a leading break is the remaining time minus its host's lead."""
    timing = term_timing_for_remaining(
        remaining=30.0,
        clip_seconds=8.62,
        allow_pre=True,
        lead_seconds=5.0,
        render_estimate=RENDER_ESTIMATE_SECONDS,
    )
    assert timing.head_seconds == pytest.approx(5.0)
    assert timing.delay_seconds == pytest.approx(30.0 - 5.0 - TERM_START_BIAS_SECONDS)


def test_a_lead_of_zero_puts_the_break_on_the_boundary() -> None:
    """No lead means the break starts with the boundary, its clip running on past it."""
    timing = term_timing_for_remaining(
        remaining=30.0,
        clip_seconds=8.62,
        allow_pre=True,
        lead_seconds=0.0,
        render_estimate=RENDER_ESTIMATE_SECONDS,
    )
    assert timing.head_seconds == 0.0
    assert timing.delay_seconds == pytest.approx(30.0 - TERM_START_BIAS_SECONDS)


def test_the_start_carries_a_quarter_second_bias() -> None:
    """A term starts a moment before its head so the voice is in before the boundary."""
    timing = term_timing(
        duration=100.0,
        elapsed=60.0,
        fade_seconds=0.0,
        clip_seconds=10.0,
        allow_pre=False,
        render_estimate=RENDER_ESTIMATE_SECONDS,
    )
    assert timing.delay_seconds == pytest.approx(40.0 - TERM_START_BIAS_SECONDS)


def test_a_head_longer_than_the_boundary_is_truncated_not_clamped() -> None:
    """A boundary closer than the head keeps its negative delay, flagged as truncated."""
    timing = term_timing(
        duration=100.0,
        elapsed=97.0,
        fade_seconds=0.0,
        clip_seconds=30.0,
        allow_pre=True,
        lead_seconds=TALK_OVER_MAX_PRE,
        render_estimate=RENDER_ESTIMATE_SECONDS,
    )
    assert timing.head_truncated is True
    # the api clamps a negative delay to zero itself; the term does not pretend the head fits
    assert timing.delay_seconds is not None
    assert timing.delay_seconds < 0


def test_a_boundary_without_a_known_fade_is_unknown() -> None:
    """A fade length nothing can tell is not estimated."""
    timing = term_timing(
        duration=100.0,
        elapsed=60.0,
        fade_seconds=None,
        clip_seconds=10.0,
        allow_pre=False,
        render_estimate=RENDER_ESTIMATE_SECONDS,
    )
    assert timing.unknown_boundary is True
    assert timing.remaining_seconds is None
    assert timing.delay_seconds is None


def test_an_item_without_a_duration_marks_no_duration() -> None:
    """A dynamic source is not timed at all."""
    timing = term_timing(
        duration=None,
        elapsed=0.0,
        fade_seconds=0.0,
        clip_seconds=10.0,
        allow_pre=False,
        render_estimate=RENDER_ESTIMATE_SECONDS,
    )
    assert timing.no_duration is True
    assert timing.remaining_seconds is None


def test_the_deadline_counts_from_the_scheduled_start() -> None:
    """A term is done when its start, its clip and the tolerance have passed."""
    timing = term_timing(
        duration=100.0,
        elapsed=60.0,
        fade_seconds=0.0,
        clip_seconds=10.0,
        allow_pre=False,
        render_estimate=RENDER_ESTIMATE_SECONDS,
    )
    assert timing.delay_seconds == pytest.approx(40.0 - TERM_START_BIAS_SECONDS)
    assert timing.deadline_seconds == pytest.approx(
        40.0 - TERM_START_BIAS_SECONDS + 10.0 + TERM_SCHEDULE_TOLERANCE_SECONDS
    )


def test_the_deadline_of_a_late_start_counts_from_now() -> None:
    """A start that can not be met still ends after the clip plus the tolerance."""
    timing = term_timing(
        duration=100.0,
        elapsed=97.0,
        fade_seconds=0.0,
        clip_seconds=30.0,
        allow_pre=True,
        lead_seconds=TALK_OVER_MAX_PRE,
        render_estimate=RENDER_ESTIMATE_SECONDS,
    )
    assert timing.deadline_seconds == pytest.approx(30.0 + TERM_SCHEDULE_TOLERANCE_SECONDS)


def test_the_render_waits_until_the_budget_window() -> None:
    """A term far from its boundary is not rendered yet."""
    timing = term_timing(
        duration=100.0,
        elapsed=60.0,
        fade_seconds=0.0,
        clip_seconds=10.0,
        allow_pre=False,
        render_estimate=RENDER_ESTIMATE_SECONDS,
    )
    assert timing.render_due is False
    assert timing.too_late is False


def test_the_render_starts_at_the_head_plus_the_estimate() -> None:
    """A term whose boundary is within its head and render time renders now."""
    timing = term_timing(
        duration=100.0,
        elapsed=82.0,
        fade_seconds=0.0,
        clip_seconds=30.0,
        allow_pre=True,
        lead_seconds=TALK_OVER_MAX_PRE,
        render_estimate=RENDER_ESTIMATE_SECONDS,
    )
    # 18s left, a 12s head and a 6s render: this is exactly the last moment to start
    assert timing.remaining_seconds == pytest.approx(18.0)
    assert timing.render_due is True
    assert timing.too_late is False


def test_a_render_that_can_not_finish_in_time_is_too_late() -> None:
    """Below the render estimate the term is given up without spending an engine call."""
    timing = term_timing(
        duration=100.0,
        elapsed=98.0,
        fade_seconds=0.0,
        clip_seconds=10.0,
        allow_pre=False,
        render_estimate=RENDER_ESTIMATE_SECONDS,
    )
    assert timing.remaining_seconds == pytest.approx(2.0)
    assert timing.too_late is True
    assert timing.render_due is False


def test_a_render_may_start_within_the_lateness_tolerance() -> None:
    """A pass that lands a moment after its budget window still tries the boundary."""
    timing = term_timing(
        duration=100.0,
        elapsed=94.5,
        fade_seconds=0.0,
        clip_seconds=10.0,
        allow_pre=False,
        render_estimate=RENDER_ESTIMATE_SECONDS,
    )
    # half a second less than the estimate: the term may still miss the boundary by <= 1s
    assert timing.remaining_seconds == pytest.approx(5.5)
    assert timing.too_late is False
    assert timing.render_due is True


def test_a_boundary_exactly_at_the_estimate_still_renders() -> None:
    """The budget boundary itself is not too late."""
    timing = term_timing(
        duration=100.0,
        elapsed=94.0,
        fade_seconds=0.0,
        clip_seconds=10.0,
        allow_pre=False,
        render_estimate=RENDER_ESTIMATE_SECONDS,
    )
    assert timing.remaining_seconds == pytest.approx(6.0)
    assert timing.too_late is False
    assert timing.render_due is True


def test_a_producer_lead_widens_the_render_window() -> None:
    """A bus that already produced 12.34s ahead opens its window that much earlier."""
    produced = term_timing_for_remaining(
        remaining=40.0,
        clip_seconds=30.0,
        allow_pre=True,
        lead_seconds=TALK_OVER_MAX_PRE,
        render_estimate=RENDER_ESTIMATE_SECONDS,
        render_lead=PRODUCER_LEAD_SECONDS,
    )
    unproduced = term_timing_for_remaining(
        remaining=40.0,
        clip_seconds=30.0,
        allow_pre=True,
        lead_seconds=TALK_OVER_MAX_PRE,
        render_estimate=RENDER_ESTIMATE_SECONDS,
        render_lead=0.0,
    )
    assert produced.head_seconds == pytest.approx(TALK_OVER_MAX_PRE)
    # 40s left of a 12s head, a 6s render and 12.34s of produced audio: still reachable
    assert produced.render_due is True
    assert produced.too_late is False
    # the very same boundary is more than a window away without the lead
    assert unproduced.render_due is False
    assert unproduced.too_late is False


def test_a_boundary_inside_the_produced_lead_is_too_late() -> None:
    """A render would land in frames the producer emitted, so it is not paid for."""
    timing = term_timing_for_remaining(
        remaining=17.0,
        clip_seconds=30.0,
        allow_pre=True,
        lead_seconds=TALK_OVER_MAX_PRE,
        render_estimate=RENDER_ESTIMATE_SECONDS,
        render_lead=PRODUCER_LEAD_SECONDS,
    )
    assert timing.too_late is True
    assert timing.render_due is False


def test_the_late_threshold_is_the_head_the_lead_and_the_render_budget() -> None:
    """The window shuts head+lead+estimate-tolerance before the boundary, not a moment later."""
    window = (
        TERM_HEAD_DEFAULT_SECONDS
        + PRODUCER_LEAD_SECONDS
        + RENDER_ESTIMATE_SECONDS
        - TERM_SCHEDULE_TOLERANCE_SECONDS
    )
    just_in = term_timing_for_remaining(
        remaining=window + 0.01,
        clip_seconds=30.0,
        allow_pre=True,
        render_estimate=RENDER_ESTIMATE_SECONDS,
        render_lead=PRODUCER_LEAD_SECONDS,
    )
    just_out = term_timing_for_remaining(
        remaining=window - 0.01,
        clip_seconds=30.0,
        allow_pre=True,
        render_estimate=RENDER_ESTIMATE_SECONDS,
        render_lead=PRODUCER_LEAD_SECONDS,
    )
    assert just_in.too_late is False
    assert just_out.too_late is True


def test_without_a_known_lead_the_window_is_the_one_the_item_path_had() -> None:
    """No bus lead means no wider window: every value stays what it was before."""
    timing = term_timing_for_remaining(
        remaining=18.0,
        clip_seconds=30.0,
        allow_pre=True,
        lead_seconds=TALK_OVER_MAX_PRE,
        render_estimate=RENDER_ESTIMATE_SECONDS,
        render_lead=0.0,
    )
    assert timing.remaining_seconds == pytest.approx(18.0)
    assert timing.head_seconds == pytest.approx(TALK_OVER_MAX_PRE)
    assert timing.delay_seconds == pytest.approx(18.0 - TALK_OVER_MAX_PRE - TERM_START_BIAS_SECONDS)
    assert timing.deadline_seconds == pytest.approx(
        18.0 - TALK_OVER_MAX_PRE - TERM_START_BIAS_SECONDS + 30.0 + TERM_SCHEDULE_TOLERANCE_SECONDS
    )
    assert timing.head_truncated is False
    # the same numbers the item path's own tests pin: 18s is the last moment to start
    assert (timing.render_due, timing.too_late) == (True, False)
    # half a second inside the estimate may still miss the boundary by the tolerance
    late = term_timing_for_remaining(
        remaining=5.5,
        clip_seconds=10.0,
        allow_pre=False,
        render_estimate=RENDER_ESTIMATE_SECONDS,
        render_lead=0.0,
    )
    assert (late.render_due, late.too_late) == (True, False)
    past = term_timing_for_remaining(
        remaining=2.0,
        clip_seconds=10.0,
        allow_pre=False,
        render_estimate=RENDER_ESTIMATE_SECONDS,
        render_lead=0.0,
    )
    assert (past.render_due, past.too_late) == (False, True)


def test_the_next_pass_is_booked_from_the_window_that_was_decided_on() -> None:
    """The booking is the window itself, so the pass it books lands where it is due."""
    timing = term_timing_for_remaining(
        remaining=60.0,
        clip_seconds=30.0,
        allow_pre=True,
        lead_seconds=TALK_OVER_MAX_PRE,
        render_estimate=RENDER_ESTIMATE_SECONDS,
        render_lead=PRODUCER_LEAD_SECONDS,
    )
    # 60s - (12s head + 12.34s lead + 6s render + 1s tolerance + 15s reserve)
    window = (
        TALK_OVER_MAX_PRE
        + PRODUCER_LEAD_SECONDS
        + RENDER_ESTIMATE_SECONDS
        + TERM_SCHEDULE_TOLERANCE_SECONDS
        + TERM_RENDER_MARGIN_SECONDS
    )
    assert timing.render_in_seconds == pytest.approx(60.0 - window)
    assert timing.render_due is False

    due = term_timing_for_remaining(
        remaining=60.0 - (timing.render_in_seconds or 0.0),
        clip_seconds=30.0,
        allow_pre=True,
        lead_seconds=TALK_OVER_MAX_PRE,
        render_estimate=RENDER_ESTIMATE_SECONDS,
        render_lead=PRODUCER_LEAD_SECONDS,
    )
    assert due.render_due is True
    assert due.render_in_seconds == 0.0


def test_a_boundary_that_is_not_known_books_no_pass() -> None:
    """Without a boundary there is no window to book a pass for."""
    timing = term_timing(
        duration=100.0,
        elapsed=60.0,
        fade_seconds=None,
        clip_seconds=10.0,
        allow_pre=False,
        render_estimate=RENDER_ESTIMATE_SECONDS,
    )
    assert timing.render_in_seconds is None


def test_the_render_estimate_starts_at_six_seconds() -> None:
    """The measured estimate has a documented starting point."""
    assert RenderEstimate().seconds == pytest.approx(RENDER_ESTIMATE_SECONDS)


def test_the_render_estimate_is_smoothed() -> None:
    """A single slow render moves the estimate only part of the way."""
    estimate = RenderEstimate()
    estimate.record(12.0)
    assert estimate.seconds == pytest.approx(0.7 * RENDER_ESTIMATE_SECONDS + 0.3 * 12.0)
    assert estimate.samples == 1


def test_the_render_estimate_is_bounded() -> None:
    """An outlier neither makes the estimate unbounded nor unusable."""
    estimate = RenderEstimate()
    estimate.record(600.0)
    assert estimate.seconds == pytest.approx(MAX_RENDER_ESTIMATE_SECONDS)
    for _ in range(50):
        estimate.record(0.0)
    assert estimate.seconds == pytest.approx(MIN_RENDER_ESTIMATE_SECONDS)


def test_the_live_boundary_that_was_dropped_is_rendered_in_its_window() -> None:
    """The boundary lost to a window one render wide is due, and its clip still lands."""
    timing = term_timing_for_remaining(
        remaining=LIVE_REMAINING_SECONDS,
        clip_seconds=30.0,
        allow_pre=True,
        lead_seconds=TERM_HEAD_DEFAULT_SECONDS,
        render_estimate=LIVE_ESTIMATE_SECONDS,
        render_lead=LIVE_LEAD_SECONDS,
    )
    assert timing.head_seconds == pytest.approx(TERM_HEAD_DEFAULT_SECONDS)
    assert timing.render_due is True
    assert timing.too_late is False

    # the window opens a pump period before the pass that used to render it, so the render
    # starts there and is done long before the clip could land behind the produced frames
    window = (
        TERM_HEAD_DEFAULT_SECONDS
        + LIVE_LEAD_SECONDS
        + LIVE_ESTIMATE_SECONDS
        + TERM_SCHEDULE_TOLERANCE_SECONDS
        + TERM_RENDER_MARGIN_SECONDS
    )
    placed = term_timing_for_remaining(
        remaining=window - LIVE_RENDER_SECONDS,
        clip_seconds=30.0,
        allow_pre=True,
        lead_seconds=TERM_HEAD_DEFAULT_SECONDS,
        render_estimate=LIVE_RENDER_SECONDS,
        render_lead=LIVE_LEAD_SECONDS + LIVE_LEAD_GROWTH_SECONDS,
    )
    assert placed.too_late is False
    delay = window - LIVE_RENDER_SECONDS - TERM_HEAD_DEFAULT_SECONDS - TERM_START_BIAS_SECONDS
    assert placed.delay_seconds == pytest.approx(delay)
    # the clip starts ahead of what the producer already emitted, so it lands without late
    assert delay >= LIVE_LEAD_SECONDS + LIVE_LEAD_GROWTH_SECONDS

    # a pass that only came due after the render had started would still lose the term,
    # which is what the margin takes out of the cadence: no pass is booked that late
    inside = term_timing_for_remaining(
        remaining=LIVE_REMAINING_SECONDS - LIVE_RENDER_SECONDS,
        clip_seconds=30.0,
        allow_pre=True,
        lead_seconds=TERM_HEAD_DEFAULT_SECONDS,
        render_estimate=LIVE_RENDER_SECONDS,
        render_lead=LIVE_LEAD_SECONDS + LIVE_LEAD_GROWTH_SECONDS,
    )
    assert inside.too_late is True


def test_the_window_is_a_pump_period_wider_than_the_render_budget() -> None:
    """A pass may land a whole pump period late, so the window carries that as reserve."""
    assert pytest.approx(TERM_PUMP_MAX_SECONDS) == TERM_RENDER_MARGIN_SECONDS
    head = TERM_HEAD_DEFAULT_SECONDS
    lead = 10.0
    estimate = 6.0
    due_at = head + lead + estimate + TERM_SCHEDULE_TOLERANCE_SECONDS + TERM_RENDER_MARGIN_SECONDS
    late_at = head + lead + estimate - TERM_SCHEDULE_TOLERANCE_SECONDS
    assert due_at - late_at == pytest.approx(
        TERM_RENDER_MARGIN_SECONDS + 2 * TERM_SCHEDULE_TOLERANCE_SECONDS
    )
    assert due_at - late_at == pytest.approx(17.0)

    at_edge = term_timing_for_remaining(
        remaining=due_at,
        clip_seconds=30.0,
        allow_pre=True,
        render_estimate=estimate,
        render_lead=lead,
    )
    assert at_edge.render_due is True
    above_edge = term_timing_for_remaining(
        remaining=due_at + 0.01,
        clip_seconds=30.0,
        allow_pre=True,
        render_estimate=estimate,
        render_lead=lead,
    )
    assert above_edge.render_due is False
    # the booking is the same edge, so the pass it books lands exactly where it is due
    assert above_edge.render_in_seconds == pytest.approx(0.01)


def test_the_late_threshold_is_the_last_moment_a_render_can_still_finish() -> None:
    """A tenth above the threshold still renders, a tenth below it can no longer reach."""
    head = TERM_HEAD_DEFAULT_SECONDS
    lead = 10.0
    estimate = 6.0
    threshold = head + lead + estimate - TERM_SCHEDULE_TOLERANCE_SECONDS
    just_in = term_timing_for_remaining(
        remaining=threshold + 0.1,
        clip_seconds=30.0,
        allow_pre=True,
        render_estimate=estimate,
        render_lead=lead,
    )
    just_out = term_timing_for_remaining(
        remaining=threshold - 0.1,
        clip_seconds=30.0,
        allow_pre=True,
        render_estimate=estimate,
        render_lead=lead,
    )
    assert just_in.too_late is False
    assert just_out.too_late is True
    assert just_out.render_due is False
    # the head is part of the threshold: a term starts that much before the boundary
    assert threshold - (lead + estimate - TERM_SCHEDULE_TOLERANCE_SECONDS) == pytest.approx(head)


def test_a_far_boundary_books_a_pass_at_the_window_edge() -> None:
    """The booking is the window, and the pump books no more of it than one period."""
    window = (
        TERM_HEAD_DEFAULT_SECONDS
        + 10.0
        + 6.0
        + TERM_SCHEDULE_TOLERANCE_SECONDS
        + TERM_RENDER_MARGIN_SECONDS
    )
    far = term_timing_for_remaining(
        remaining=300.0,
        clip_seconds=30.0,
        allow_pre=True,
        render_estimate=6.0,
        render_lead=10.0,
    )
    assert far.render_in_seconds == pytest.approx(300.0 - window)
    assert min(far.render_in_seconds or 0.0, TERM_PUMP_MAX_SECONDS) == pytest.approx(
        TERM_PUMP_MAX_SECONDS
    )

    near = term_timing_for_remaining(
        remaining=40.0,
        clip_seconds=30.0,
        allow_pre=True,
        render_estimate=6.0,
        render_lead=10.0,
    )
    assert near.render_in_seconds == pytest.approx(40.0 - window)
    assert near.render_in_seconds == pytest.approx(6.0)


def test_a_long_head_keeps_the_window_at_least_as_wide_as_it_was() -> None:
    """A 12s head widens both edges alike, so the old window still sits inside the new one."""
    old_window = (
        TALK_OVER_MAX_PRE
        + PRODUCER_LEAD_SECONDS
        + RENDER_ESTIMATE_SECONDS
        + TERM_SCHEDULE_TOLERANCE_SECONDS
    )
    timing = term_timing_for_remaining(
        remaining=old_window,
        clip_seconds=30.0,
        allow_pre=True,
        lead_seconds=TALK_OVER_MAX_PRE,
        render_estimate=RENDER_ESTIMATE_SECONDS,
        render_lead=PRODUCER_LEAD_SECONDS,
    )
    assert timing.head_seconds == pytest.approx(TALK_OVER_MAX_PRE)
    assert timing.render_due is True
    assert timing.too_late is False
    new_window = (
        TALK_OVER_MAX_PRE
        + PRODUCER_LEAD_SECONDS
        + RENDER_ESTIMATE_SECONDS
        + TERM_SCHEDULE_TOLERANCE_SECONDS
        + TERM_RENDER_MARGIN_SECONDS
    )
    assert new_window > old_window
    assert new_window - old_window == pytest.approx(TERM_RENDER_MARGIN_SECONDS)


def test_a_term_that_may_not_post_ends_before_the_boundary() -> None:
    """A break that may not run over the next record keeps its whole clip in front of it."""
    timing = term_timing_for_remaining(
        remaining=30.0,
        clip_seconds=8.0,
        allow_pre=False,
        lead_seconds=TERM_HEAD_DEFAULT_SECONDS,
        allow_post=False,
        render_estimate=RENDER_ESTIMATE_SECONDS,
    )
    # the whole clip is the head: it starts its own length before the boundary, not the lead
    assert timing.head_seconds == pytest.approx(8.0)
    # no start bias is taken off: a break that may not post ends on the boundary, not later
    assert timing.delay_seconds == pytest.approx(30.0 - 8.0)
    # the clip ends on the boundary, so nothing of it plays over the next record
    assert (timing.delay_seconds or 0.0) + 8.0 == pytest.approx(30.0)
    assert timing.too_late is False


def test_a_term_that_may_not_post_whose_clip_exactly_fits_ends_on_the_boundary() -> None:
    """A clip of exactly the remaining time starts now and ends on the boundary."""
    timing = term_timing_for_remaining(
        remaining=8.0,
        clip_seconds=8.0,
        allow_pre=False,
        allow_post=False,
        render_estimate=0.0,
    )
    # the start bias is no part of a break that has to be done at the boundary: the delay is
    # zero, not a negative one the api would clamp, which would push the whole clip that much
    # past the boundary
    assert timing.delay_seconds == 0.0
    assert timing.head_seconds == pytest.approx(8.0)
    assert (timing.delay_seconds or 0.0) + 8.0 == pytest.approx(8.0)
    assert timing.too_late is False


def test_a_term_that_may_post_keeps_the_head_its_host_set() -> None:
    """Allowing the post leaves the placement as it was: the head is the host's lead."""
    timing = term_timing_for_remaining(
        remaining=30.0,
        clip_seconds=8.0,
        allow_pre=True,
        lead_seconds=TERM_HEAD_DEFAULT_SECONDS,
        allow_post=True,
        render_estimate=RENDER_ESTIMATE_SECONDS,
    )
    assert timing.head_seconds == pytest.approx(TERM_HEAD_DEFAULT_SECONDS)
    assert timing.delay_seconds == pytest.approx(
        30.0 - TERM_HEAD_DEFAULT_SECONDS - TERM_START_BIAS_SECONDS
    )


def test_a_term_that_can_not_end_before_its_boundary_is_too_late() -> None:
    """A boundary too close for the clip and its render is given up, never run over."""
    timing = term_timing_for_remaining(
        remaining=10.0,
        clip_seconds=8.0,
        allow_pre=False,
        allow_post=False,
        render_estimate=RENDER_ESTIMATE_SECONDS,
    )
    # ten seconds can not hold the eight second clip plus the render before the boundary
    assert timing.too_late is True
    assert timing.render_due is False


def test_a_clip_longer_than_the_boundary_is_capped_there_and_too_late() -> None:
    """A clip that no longer fits starts as early as the boundary allows and is given up."""
    timing = term_timing_for_remaining(
        remaining=5.0,
        clip_seconds=8.0,
        allow_pre=False,
        allow_post=False,
        render_estimate=RENDER_ESTIMATE_SECONDS,
    )
    # the head never reaches behind the listener: the boundary is all the room there is
    assert timing.head_seconds == pytest.approx(5.0)
    assert timing.too_late is True


def test_the_deadline_of_a_term_that_may_not_post_is_its_boundary() -> None:
    """A break that ends before the boundary is done when the boundary arrives."""
    timing = term_timing_for_remaining(
        remaining=30.0,
        clip_seconds=8.0,
        allow_pre=False,
        allow_post=False,
        render_estimate=RENDER_ESTIMATE_SECONDS,
    )
    assert timing.deadline_seconds == pytest.approx(
        30.0 - 8.0 + 8.0 + TERM_SCHEDULE_TOLERANCE_SECONDS
    )

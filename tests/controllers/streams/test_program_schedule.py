"""
Scheduling a program through the streams API: the result, its ledger and its events.

The bus itself is exercised in ``test_program_overlay``; these tests pin what the
controller reports back to the caller that planned the program: where it was anchored,
how late it is, why it was not placed, and which state the per-queue ledger reached.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import ContentType
from music_assistant_models.media_items import AudioFormat

from music_assistant.controllers.streams.constants import (
    LEAD_MIN_LEAD_SECONDS,
    VOICE_OVER_RAMP,
)
from music_assistant.controllers.streams.controller import ScheduleReason
from music_assistant.controllers.streams.program_mixer import BusPlacement
from music_assistant.controllers.streams.program_schedule import (
    ProgramKind,
    ProgramPriority,
    ProgramRequest,
    place_program,
    priority_for,
)
from tests.controllers.streams.program_bus_helpers import (
    AUDIBLE_BEFORE,
    FRAME_BYTES,
    LEAD_FRAMES,
    PRODUCED_BEFORE,
    QUEUE_ID,
    SECOND,
)
from tests.controllers.streams.program_bus_helpers import (
    bare_streams_controller as _controller,
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

# a stereo 24-bit format the bus can not mix
PASSTHROUGH_FORMAT = AudioFormat(
    content_type=ContentType.PCM_S24LE,
    sample_rate=48000,
    bit_depth=24,
    channels=2,
)


# --- the reported anchor and its lateness ---


def test_no_live_flow_stream_reports_no_bus() -> None:
    """A queue that is not streaming says why nothing can be mixed in."""
    controller = _controller()

    result = _schedule(controller)

    assert result.scheduled is False
    assert result.reason == "no_bus"
    assert result.anchor_frame == 0
    assert result.late_by_seconds == 0.0


async def test_a_program_without_a_delay_is_anchored_a_lead_after_the_listener() -> None:
    """The unchanged announcement placement is reported with its anchor and its lead."""
    controller = _controller(audible=AUDIBLE_BEFORE)
    program_bus = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))

    result = _schedule(
        controller, priority=ProgramPriority.ANNOUNCEMENT, kind=ProgramKind.ANNOUNCEMENT
    )

    assert result.scheduled is True
    assert result.anchor_frame == int(AUDIBLE_BEFORE * SECOND) + LEAD_FRAMES
    # the anchor is raised to the produced frames, but an announcement has no boundary it
    # could be late for
    assert result.late_by_seconds == 0.0
    # the minimum lead is reported separately, never as lateness
    assert result.lead_shortfall_seconds == pytest.approx(LEAD_MIN_LEAD_SECONDS)
    # the anchor is the lead after the listener, whatever the producer already emitted
    assert result.anchor_frame - int(AUDIBLE_BEFORE * SECOND) == LEAD_FRAMES
    assert result.frames == SECOND
    assert result.reason is None


async def test_a_delay_shorter_than_the_lead_reports_its_shortfall_separately() -> None:
    """A boundary closer than the lead is still placed at the lead, which is no lateness."""
    controller = _controller(audible=AUDIBLE_BEFORE)
    program_bus = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))

    result = _schedule(controller, start_in_seconds=0.5)

    assert result.scheduled is True
    assert result.anchor_frame == int(AUDIBLE_BEFORE * SECOND) + LEAD_FRAMES
    # the producer is already ahead of the boundary, that part is lateness
    assert result.late_by_seconds == pytest.approx(PRODUCED_BEFORE - (AUDIBLE_BEFORE + 0.5))
    # the rest of the shortfall comes from the lead the bus needs
    assert result.lead_shortfall_seconds == pytest.approx(LEAD_MIN_LEAD_SECONDS - 0.5)


async def test_a_delay_past_the_lead_anchors_where_the_listener_will_be() -> None:
    """A boundary beyond the lead is hit exactly, so nothing is late."""
    controller = _controller(audible=AUDIBLE_BEFORE)
    program_bus = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))

    result = _schedule(controller, start_in_seconds=12.0)

    assert result.scheduled is True
    assert result.anchor_frame == int((AUDIBLE_BEFORE + 12.0) * SECOND)
    assert result.late_by_seconds == 0.0
    assert result.lead_shortfall_seconds == 0.0


async def test_the_result_reports_the_anchor_before_it_was_moved_to_the_produced_end() -> None:
    """An anchor the session already produced is moved, and its lateness is reported."""
    controller = _controller(audible=0.5)
    program_bus = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))

    result = _schedule(controller, kind=ProgramKind.ANNOUNCEMENT)

    assert result.scheduled is True
    # the program can not be mixed into frames that went out already
    assert result.anchor_frame == int(PRODUCED_BEFORE * SECOND)
    # an announcement has no boundary, so the move is no lateness
    assert result.late_by_seconds == 0.0
    assert result.lead_shortfall_seconds == pytest.approx(LEAD_MIN_LEAD_SECONDS)


async def test_a_delayed_program_without_a_position_is_not_placed() -> None:
    """Without a heard position a boundary program has nothing to be timed against."""
    controller = _controller()
    program_bus = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))

    result = _schedule(controller, start_in_seconds=5.0)

    assert result.scheduled is False
    assert result.reason == "no_audible_position"
    assert program_bus.active is False


async def test_an_undelayed_program_without_a_position_keeps_the_lead_placement() -> None:
    """An announcement needs no position: it is placed a lead after the produced audio."""
    controller = _controller()
    program_bus = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))

    result = _schedule(
        controller, priority=ProgramPriority.ANNOUNCEMENT, kind=ProgramKind.ANNOUNCEMENT
    )

    assert result.scheduled is True
    assert result.anchor_frame == int(PRODUCED_BEFORE * SECOND) + LEAD_FRAMES
    # without a position there is no boundary to be late for either
    assert result.late_by_seconds == 0.0
    # the anchor is a whole lead after the produced frames, so that is the shortfall
    assert result.lead_shortfall_seconds == pytest.approx(LEAD_MIN_LEAD_SECONDS)
    assert program_bus.active is True


async def test_a_zero_or_negative_delay_without_a_position_keeps_the_lead_placement() -> None:
    """Nothing ahead of the listener also uses the unchanged placement without a position."""
    for start_in_seconds in (0.0, -0.001, -30.0):
        controller = _controller()
        program_bus = _register(controller)
        await program_bus.feed(_music(PRODUCED_BEFORE))

        result = _schedule(controller, start_in_seconds=start_in_seconds)

        assert result.scheduled is True
        assert result.anchor_frame == int(PRODUCED_BEFORE * SECOND) + LEAD_FRAMES
        assert result.late_by_seconds == 0.0
        assert result.lead_shortfall_seconds == pytest.approx(LEAD_MIN_LEAD_SECONDS)


@pytest.mark.parametrize("start_in_seconds", [0.001, 5.0, 3600.0])
async def test_a_delay_without_a_position_is_refused_at_every_size(start_in_seconds: float) -> None:
    """A boundary program is not silently placed from the producer edge, however small."""
    controller = _controller()
    program_bus = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))

    result = _schedule(controller, start_in_seconds=start_in_seconds)

    assert result.scheduled is False
    assert result.reason == "no_audible_position"
    assert program_bus.active is False


@pytest.mark.parametrize("start_in_seconds", [0.0, 0.001, 5.0, 3600.0])
async def test_a_positioned_delay_is_always_defined(start_in_seconds: float) -> None:
    """Every delay against a heard position lands on the lead or on the boundary, never off it."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))

    result = _schedule(controller, start_in_seconds=start_in_seconds)

    assert result.scheduled is True
    expected = max(LEAD_FRAMES, round(start_in_seconds * SECOND))
    assert result.anchor_frame == expected


# --- the format the bus can mix ---


def test_a_format_the_bus_cannot_mix_is_reported_as_format() -> None:
    """Passthrough audio is refused instead of silently mixed."""
    controller = _controller(audible=0.0)
    _register(controller, pcm_format=PASSTHROUGH_FORMAT)

    result = controller.schedule_overlay(QUEUE_ID, b"\x00\x00\x00" * 4800, label="term")

    assert result.scheduled is False
    assert result.reason == "format"


async def test_empty_audio_is_refused_as_format() -> None:
    """A program without audio would only duck the music, so it is not placed."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))

    result = controller.schedule_overlay(QUEUE_ID, b"", label="term")

    assert result.scheduled is False
    assert result.reason == "format"
    assert result.frames == 0
    assert program_bus.active is False


async def test_audio_shorter_than_one_frame_is_refused() -> None:
    """Audio that does not fill a single frame has nothing the bus could mix."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))

    result = controller.schedule_overlay(QUEUE_ID, b"\x08" * (FRAME_BYTES - 1), label="term")

    assert result.scheduled is False
    assert result.reason == "format"
    assert result.frames == 0
    assert program_bus.active is False


async def test_audio_not_aligned_to_a_frame_is_cut_to_whole_frames() -> None:
    """A trailing partial frame is not mixed, and the reported length is the whole frames."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    pcm = _voice(1.0) + b"\x08" * (FRAME_BYTES - 1)

    result = controller.schedule_overlay(QUEUE_ID, pcm, label="term", start_in_seconds=5.0)

    assert result.scheduled is True
    assert result.frames == SECOND
    assert program_bus.scheduled_voice_frames("term") == SECOND


# --- windows and priorities ---


async def test_a_window_taken_by_a_higher_priority_program_drops_the_new_one() -> None:
    """An announcement keeps its window: the term planned onto it is not placed."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    announcement = _schedule(
        controller,
        label="announcement",
        kind=ProgramKind.ANNOUNCEMENT,
    )

    term = _schedule(controller, label="term", start_in_seconds=2.0)

    assert announcement.scheduled is True
    assert term.scheduled is False
    assert term.reason == "window_taken"
    assert program_bus.scheduled_voice_frames("announcement") is not None
    assert program_bus.scheduled_voice_frames("term") is None


async def test_a_higher_priority_program_takes_the_window_of_a_lower_one() -> None:
    """The announcement is placed where it was planned; the term in that window is dropped."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    term = _schedule(controller, label="term", start_in_seconds=2.0)

    announcement = _schedule(
        controller,
        label="announcement",
        kind=ProgramKind.ANNOUNCEMENT,
    )

    assert term.scheduled is True
    assert announcement.scheduled is True
    # the announcement kept the anchor it planned for, it was not moved aside
    assert announcement.anchor_frame == 2 * SECOND
    assert program_bus.scheduled_voice_frames("announcement") is not None
    assert program_bus.scheduled_voice_frames("term") is None


async def test_a_term_takes_the_window_of_a_crossplay_program() -> None:
    """The priority order is announcement, then term, then crossplay."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    crossplay = _schedule(
        controller, label="crossplay", start_in_seconds=2.0, kind=ProgramKind.CROSSPLAY
    )

    term = _schedule(controller, label="term", start_in_seconds=2.0)

    assert crossplay.scheduled is True
    assert term.scheduled is True
    assert program_bus.scheduled_voice_frames("term") is not None
    assert program_bus.scheduled_voice_frames("crossplay") is None


async def test_two_programs_on_the_same_anchor_keep_only_the_higher_priority_one() -> None:
    """Exactly one program occupies an anchor: the lower ranked ones are dropped."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(
        controller,
        label="announcement",
        kind=ProgramKind.ANNOUNCEMENT,
    )

    term = _schedule(controller, label="term", start_in_seconds=2.0)
    crossplay = _schedule(
        controller, label="crossplay", start_in_seconds=2.0, kind=ProgramKind.CROSSPLAY
    )

    assert term.scheduled is False
    assert term.reason == "window_taken"
    assert crossplay.scheduled is False
    assert crossplay.reason == "window_taken"
    assert program_bus.scheduled_voice_frames("announcement") is not None


async def test_equal_priority_programs_are_mixed_one_after_the_other() -> None:
    """Two programs of the same priority both play; the second is moved and says why."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    first = _schedule(controller, label="first", start_in_seconds=5.0, seconds=2.0)

    second = _schedule(controller, label="second", start_in_seconds=5.0)

    assert first.scheduled is True
    assert second.scheduled is True
    # the bus keeps two programs of the same priority apart, past the first one's duck tail
    assert second.anchor_frame == 7 * SECOND + int(VOICE_OVER_RAMP * SECOND)
    assert second.late_by_seconds == 0.0
    assert second.reason == "window_taken"


def test_a_priority_that_does_not_match_the_kind_is_rejected() -> None:
    """The rank follows the kind, so a contradicting priority is an error, not a policy."""
    controller = _controller(audible=0.0)
    _register(controller)

    with pytest.raises(ValueError, match="priority"):
        _schedule(
            controller,
            label="mixed-up",
            kind=ProgramKind.ANNOUNCEMENT,
            priority=ProgramPriority.TERM,
        )
    with pytest.raises(ValueError, match="priority"):
        _schedule(controller, label="unknown", priority=cast("Any", 99))


@pytest.mark.parametrize(
    "kind", [ProgramKind.ANNOUNCEMENT, ProgramKind.INTRO], ids=["announcement", "intro"]
)
async def test_a_program_without_a_boundary_is_never_late(kind: ProgramKind) -> None:
    """
    A kind without a boundary reports no lateness, however deep the buffer runs.

    The anchor is still moved to the produced frames, but there is no boundary it could be
    late for: the ledger records 0.0 for the same placement, so the answer says the same.
    """
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(10.0))

    result = _schedule(controller, kind=kind, start_in_seconds=0.0)

    assert result.scheduled is True
    assert result.anchor_frame == 10 * SECOND
    assert result.late_by_seconds == 0.0


def test_an_intro_ranks_like_an_announcement() -> None:
    """The intro is heard over a term as well, so it shares the announcement's rank."""
    assert priority_for(ProgramKind.INTRO) == ProgramPriority.ANNOUNCEMENT


# --- cancelling ---


async def test_cancel_overlay_drops_exactly_one_label() -> None:
    """A single program is withdrawn by its label, the others stay scheduled."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="first", start_in_seconds=5.0)
    _schedule(controller, label="second", start_in_seconds=9.0)

    cancelled = controller.cancel_overlay(QUEUE_ID, "first", "skipped")

    assert cancelled is True
    assert program_bus.scheduled_voice_frames("first") is None
    assert program_bus.scheduled_voice_frames("second") is not None
    assert controller.cancel_overlay(QUEUE_ID, "first", "skipped") is False


async def test_cancel_overlay_of_an_unknown_label_is_false() -> None:
    """An unknown label is a no-op, not an error."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="term", start_in_seconds=5.0)

    assert controller.cancel_overlay(QUEUE_ID, "not-there", "skipped") is False
    assert program_bus.scheduled_voice_frames("term") is not None


async def test_cancel_overlay_drops_every_program_with_the_label() -> None:
    """A label stands for every placement under it, so cancelling drops all of them."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="same", start_in_seconds=5.0)
    _schedule(controller, label="same", start_in_seconds=9.0)
    _schedule(controller, label="other", start_in_seconds=13.0)

    assert controller.cancel_overlay(QUEUE_ID, "same", "skipped") is True

    assert program_bus.scheduled_voice_frames("same") is None
    assert program_bus.scheduled_voice_frames("other") is not None
    states = controller.overlay_state(QUEUE_ID)
    assert [(state.label, state.state) for state in states] == [
        ("same", "dropped"),
        ("same", "dropped"),
        ("other", "scheduled"),
    ]


def test_a_result_is_false_when_nothing_was_placed() -> None:
    """A caller that only tests the answer keeps the outcome it tested before."""
    controller = _controller()

    result = _schedule(controller)

    assert bool(result) is False
    assert not result


async def test_a_result_is_true_when_the_program_was_placed() -> None:
    """A placed program answers truthy, like the bare boolean it replaced."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))

    result = _schedule(controller)

    assert bool(result) is True


async def test_a_boundary_beyond_the_lead_keeps_its_anchor_and_is_not_late() -> None:
    """The unchanged path: a boundary past the lead is hit exactly, with no lateness."""
    controller = _controller(audible=AUDIBLE_BEFORE)
    program_bus = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))

    result = _schedule(controller, start_in_seconds=5.0)

    assert result.scheduled is True
    assert result.anchor_frame == int((AUDIBLE_BEFORE + 5.0) * SECOND)
    assert result.late_by_seconds == 0.0


async def test_the_shared_controller_fixture_mixes_a_program(make_streams_controller: Any) -> None:
    """The shared fixture builds a controller that can place and report a program."""
    controller = make_streams_controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))

    result = _schedule(controller, start_in_seconds=5.0)

    assert result.scheduled is True
    assert [state.label for state in controller.overlay_state(QUEUE_ID)] == ["term"]


# --- lateness and lead shortfall ---------------------------------------------------------


async def test_a_boundary_program_past_the_tolerance_is_dropped_as_too_late() -> None:
    """A term that missed its boundary by more than the tolerance is not played late."""
    controller = _controller(audible=0.5)
    program_bus = _register(controller)
    await program_bus.feed(_music(10.0))

    result = _schedule(controller, start_in_seconds=7.0)

    assert result.scheduled is False
    assert result.reason == "too_late"
    # the boundary was 7.5 s from the listener, the producer is at 10 s
    assert result.late_by_seconds == pytest.approx(2.5)
    assert program_bus.active is False
    assert [(state.state, state.reason) for state in controller.overlay_state(QUEUE_ID)] == [
        ("dropped", "too_late")
    ]


async def test_an_announcement_is_not_late_in_a_deep_buffer_and_is_not_dropped() -> None:
    """An announcement has no boundary to miss: a deep buffer does not discard or delay it."""
    controller = _controller(audible=0.5)
    program_bus = _register(controller)
    await program_bus.feed(_music(10.0))

    result = _schedule(controller, kind=ProgramKind.ANNOUNCEMENT)

    assert result.scheduled is True
    assert result.anchor_frame == 10 * SECOND
    # the anchor is the producer's edge, but the announcement has no boundary to be late for
    assert result.late_by_seconds == 0.0
    assert result.reason is None


async def test_the_minimum_lead_is_not_reported_as_lateness() -> None:
    """An undelayed program placed exactly at the listener reports the lead as a shortfall."""
    controller = _controller(audible=PRODUCED_BEFORE)
    program_bus = _register(controller)
    await program_bus.feed(_music(PRODUCED_BEFORE))

    result = _schedule(controller, start_in_seconds=0.0)

    assert result.scheduled is True
    assert result.anchor_frame == int(PRODUCED_BEFORE * SECOND) + LEAD_FRAMES
    assert result.late_by_seconds == 0.0
    assert result.lead_shortfall_seconds == pytest.approx(LEAD_MIN_LEAD_SECONDS)


def test_the_reason_vocabulary_covers_the_designed_drops() -> None:
    """The reasons the provider has to read are part of the public answer."""
    assert {
        "no_bus",
        "no_audible_position",
        "window_taken",
        "too_late",
        "format",
        "bus_replaced",
        "flow_ended",
        "head_truncated",
        "unknown_boundary",
    } <= {reason.value for reason in ScheduleReason}


async def test_a_lead_head_keeps_its_window_against_a_term() -> None:
    """A lead head ranks above a term, so the term gives way instead of displacing it."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    # this is how the audio controller arms a lead's head: straight on the bus
    assert (
        program_bus.schedule(
            _voice(2.0),
            int(5 * SECOND),
            label="lead:clip",
            priority=ProgramPriority.LEAD,
        )
        is not None
    )

    term = _schedule(controller, label="term", start_in_seconds=5.0)

    assert term.scheduled is False
    assert term.reason == "window_taken"
    assert program_bus.scheduled_voice_frames("lead:clip") is not None


async def test_an_announcement_takes_a_lead_heads_window() -> None:
    """An announcement outranks a lead head, so it plays where it was planned."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    assert (
        program_bus.schedule(
            _voice(2.0),
            int(2 * SECOND),
            label="lead:clip",
            priority=ProgramPriority.LEAD,
        )
        is not None
    )

    announcement = _schedule(
        controller,
        label="announcement:one",
        kind=ProgramKind.ANNOUNCEMENT,
    )

    assert announcement.scheduled is True
    assert announcement.anchor_frame == 2 * SECOND
    assert program_bus.scheduled_voice_frames("lead:clip") is None


def test_the_program_rank_order_is_announcement_over_lead_over_term_over_crossplay() -> None:
    """The design's order: an announcement outranks a lead head, which outranks a term."""
    assert ProgramPriority.ANNOUNCEMENT > ProgramPriority.LEAD
    assert ProgramPriority.LEAD > ProgramPriority.TERM
    assert ProgramPriority.TERM > ProgramPriority.CROSSPLAY


async def test_a_term_is_too_late_even_without_a_delay() -> None:
    """A term whose boundary is already behind the buffer is dropped, delay or not."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(10.0))

    result = _schedule(controller, start_in_seconds=0.0)

    assert result.scheduled is False
    assert result.reason == "too_late"
    assert result.late_by_seconds == pytest.approx(10.0)
    assert program_bus.active is False


async def test_an_announcement_without_a_delay_is_placed_even_in_a_deep_buffer() -> None:
    """An announcement has no boundary, so the buffer depth is not lateness for it."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(10.0))

    result = _schedule(controller, kind=ProgramKind.ANNOUNCEMENT, start_in_seconds=0.0)

    assert result.scheduled is True
    assert result.anchor_frame == 10 * SECOND


async def test_a_ledger_drop_stops_the_program_on_the_bus() -> None:
    """Every program the ledger drops is cancelled on the bus, not only on a queue command."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    _schedule(controller, label="term", start_in_seconds=5.0)
    assert program_bus.scheduled_voice_frames("term") is not None

    controller._program_ledger.mark_dropped(QUEUE_ID, "term", "test")

    assert program_bus.scheduled_voice_frames("term") is None


def test_an_announcement_with_a_boundary_delay_is_rejected() -> None:
    """An announcement is immediate; a delay would give it a boundary it can not have."""
    controller = _controller(audible=0.0)
    _register(controller)

    with pytest.raises(ValueError, match="delay"):
        _schedule(controller, kind=ProgramKind.ANNOUNCEMENT, start_in_seconds=5.0)


def test_an_intro_with_a_boundary_delay_is_rejected() -> None:
    """An intro is immediate too, so a delay is a caller mistake, not an exemption."""
    controller = _controller(audible=0.0)
    _register(controller)

    with pytest.raises(ValueError, match="delay"):
        _schedule(controller, kind=ProgramKind.INTRO, start_in_seconds=5.0)


async def test_a_bus_that_refuses_a_window_reports_window_taken() -> None:
    """A refusal from the bus keeps the reason the bus gave instead of blaming lateness."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    await program_bus.feed(_music(1.0))
    program_bus.place = MagicMock(return_value=None)
    program_bus.last_refusal = ScheduleReason.WINDOW_TAKEN

    result = _schedule(controller, label="term", start_in_seconds=2.0)

    assert result.scheduled is False
    assert result.reason == "window_taken"


# --- the length the bus really plays ---


def test_the_result_and_the_ledger_keep_the_length_the_bus_really_plays() -> None:
    """A bus that kept less than was offered answers and records the shorter length."""
    controller = _controller(audible=0.0)
    program_bus = _register(controller)
    # the bus drops the front of the program: two and a half of the three seconds are left
    kept = int(2.5 * SECOND)
    program_bus.place = MagicMock(return_value=BusPlacement(start_frame=5 * SECOND, frames=kept))

    result = place_program(
        program_bus,
        _voice(3.0),
        request=ProgramRequest(
            queue_id=QUEUE_ID,
            label="term",
            start_in_seconds=0.0,
            priority=ProgramPriority.TERM,
            kind=ProgramKind.TERM,
        ),
        audible_frame=0,
        ledger=controller._program_ledger,
        logger=controller.logger,
        report_no_position=lambda *_args: None,
    )

    assert result.scheduled is True
    assert result.frames == kept
    (entry,) = controller._program_ledger.entries(QUEUE_ID)
    assert entry.frames == result.frames == kept

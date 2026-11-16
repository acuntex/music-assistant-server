"""Wiring tests for the AI DJ's program terms on a queue's program bus."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from music_assistant_models.enums import CrossfadeMode, EventType, PlaybackState

import music_assistant.providers.ai_radio.runtime
from music_assistant.controllers.streams.program_schedule import ProgramKind
from music_assistant.providers.ai_radio.constants import (
    AI_COOLDOWN_SECONDS,
    ATTR_QUEUE_DJ,
    MAX_CONTRIBUTION_DEFAULT_SECONDS,
    MAX_CONTRIBUTION_MIN_SECONDS,
    OVERLAY_LABEL_TRANSITION,
    RENDER_FAILURES_BEFORE_COOLDOWN,
    TTS_COOLDOWN_SECONDS,
)
from music_assistant.providers.ai_radio.models import RenderPart
from music_assistant.providers.ai_radio.term_engine import (
    BUS_GRACE_SECONDS,
)
from music_assistant.providers.ai_radio.term_timing import TERM_START_BIAS_SECONDS
from music_assistant.providers.ai_radio.terms import (
    TERMS_PER_QUEUE,
    BoundaryState,
    ProgramTerm,
    TermDropReason,
    TermState,
)

from .test_queue_dj import (
    PLAYBACK_PCM_FORMAT,
    POSITION_SECONDS,
    TRACK_SECONDS,
    FakeQueueItem,
    ReplanQueueDJ,
    _announced_gaps,
    _armed_intro_dj,
    _dj_harness,
    _drain_tasks,
    _due_term,
    _fake_queue_with_sources,
    _intro_host,
    _let_amy_lead,
    _make_replan_dj,
    _must_host,
    _overlay_dj,
    _replan_with_the_guard_moved,
    _track,
)


def _terms(dummy: ReplanQueueDJ) -> list[ProgramTerm]:
    """Return the terms the queue's DJ holds."""
    return list(dummy._term_store("queue-1").terms())


def _outro_host() -> dict[str, Any]:
    """Return a host that speaks between songs and once more at the end of the show."""
    host = _must_host()
    host["section_order"] = [
        {"when": "between_songs", "flow": [{"MUST": "Song_Transition"}]},
        {"when": "end_of_playlist", "flow": [{"MUST": "Song_Transition"}]},
    ]
    return host


async def test_the_switch_off_still_splices_a_clip(tmp_path: Path) -> None:
    """With the experimental switch off the queue gets a clip item, exactly as on dev."""
    dummy, _tracks = _overlay_dj(tmp_path)
    dummy.mass.streams.program_bus_enabled_flag = False

    await dummy._replan_queue("queue-1")

    assert len(dummy.player_queues.loads) == 2
    assert _terms(dummy) == []
    assert dummy.mass.overlays == []


async def test_the_switch_on_plans_no_queue_item(tmp_path: Path) -> None:
    """With the switch on a planned boundary becomes a term, and the queue stays untouched."""
    dummy, _tracks = _overlay_dj(tmp_path)

    await dummy._replan_queue("queue-1")

    assert dummy.player_queues.loads == []
    assert dummy.player_queues.update_calls == 0
    # one term per boundary, the one the running item ends at included
    assert len(_terms(dummy)) == 3
    assert len(dummy.player_queues.items("queue-1")) == 4


async def test_the_switch_on_places_a_due_term_without_touching_the_queue(tmp_path: Path) -> None:
    """A term due at the end of the running item is mixed in there, not spliced in."""
    dummy, tracks = _overlay_dj(tmp_path)
    dummy._sections["Song_Transition"]["allow_post"] = True
    await dummy._replan_queue("queue-1")

    queue = dummy.player_queues._queue
    queue.current_item = tracks[1]
    queue.corrected_elapsed_time = TRACK_SECONDS - 8
    await dummy._pump_terms("queue-1")

    assert len(dummy.mass.overlays) == 1
    queue_id, pcm, label, delay = dummy.mass.overlays[0]
    assert queue_id == "queue-1"
    assert label == OVERLAY_LABEL_TRANSITION
    assert dummy.mass.overlay_kinds == [ProgramKind.TERM]
    assert pcm
    assert delay == pytest.approx(8.0 - TERM_START_BIAS_SECONDS)
    # the audio path can only show a head the bus was told about
    assert dummy.mass.overlay_heads == [
        pytest.approx(next(term for term in _terms(dummy)).head_seconds)
    ]
    assert dummy.player_queues.update_calls == 0
    assert next(term for term in _terms(dummy) if term.queue_item_id == tracks[1].queue_item_id)


async def test_a_hosts_lead_places_the_term_that_much_before_the_boundary(tmp_path: Path) -> None:
    """A host that asks for a two second lead gets its break two seconds before the boundary."""
    host = _must_host()
    host["term_head_seconds"] = 2.0
    dummy, tracks = _overlay_dj(tmp_path, host=host)
    dummy._sections["Song_Transition"]["allow_pre"] = True
    dummy._sections["Song_Transition"]["allow_post"] = True
    await dummy._replan_queue("queue-1")

    queue = dummy.player_queues._queue
    queue.current_item = tracks[1]
    queue.corrected_elapsed_time = TRACK_SECONDS - 12
    await dummy._pump_terms("queue-1")

    assert len(dummy.mass.overlays) == 1
    assert dummy.mass.overlays[0][3] == pytest.approx(12.0 - 2.0 - TERM_START_BIAS_SECONDS)
    assert dummy.mass.overlay_heads == [pytest.approx(2.0)]


async def test_a_hosts_lead_of_zero_starts_the_break_with_the_crossfade(tmp_path: Path) -> None:
    """No lead means the break starts exactly on the boundary, over the crossfade."""
    host = _must_host()
    host["term_head_seconds"] = 0.0
    dummy, tracks = _overlay_dj(tmp_path, host=host)
    dummy._sections["Song_Transition"]["allow_pre"] = True
    dummy._sections["Song_Transition"]["allow_post"] = True
    await dummy._replan_queue("queue-1")

    queue = dummy.player_queues._queue
    queue.current_item = tracks[1]
    queue.corrected_elapsed_time = TRACK_SECONDS - 8
    await dummy._pump_terms("queue-1")

    assert len(dummy.mass.overlays) == 1
    assert dummy.mass.overlays[0][3] == pytest.approx(8.0 - TERM_START_BIAS_SECONDS)
    assert dummy.mass.overlay_heads == [pytest.approx(0.0)]


async def test_a_host_lead_does_not_move_a_break_that_may_not_start_over_the_tail(
    tmp_path: Path,
) -> None:
    """A section without a pre keeps its break on the boundary, whatever lead the host set."""
    host = _must_host()
    host["term_head_seconds"] = 5.0
    dummy, tracks = _overlay_dj(tmp_path, host=host)
    dummy._sections["Song_Transition"]["allow_post"] = True
    await dummy._replan_queue("queue-1")

    queue = dummy.player_queues._queue
    queue.current_item = tracks[1]
    queue.corrected_elapsed_time = TRACK_SECONDS - 8
    await dummy._pump_terms("queue-1")

    assert dummy.mass.overlays[0][3] == pytest.approx(8.0 - TERM_START_BIAS_SECONDS)
    assert dummy.mass.overlay_heads == [pytest.approx(0.0)]


async def test_the_switch_on_without_a_bus_drops_the_term(tmp_path: Path) -> None:
    """A queue whose stream carries no program bus can not be spoken over."""
    tracks = [_track(index) for index in range(4)]
    for track in tracks:
        track.duration = TRACK_SECONDS
    dummy = _make_replan_dj(tmp_path, tracks)
    dummy.mass.streams.program_bus_enabled_flag = True
    dummy.player_queues._queue.current_item = tracks[0]
    dummy.player_queues._queue.corrected_elapsed_time = POSITION_SECONDS

    await dummy._replan_queue("queue-1")
    await _drain_tasks(dummy.mass)

    # the bus may still arrive with the play, so the terms wait out their grace
    assert all(term.state is TermState.DECIDED for term in _terms(dummy))
    for term in _terms(dummy):
        term.bus_missing_since = asyncio.get_running_loop().time() - BUS_GRACE_SECONDS - 1
    await dummy._pump_terms("queue-1")

    assert dummy.player_queues.loads == []
    assert dummy.mass.overlays == []
    assert dummy.rendered == []
    assert {term.dropped_reason for term in _terms(dummy)} == {"no_bus"}


async def test_a_show_outro_is_a_term_without_a_successor(tmp_path: Path) -> None:
    """The end of the queue is spoken as a term that announces nothing after it."""
    tracks = [_track(index) for index in range(4)]
    for track in tracks:
        track.duration = TRACK_SECONDS
    dummy = _make_replan_dj(tmp_path, tracks, host=_outro_host())
    dummy.mass.activate_program_bus("queue-1")
    dummy._dj_queues["queue-1"].station_id = "morning_show"
    dummy.player_queues._queue.current_item = tracks[0]
    dummy.player_queues._queue.corrected_elapsed_time = POSITION_SECONDS

    await dummy._replan_queue("queue-1")

    outros = [term for term in _terms(dummy) if term.successor_id is None]
    assert len(outros) == 1
    assert outros[0].queue_item_id == tracks[3].queue_item_id
    assert dummy.player_queues.loads == []


async def test_switching_the_dj_off_lets_go_of_its_terms(tmp_path: Path) -> None:
    """A disabled DJ takes its placed terms back instead of leaving them on the bus."""
    dummy, tracks = _overlay_dj(tmp_path)
    await dummy._replan_queue("queue-1")
    queue = dummy.player_queues._queue
    queue.current_item = tracks[1]
    queue.corrected_elapsed_time = TRACK_SECONDS - 8
    await dummy._pump_terms("queue-1")
    term = next(term for term in _terms(dummy) if term.state is TermState.SCHEDULED)

    await dummy.set_queue_dj("queue-1", None)

    assert term.state is TermState.DROPPED
    assert term.dropped_reason == "flow_ended"
    assert dummy.mass.streams.cancelled == [
        ("queue-1", OVERLAY_LABEL_TRANSITION, "flow_ended", term.scheduled_frame)
    ]


async def test_a_pause_keeps_the_term_and_a_stop_ends_it(tmp_path: Path) -> None:
    """A pause is not a lost boundary; a stopped playback is."""
    dummy, _tracks = _overlay_dj(tmp_path)
    await dummy._replan_queue("queue-1")
    queue = dummy.player_queues._queue

    queue.state = PlaybackState.PAUSED
    await _drain_tasks(dummy.mass)
    assert all(not term.is_terminal for term in _terms(dummy))

    queue.state = PlaybackState.IDLE
    await dummy._pump_terms("queue-1")
    assert {term.dropped_reason for term in _terms(dummy)} == {"flow_ended"}


async def test_three_boundaries_are_three_terms(tmp_path: Path) -> None:
    """Each boundary of a pass is decided on its own, never merged into one program."""
    dummy, tracks = _overlay_dj(tmp_path)

    await dummy._replan_queue("queue-1")

    successors = [term.successor_id for term in _terms(dummy)]
    assert successors == [tracks[1].queue_item_id, tracks[2].queue_item_id, tracks[3].queue_item_id]


async def test_a_split_section_stays_one_term_at_its_boundary(tmp_path: Path) -> None:
    """Sections planned for the same boundary are spoken as one term, in order."""
    dummy = _make_replan_dj(tmp_path, [_track(index) for index in range(4)])
    dummy.mass.activate_program_bus("queue-1")
    dummy._hosts["rick"]["section_ids"] = ["Song_Transition", "News"]
    dummy._sections["News"] = {
        "id": "News",
        "name": "News",
        "type": "ai_text",
        "web_search": "disabled",
        "prompt": "Give the headlines",
        "constraints": {"max_chars": 200},
    }
    dummy._hosts["rick"]["section_order"] = [
        {
            "when": "between_songs",
            "flow": [{"MUST": "Song_Transition"}, {"MUST": "News"}],
        }
    ]

    await dummy._replan_queue("queue-1")

    assert len(_terms(dummy)) == 2
    assert _terms(dummy)[0].section_ids == ("Song_Transition", "News")


async def test_the_switch_on_holds_the_flow_instead_of_the_item_claims(tmp_path: Path) -> None:
    """The bus only exists in a flow stream, so its claim replaces the item claims."""
    dj = _dj_harness(tmp_path)
    _let_amy_lead(dj)
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])
    dj.mass.streams.program_bus_enabled_flag = True

    await dj._ensure_show_dj("q1")

    assert dj.mass.flow_mode_calls == [("q1", True)]
    assert dj.mass.boundary_mix_calls == [("q1", False)]
    assert dj.mass.lead_hold_calls == [("q1", False)]


async def test_the_switch_off_keeps_the_item_claims(tmp_path: Path) -> None:
    """With the switch off the claims that the item path needs are placed as before."""
    dj = _dj_harness(tmp_path)
    _let_amy_lead(dj)
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])

    await dj._ensure_show_dj("q1")

    # no flow claim is placed or released, so the path is the one it always was
    assert dj.mass.flow_mode_calls == []
    assert dj.mass.boundary_mix_calls == [("q1", True)]
    assert dj.mass.lead_hold_calls == [("q1", True)]


async def test_disabling_a_switch_on_dj_releases_the_flow(tmp_path: Path) -> None:
    """Letting go of the DJ also lets go of the stream it claimed."""
    dj = _dj_harness(tmp_path)
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])
    dj.mass.streams.program_bus_enabled_flag = True
    await dj._ensure_show_dj("q1")

    await dj.set_queue_dj("q1", None)

    assert dj.mass.flow_mode_calls == [("q1", True), ("q1", False)]


async def test_a_restored_dj_claims_the_flow_but_speaks_nothing(tmp_path: Path) -> None:
    """A restart has no terms and owes no intro, but its stream is claimed for the next play."""
    first = _dj_harness(tmp_path)
    first.mass.streams.program_bus_enabled_flag = True
    _fake_queue_with_sources(first, "q1", [f"{first.instance_id}://radio/morning_show"])
    await first._ensure_show_dj("q1")
    await _drain_tasks(first.mass)

    restored = _dj_harness(tmp_path)
    restored.mass.streams.program_bus_enabled_flag = True
    await restored._load_queue_dj()

    assert restored.mass.flow_mode_calls == [("q1", True)]
    # the flow is claimed by this provider, which holds it on behalf of the host
    assert restored.mass.flow_claim_owners == [("q1", restored.instance_id)]
    # the bus carries the breaks itself, so the item path's claims are let go
    assert restored.mass.boundary_mix_calls == [("q1", False)]
    assert restored.mass.lead_hold_calls == [("q1", False)]
    assert restored._term_store("q1").terms() == ()
    assert restored.mass.overlays == []


async def test_a_restored_dj_places_no_flow_claim_with_the_switch_off(tmp_path: Path) -> None:
    """With the switch off a restart claims nothing the item path did not claim before."""
    armed = _dj_harness(tmp_path)
    await armed.set_queue_dj("q1", "rick")
    await _drain_tasks(armed.mass)

    restored = _dj_harness(tmp_path)
    await restored._load_queue_dj()

    assert restored.mass.flow_mode_calls == []
    # the host has no section that straddles a boundary, so both claims are let go
    assert restored.mass.boundary_mix_calls == [("q1", False)]
    assert restored.mass.lead_hold_calls == [("q1", False)]


async def test_a_restored_dj_claims_its_boundaries_with_the_switch_off(tmp_path: Path) -> None:
    """
    A restart with the switch off claims the boundaries its sections talk over.

    The item path places those claims when the DJ is set, not when it is restored, so a
    restart used to come back without them: the host's ``allow_pre`` breaks then no longer
    found a held tail or a mixed boundary, and the DJ could not talk over one until the user
    set it again.
    """
    armed = _dj_harness(tmp_path)
    _let_amy_lead(armed)
    await armed.set_queue_dj("q1", "amy")
    await _drain_tasks(armed.mass)

    restored = _dj_harness(tmp_path)
    _let_amy_lead(restored)
    await restored._load_queue_dj()

    assert restored.mass.boundary_mix_calls == [("q1", True)]
    assert restored.mass.lead_hold_calls == [("q1", True)]
    # nothing of the bus is claimed while the switch is off
    assert restored.mass.flow_mode_calls == []


async def test_claiming_a_restored_djs_boundaries_again_changes_nothing(tmp_path: Path) -> None:
    """The claim is idempotent: the queue's next event places the same requirement again."""
    armed = _dj_harness(tmp_path)
    _let_amy_lead(armed)
    await armed.set_queue_dj("q1", "amy")
    await _drain_tasks(armed.mass)

    restored = _dj_harness(tmp_path)
    _let_amy_lead(restored)
    await restored._load_queue_dj()

    restored._ensure_stream_claims("q1")
    restored._ensure_stream_claims("q1")

    # every claim ever placed says the same thing, so repeating it takes nothing back
    assert set(restored.mass.boundary_mix_calls) == {("q1", True)}
    assert set(restored.mass.lead_hold_calls) == {("q1", True)}
    assert restored.mass.flow_mode_calls == []


async def test_a_restored_dj_claims_the_flow_when_the_switch_turns_on(tmp_path: Path) -> None:
    """The switch can be unreadable at load, so the first event claims the stream."""
    armed = _dj_harness(tmp_path)
    await armed.set_queue_dj("q1", "rick")
    await _drain_tasks(armed.mass)

    # a restart reads the switch too early, so nothing is claimed at load
    restored = _dj_harness(tmp_path)
    await restored._load_queue_dj()
    assert restored.mass.flow_mode_calls == []

    restored.mass.streams.program_bus_enabled_flag = True
    await restored._on_dj_queue_event(
        cast("Any", SimpleNamespace(event=EventType.QUEUE_UPDATED, object_id="q1"))
    )

    assert restored.mass.flow_mode_calls == [("q1", True)]
    assert restored.mass.flow_claim_owners == [("q1", restored.instance_id)]


async def test_a_render_failure_books_the_cool_down_and_drops_the_term(tmp_path: Path) -> None:
    """A failing engine pauses the DJ through the same render health as the item path."""
    dummy, tracks = _overlay_dj(tmp_path)
    state = dummy._dj_queues["queue-1"]
    state.render_health.tts.failures = 2
    dummy.fail_overlay_render = True
    await dummy._replan_queue("queue-1")

    queue = dummy.player_queues._queue
    queue.current_item = tracks[1]
    queue.corrected_elapsed_time = TRACK_SECONDS - 8
    await dummy._pump_terms("queue-1")

    failed = next(term for term in _terms(dummy) if term.successor_id == tracks[2].queue_item_id)
    assert failed.dropped_reason == "render_failed"
    assert state.render_health.cooling_down(asyncio.get_running_loop().time())
    assert (
        TTS_COOLDOWN_SECONDS,
        dummy._schedule_replan,
        ("queue-1",),
        "ai_radio_dj_cooldown_queue-1",
    ) in dummy.mass.timers


async def test_an_ai_failure_books_the_long_pause_and_names_the_query(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed query pauses the AI engine for fifteen minutes, not the voice for five."""
    dummy, tracks = _overlay_dj(tmp_path)
    state = dummy._dj_queues["queue-1"]
    state.render_health.ai.failures = RENDER_FAILURES_BEFORE_COOLDOWN - 1
    dummy.fail_overlay_render = True
    dummy.fail_overlay_part = RenderPart.AI
    await dummy._replan_queue("queue-1")

    queue = dummy.player_queues._queue
    queue.current_item = tracks[1]
    queue.corrected_elapsed_time = TRACK_SECONDS - 8
    with caplog.at_level(logging.WARNING):
        await dummy._pump_terms("queue-1")

    assert state.render_health.ai_cooling_down(asyncio.get_running_loop().time())
    assert not state.render_health.tts_cooling_down(asyncio.get_running_loop().time())
    assert (
        AI_COOLDOWN_SECONDS,
        dummy._schedule_replan,
        ("queue-1",),
        "ai_radio_dj_cooldown_queue-1",
    ) in dummy.mass.timers
    assert "AI engine failed 3 times" in caplog.text
    assert "15 minutes" in caplog.text


async def test_a_restored_dj_forgets_the_item_paths_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restart with the bus on brings no item-era gaps or guards into the term path."""
    first, tracks = _overlay_dj(tmp_path)
    await first._write_queue_dj()

    restored = _make_replan_dj(tmp_path, tracks)
    arm = restored._arm_dj_state

    def _arm_with_item_era_state(queue_id: str, host_id: str) -> Any:
        # a state that comes back still holding what the item path settled before the bus
        state = arm(queue_id, host_id)
        state.decided_gap_ids = {tracks[2].queue_item_id}
        state.history = {"Song_Transition": [(1, 12.0)]}
        return state

    monkeypatch.setattr(restored, "_arm_dj_state", _arm_with_item_era_state)
    restored.mass.activate_program_bus("queue-1")
    await restored._load_queue_dj()

    state = restored._dj_queues["queue-1"]
    assert state.decided_gap_ids == set()
    assert state.history == {}

    # the boundary the restored set had settled is decided as a term by the next pass
    await restored._replan_queue("queue-1")
    assert tracks[2].queue_item_id in {term.successor_id for term in _terms(restored)}


async def test_a_restored_dj_keeps_the_item_paths_decisions_with_the_switch_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The item path still needs its ledger after a restart, so nothing is forgotten."""
    first, tracks = _overlay_dj(tmp_path)
    first.mass.streams.program_bus_enabled_flag = False
    await first._write_queue_dj()

    restored = _make_replan_dj(tmp_path, tracks)
    arm = restored._arm_dj_state

    def _arm_with_item_era_state(queue_id: str, host_id: str) -> Any:
        state = arm(queue_id, host_id)
        state.decided_gap_ids = {tracks[2].queue_item_id}
        state.history = {"Song_Transition": [(1, 12.0)]}
        return state

    monkeypatch.setattr(restored, "_arm_dj_state", _arm_with_item_era_state)
    await restored._load_queue_dj()

    state = restored._dj_queues["queue-1"]
    assert state.decided_gap_ids == {tracks[2].queue_item_id}
    assert state.history == {"Song_Transition": [(1, 12.0)]}


async def test_switching_the_bus_on_forgets_the_item_paths_decisions(tmp_path: Path) -> None:
    """A queue switched to terms starts from an empty ledger, not the item path's gaps."""
    dummy, tracks = _overlay_dj(tmp_path)
    dummy.mass.streams.program_bus_enabled_flag = False
    # the item pass can not splice the near boundary: the player buffered past it
    await _replan_with_the_guard_moved(dummy, 1)
    state = dummy._dj_queues["queue-1"]
    assert tracks[2].queue_item_id in state.decided_gap_ids
    assert state.history

    dummy.player_queues._queue.index_in_buffer = 0
    dummy.mass.activate_program_bus("queue-1")
    # the takeover forgets both ledgers once, before any term is planned
    dummy._sync_item_path("queue-1")
    assert state.decided_gap_ids == set()
    assert state.history == {}

    await dummy._replan_queue("queue-1")

    # the boundary the item path had settled is planned as a term now
    assert tracks[2].queue_item_id in {term.successor_id for term in _terms(dummy)}


async def test_the_switch_to_the_bus_keeps_the_clip_the_player_owns(tmp_path: Path) -> None:
    """A clip the player already holds survives the takeover, the ones ahead of it do not."""
    dummy, _tracks = _overlay_dj(tmp_path)
    dummy.mass.streams.program_bus_enabled_flag = False
    await dummy._replan_queue("queue-1")
    items = dummy.player_queues.items("queue-1")
    clips = [item for item in items if item.extra_attributes.get(ATTR_QUEUE_DJ)]
    assert len(clips) == 2
    # the player buffered up to the first clip before the switch came on, so it holds it
    dummy.player_queues._queue.index_in_buffer = items.index(clips[0])
    dummy.mass.activate_program_bus("queue-1")

    dummy._sync_item_path("queue-1")

    kept = [item.queue_item_id for item in dummy.player_queues.items("queue-1")]
    # the clip the player owns stays an item, and the bus is what places its head over the
    # track before it; the clip behind the guard is the one the takeover takes out
    assert clips[0].queue_item_id in kept
    assert clips[1].queue_item_id not in kept


async def test_the_switch_off_decides_every_open_gap_of_the_window(tmp_path: Path) -> None:
    """With the switch off a pass settles every gap it evaluated, however far out it is."""
    tracks = [_track(index) for index in range(21)]
    dummy = _make_replan_dj(tmp_path, tracks)
    dummy.mass.streams.program_bus_enabled_flag = False
    queue = dummy.player_queues._queue
    queue.current_item = tracks[0]
    queue.corrected_elapsed_time = POSITION_SECONDS

    await dummy._replan_queue("queue-1")

    state = dummy._dj_queues["queue-1"]
    # every gap of the window the guard left ahead of the player, however far out it is
    assert state.decided_gap_ids == {track.queue_item_id for track in tracks[2:]}
    assert _terms(dummy) == []


async def test_a_pass_decides_only_the_boundaries_its_store_can_hold(tmp_path: Path) -> None:
    """A big window decides the nearest boundaries, and leaves the far ones open."""
    tracks = [_track(index) for index in range(61)]
    dummy = _make_replan_dj(tmp_path, tracks)
    dummy.mass.activate_program_bus("queue-1")
    queue = dummy.player_queues._queue
    queue.current_item = tracks[0]
    queue.corrected_elapsed_time = POSITION_SECONDS

    await dummy._replan_queue("queue-1")

    store = dummy._term_store("queue-1")
    nearest = {track.queue_item_id for track in tracks[1 : TERMS_PER_QUEUE + 1]}
    assert len(_terms(dummy)) == TERMS_PER_QUEUE
    assert {term.successor_id for term in _terms(dummy)} == nearest
    # the nearest boundary is spoken; the ones past the horizon are still open, so the
    # playback can roll them into reach instead of losing them in the bounded store
    assert store.state((tracks[0].queue_item_id, tracks[1].queue_item_id)) is (
        BoundaryState.DECIDED
    )
    assert store.state(
        (tracks[TERMS_PER_QUEUE].queue_item_id, tracks[TERMS_PER_QUEUE + 1].queue_item_id)
    ) is (BoundaryState.OPEN)


async def test_the_horizon_rolls_forward_as_the_playback_crosses_a_boundary(tmp_path: Path) -> None:
    """Crossing a boundary frees a store slot, and the next gap in line is decided."""
    tracks = [_track(index) for index in range(61)]
    dummy = _make_replan_dj(tmp_path, tracks)
    dummy.mass.activate_program_bus("queue-1")
    queue = dummy.player_queues._queue
    queue.current_item = tracks[0]
    queue.corrected_elapsed_time = POSITION_SECONDS
    await dummy._replan_queue("queue-1")
    store = dummy._term_store("queue-1")
    assert len(_terms(dummy)) == TERMS_PER_QUEUE

    # the nearest boundary is behind the player now, so its term goes and takes its gap along
    queue.current_item = tracks[2]
    queue.current_index = 2
    queue.index_in_buffer = 2
    await dummy._pump_terms("queue-1")
    assert store.state((tracks[1].queue_item_id, tracks[2].queue_item_id)) is (BoundaryState.OPEN)

    await dummy._replan_queue("queue-1")

    # the slot the lost term held is used by the next gap in line, and no other one moved
    assert store.state((tracks[2].queue_item_id, tracks[3].queue_item_id)) is (
        BoundaryState.DECIDED
    )
    assert store.state(
        (tracks[TERMS_PER_QUEUE + 1].queue_item_id, tracks[TERMS_PER_QUEUE + 2].queue_item_id)
    ) is (BoundaryState.DECIDED)
    assert len(store.active()) <= TERMS_PER_QUEUE


async def test_the_show_outro_keeps_its_slot_when_the_store_fills_up(tmp_path: Path) -> None:
    """The outro is the farthest boundary of a show, so a slot is kept free for it."""
    tracks = [_track(index) for index in range(41)]
    dummy = _make_replan_dj(tmp_path, tracks, host=_outro_host())
    dummy.mass.activate_program_bus("queue-1")
    dummy._dj_queues["queue-1"].station_id = "morning_show"
    queue = dummy.player_queues._queue
    queue.current_item = tracks[0]
    queue.corrected_elapsed_time = POSITION_SECONDS

    await dummy._replan_queue("queue-1")

    store = dummy._term_store("queue-1")
    outros = [term for term in _terms(dummy) if term.successor_id is None]
    assert len(outros) == 1
    assert outros[0].queue_item_id == tracks[-1].queue_item_id
    assert len(_terms(dummy)) == TERMS_PER_QUEUE
    # the boundaries between the running item and the outro keep the rest of the store
    assert len([term for term in _terms(dummy) if term.successor_id is not None]) == (
        TERMS_PER_QUEUE - 1
    )
    assert store.state((tracks[1].queue_item_id, tracks[2].queue_item_id)) is (
        BoundaryState.DECIDED
    )


async def test_a_pass_that_plans_the_show_outro_gives_up_no_break_it_just_decided(
    tmp_path: Path,
) -> None:
    """The outro's slot is kept free, so no break is decided only to be given up at once."""
    tracks = [_track(index) for index in range(41)]
    dummy = _make_replan_dj(tmp_path, tracks, host=_outro_host())
    dummy.mass.activate_program_bus("queue-1")
    dummy._dj_queues["queue-1"].station_id = "morning_show"
    queue = dummy.player_queues._queue
    queue.current_item = tracks[0]
    queue.corrected_elapsed_time = POSITION_SECONDS
    reported: list[ProgramTerm] = []
    give_up = dummy._term_evicted

    def _report(queue_id: str, term: ProgramTerm) -> None:
        reported.append(term)
        give_up(queue_id, term)

    dummy._term_evicted = _report  # type: ignore[method-assign]

    await dummy._replan_queue("queue-1")

    # a pass that fills the store to the brim and then lets the outro push a term out has
    # planned a break, claimed its guard weight and given it up in the same breath
    assert reported == []
    store = dummy._term_store("queue-1")
    outros = [term for term in _terms(dummy) if term.successor_id is None]
    assert len(outros) == 1
    assert outros[0].queue_item_id == tracks[-1].queue_item_id
    assert store.state((tracks[-1].queue_item_id, None)) is BoundaryState.DECIDED


async def test_a_dropped_term_reopens_its_gap_and_an_aired_one_does_not(
    tmp_path: Path,
) -> None:
    """A break that never aired frees its boundary; one the listener heard keeps it settled."""
    dummy, _tracks = _overlay_dj(tmp_path)
    await dummy._replan_queue("queue-1")
    store = dummy._term_store("queue-1")
    aired, dropped, _other = _terms(dummy)
    aired.mark_aired()

    dummy._drop_terms("queue-1", TermDropReason.FLOW_ENDED)

    assert store.state((aired.queue_item_id, aired.successor_id)) is BoundaryState.DECIDED
    assert store.state((dropped.queue_item_id, dropped.successor_id)) is BoundaryState.OPEN


async def test_only_the_dropped_boundary_is_planned_again(tmp_path: Path) -> None:
    """With two boundaries open, only the one whose term was dropped is decided again."""
    dummy, _tracks = _overlay_dj(tmp_path)
    await dummy._replan_queue("queue-1")
    store = dummy._term_store("queue-1")
    _running, first, second = _terms(dummy)

    dummy._drop_term("queue-1", store, first, TermDropReason.FLOW_ENDED)
    await dummy._replan_queue("queue-1")

    active = [term for term in _terms(dummy) if not term.is_terminal]
    # the boundary that kept its term is never decided a second time
    assert any(term is second for term in active)
    assert any(term is not second and term.successor_id == first.successor_id for term in active)


def _record_replans(dummy: ReplanQueueDJ, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every replan request of the DJ, and run it exactly as the real one does."""
    requests: list[str] = []
    scheduled = dummy._schedule_replan

    def _record(queue_id: str) -> None:
        requests.append(queue_id)
        scheduled(queue_id)

    monkeypatch.setattr(dummy, "_schedule_replan", _record)
    return requests


def _a_full_store_of_far_terms(
    dummy: ReplanQueueDJ, tracks: list[FakeQueueItem], *, first: int = 8
) -> None:
    """Fill the store with terms for far boundaries, as a long pass left them behind."""
    store = dummy._term_store("queue-1")
    for index in range(first, first + TERMS_PER_QUEUE):
        store.add(
            ProgramTerm(
                queue_item_id=tracks[index].queue_item_id,
                successor_id=tracks[index + 1].queue_item_id,
                section_ids=("Song_Transition",),
                head_seconds=0.0,
            ),
            position=index,
        )


def _live_queue(
    tmp_path: Path, *, duration: int = 235, elapsed: float = 83
) -> tuple[ReplanQueueDJ, list[FakeQueueItem]]:
    """Build the queue of the live report: one long track playing, a store full of far terms."""
    tracks = [_track(index) for index in range(61)]
    for track in tracks:
        track.duration = duration
    dummy = _make_replan_dj(tmp_path, tracks)
    dummy.mass.activate_program_bus("queue-1")
    dummy.mass.streams.crossfade_mode = CrossfadeMode.STANDARD_CROSSFADE
    # a queue whose switch already took over: its next pass keeps the settled gaps
    dummy._dj_queues["queue-1"].bus_took_over = True
    queue = dummy.player_queues._queue
    queue.current_item = tracks[0]
    queue.corrected_elapsed_time = elapsed
    _a_full_store_of_far_terms(dummy, tracks)
    # an earlier pass settled the boundary the running item ends at without a break, so
    # only a pass that opens that gap again can still speak it
    dummy._term_store("queue-1").settle([(tracks[0].queue_item_id, tracks[1].queue_item_id)])
    return dummy, tracks


async def _pause_and_resume(dummy: ReplanQueueDJ) -> None:
    """Let a pump see the pause, so the play that follows it is the change it looks for."""
    queue = dummy.player_queues._queue
    queue.state = PlaybackState.PAUSED
    await dummy._pump_terms("queue-1")
    queue.state = PlaybackState.PLAYING


async def test_a_pump_replans_the_boundary_the_running_item_ends_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pause and resume leaves the running boundary unserved, and the pump asks for it."""
    dummy, tracks = _live_queue(tmp_path)
    store = dummy._term_store("queue-1")
    requests = _record_replans(dummy, monkeypatch)

    await _pause_and_resume(dummy)
    # the pause decides nothing about the boundary the earlier pass settled
    assert requests == []

    await dummy._pump_terms("queue-1")

    # the gap the earlier pass had settled is open again, and the pass that serves it asked
    assert store.state((tracks[0].queue_item_id, tracks[1].queue_item_id)) is (BoundaryState.OPEN)
    assert requests == ["queue-1"]

    await _drain_tasks(dummy.mass)

    spoken = next(term for term in store.active() if term.queue_item_id == tracks[0].queue_item_id)
    assert spoken.successor_id == tracks[1].queue_item_id
    # the store stayed bounded: the farthest term made room for the running boundary
    assert len(store.active()) == TERMS_PER_QUEUE
    assert all(term.queue_item_id != tracks[23].queue_item_id for term in store.active())
    assert store.state((tracks[23].queue_item_id, tracks[24].queue_item_id)) is (BoundaryState.OPEN)


async def test_the_replanned_term_speaks_at_the_end_of_the_running_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The boundary of the running item is timed from its own position, not from a far one."""
    dummy, tracks = _live_queue(tmp_path)
    _record_replans(dummy, monkeypatch)
    await _pause_and_resume(dummy)
    await dummy._pump_terms("queue-1")
    await _drain_tasks(dummy.mass)

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        await dummy._pump_terms("queue-1")

    line = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith(f"TERM queue=queue-1 item={tracks[0].queue_item_id} ")
    )
    # the running item plays, so its boundary is 235 - 83 seconds away, not the far one a
    # term of the same store reports
    assert "pos=current" in line
    assert "remaining=152.00" in line


def _silent_host() -> dict[str, Any]:
    """Return a host whose only break never rolls, so a pass legitimately decides nothing."""
    host = _must_host()
    host["section_order"] = [
        {
            "when": "between_songs",
            "flow": [{"OPTIONAL": {"section": "Song_Transition", "chance": 0.0}}],
        }
    ]
    return host


def _let_the_replan_period_pass(dummy: ReplanQueueDJ) -> None:
    """Move the pump's rate limit back, as a pump period passing does."""
    dummy._term_boundary_replan_due["queue-1"] = asyncio.get_running_loop().time() - 1.0


async def _a_queue_a_pass_cannot_serve(tmp_path: Path) -> ReplanQueueDJ:
    """Return a playing queue whose planner speaks no break at any boundary."""
    tracks = [_track(index) for index in range(21)]
    dummy = _make_replan_dj(tmp_path, tracks, host=_silent_host())
    dummy.mass.activate_program_bus("queue-1")
    dummy.player_queues._queue.current_item = tracks[0]
    await dummy._replan_queue("queue-1")
    return dummy


async def test_two_pumps_in_a_row_request_the_replan_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The running boundary is asked for once per period, however often the pump runs."""
    dummy = await _a_queue_a_pass_cannot_serve(tmp_path)
    requests = _record_replans(dummy, monkeypatch)
    queue = dummy.player_queues._queue

    # the resume is what opens the boundary the pause left settled
    queue.state = PlaybackState.PAUSED
    await dummy._pump_terms("queue-1")
    queue.state = PlaybackState.PLAYING

    await dummy._pump_terms("queue-1")
    await dummy._pump_terms("queue-1")

    assert requests == ["queue-1"]


async def test_a_pass_that_plans_nothing_settles_what_it_looked_at(tmp_path: Path) -> None:
    """A gap the host had nothing to say at is decided, so no later pass rolls it again."""
    dummy = await _a_queue_a_pass_cannot_serve(tmp_path)

    store = dummy._term_store("queue-1")
    settled = [key for key in store.boundaries() if store.state(key) is BoundaryState.DECIDED]

    assert settled
    # nothing is spoken there, so the store keeps the boundary without a term
    assert all(store.term_of(key) is None for key in settled)
    assert _terms(dummy) == []
    # and the next pass does not decide it a second time
    await dummy._replan_queue("queue-1")
    assert _terms(dummy) == []


async def test_a_pass_that_decides_nothing_is_not_asked_again_by_the_next_pump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A planner that legitimately decides nothing must not turn the pump into a carousel."""
    dummy = await _a_queue_a_pass_cannot_serve(tmp_path)
    requests = _record_replans(dummy, monkeypatch)
    queue = dummy.player_queues._queue

    # a pause is no cause; the resume that follows it is the one that asks for the boundary
    queue.state = PlaybackState.PAUSED
    await dummy._pump_terms("queue-1")
    assert requests == []

    queue.state = PlaybackState.PLAYING
    await dummy._pump_terms("queue-1")
    await _drain_tasks(dummy.mass)
    assert requests == ["queue-1"]

    # the pass ran and decided nothing, so the boundary is settled again - and no pump of
    # this period rolls it a second time
    _let_the_replan_period_pass(dummy)
    await dummy._pump_terms("queue-1")
    assert requests == ["queue-1"]


async def test_a_burst_of_queue_events_runs_one_pump_and_one_follow_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fifty events share one running pass, and the requests that landed get one more."""
    dummy, _tracks = _overlay_dj(tmp_path)
    gate = asyncio.Event()
    passes: list[str] = []
    running = 0
    peak = 0

    async def _held_pump(queue_id: str) -> None:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        passes.append(queue_id)
        try:
            if len(passes) == 1:
                # the first pass is held open while the other events land
                await gate.wait()
        finally:
            running -= 1

    monkeypatch.setattr(dummy, "_pump_terms", _held_pump)

    dummy._pump_terms_soon("queue-1")
    await asyncio.sleep(0)
    for _ in range(49):
        dummy._pump_terms_soon("queue-1")

    assert passes == ["queue-1"]
    assert peak == 1

    gate.set()
    await _drain_tasks(dummy.mass)

    # the pass that landed during the running one is served afterwards, exactly once
    assert passes == ["queue-1", "queue-1"]
    assert peak == 1


async def test_a_failed_pump_leaves_no_request_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pass that raises does not leave a request behind that would block the next one."""
    dummy, _tracks = _overlay_dj(tmp_path)
    passes: list[str] = []

    async def _failing_pump(queue_id: str) -> None:
        passes.append(queue_id)
        if len(passes) == 1:
            # a request that lands while the failing pass runs must not stay pending
            dummy._pump_terms_soon(queue_id)
            raise RuntimeError("the render failed")

    monkeypatch.setattr(dummy, "_pump_terms", _failing_pump)

    dummy._pump_terms_soon("queue-1")
    await _drain_tasks(dummy.mass)

    dummy._pump_terms_soon("queue-1")
    await _drain_tasks(dummy.mass)

    assert passes == ["queue-1", "queue-1"]


async def test_a_near_boundary_takes_the_slot_of_the_farthest_term(tmp_path: Path) -> None:
    """A full store of far terms does not keep the near boundary from being planned."""
    dummy, tracks = _live_queue(tmp_path)
    store = dummy._term_store("queue-1")
    store.reopen((tracks[0].queue_item_id, tracks[1].queue_item_id))

    await dummy._replan_queue("queue-1")

    active = store.active()
    assert len(active) == TERMS_PER_QUEUE
    assert any(
        term.queue_item_id == tracks[0].queue_item_id
        and term.successor_id == tracks[1].queue_item_id
        for term in active
    )
    assert store.state((tracks[0].queue_item_id, tracks[1].queue_item_id)) is (
        BoundaryState.DECIDED
    )
    # the farthest term made room: the store gave it up, and its gap is open again
    assert all(term.queue_item_id != tracks[23].queue_item_id for term in active)
    assert store.state((tracks[23].queue_item_id, tracks[24].queue_item_id)) is (BoundaryState.OPEN)


async def test_the_tracks_after_the_running_one_take_the_slots_of_the_far_terms(
    tmp_path: Path,
) -> None:
    """
    A store full of far terms must not leave the tracks behind the running one silent.

    The live report: the store held sixteen terms that were all minutes away, no boundary
    carried a term for the track that played or for the eight after it, and the pump kept
    reporting that full store while nothing was placed, dropped or logged. The store bound
    is no reason to leave out a boundary that is due before everything the store holds: the
    near ones take the slots of the far ones, exactly as the running boundary does.
    """
    dummy, tracks = _live_queue(tmp_path)
    store = dummy._term_store("queue-1")
    # the boundary of the running track is open again, and the eight after it are undecided
    store.reopen((tracks[0].queue_item_id, tracks[1].queue_item_id))

    await dummy._replan_queue("queue-1")

    held = {term.queue_item_id: term for term in store.active()}
    for index in range(5):
        assert tracks[index].queue_item_id in held, f"track {index} is left without a term"
        assert held[tracks[index].queue_item_id].successor_id == tracks[index + 1].queue_item_id
    # and the store stayed bounded by giving up the terms that are due last
    assert len(store.active()) == TERMS_PER_QUEUE
    assert all(term.queue_item_id != tracks[23].queue_item_id for term in store.active())


async def test_the_switch_off_requests_no_replan_from_the_pump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the switch off the pump drives nothing, and the item path is the one it was."""
    tracks = [_track(index) for index in range(4)]
    for track in tracks:
        track.duration = TRACK_SECONDS
    dummy = _make_replan_dj(tmp_path, tracks)
    dummy.player_queues._queue.current_item = tracks[0]
    dummy.player_queues._queue.corrected_elapsed_time = POSITION_SECONDS
    requests = _record_replans(dummy, monkeypatch)

    await dummy._replan_queue("queue-1")
    await dummy._pump_terms("queue-1")

    assert requests == []
    assert _terms(dummy) == []
    # the item path still splices its clips into the queue, exactly as on dev
    assert len(dummy.player_queues.loads) == 2


def _render_of(seconds: float) -> Any:
    """Return a stand-in section render that speaks the given number of seconds."""

    async def _render(
        _queue_id: str,
        _program: dict[str, Any],
        _section: Any,
        pcm_format: Any,
    ) -> bytes:
        frame_bytes = (pcm_format.bit_depth // 8) * pcm_format.channels
        return b"\x00" * (int(seconds * pcm_format.sample_rate) * frame_bytes)

    return _render


def _rendered_seconds(pcm: bytes) -> float:
    """Return the length of a placeholder program in the flow's format, in seconds."""
    frame_bytes = (PLAYBACK_PCM_FORMAT.bit_depth // 8) * PLAYBACK_PCM_FORMAT.channels
    return len(pcm) / frame_bytes / PLAYBACK_PCM_FORMAT.sample_rate


async def _speak_a_term(dummy: ReplanQueueDJ) -> None:
    """Let the player reach a boundary of the harness and let its term air."""
    tracks = dummy.player_queues._items
    dummy._sections["Song_Transition"]["allow_post"] = True
    await dummy._replan_queue("queue-1")
    await _due_term(dummy, tracks, 1, 8.0)


async def test_a_contribution_over_the_hosts_ceiling_is_cut_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A term that speaks longer than the host allows is cut to the ceiling, and logged."""
    dummy, _tracks = _overlay_dj(tmp_path)
    dummy._render_section_pcm = _render_of(70.0)  # type: ignore[method-assign]

    with caplog.at_level(logging.INFO):
        await _speak_a_term(dummy)

    assert len(dummy.mass.overlays) == 1
    pcm = dummy.mass.overlays[0][1]
    assert _rendered_seconds(pcm) == pytest.approx(MAX_CONTRIBUTION_DEFAULT_SECONDS)
    assert "queue-1" in caplog.text
    assert "Song_Transition" in caplog.text


async def test_the_hosts_own_ceiling_decides_where_the_cut_lands(tmp_path: Path) -> None:
    """A host that allows twenty seconds hears twenty seconds of a longer contribution."""
    host = _must_host()
    host["max_contribution_seconds"] = 20
    dummy, _tracks = _overlay_dj(tmp_path, host=host)
    dummy._render_section_pcm = _render_of(70.0)  # type: ignore[method-assign]

    await _speak_a_term(dummy)

    assert len(dummy.mass.overlays) == 1
    assert _rendered_seconds(dummy.mass.overlays[0][1]) == pytest.approx(20.0)


async def test_a_contribution_inside_the_ceiling_is_spoken_whole(tmp_path: Path) -> None:
    """A short break is never padded or trimmed: the cap only cuts what is too long."""
    dummy, _tracks = _overlay_dj(tmp_path)
    dummy._render_section_pcm = _render_of(12.0)  # type: ignore[method-assign]

    await _speak_a_term(dummy)

    assert len(dummy.mass.overlays) == 1
    assert _rendered_seconds(dummy.mass.overlays[0][1]) == pytest.approx(12.0)


async def test_the_intro_is_not_cut_by_the_contribution_ceiling(tmp_path: Path) -> None:
    """The intro keeps its own length: the host's character limit is its only ceiling."""
    host = _intro_host()
    host["max_contribution_seconds"] = MAX_CONTRIBUTION_MIN_SECONDS
    dummy = _armed_intro_dj(tmp_path, host=host)
    dummy._render_section_pcm = _render_of(30.0)  # type: ignore[method-assign]

    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)

    assert len(dummy.mass.overlays) == 1
    assert _rendered_seconds(dummy.mass.overlays[0][1]) == pytest.approx(30.0)


async def test_a_pass_with_the_bus_on_leaves_the_item_paths_gaps_alone(tmp_path: Path) -> None:
    """The term path owns its boundaries in the store and writes no item-path ledger."""
    tracks = [_track(index) for index in range(6)]
    dummy = _make_replan_dj(tmp_path, tracks)
    dummy.mass.activate_program_bus("queue-1")
    dummy.player_queues._queue.current_item = tracks[0]
    dummy.player_queues._queue.corrected_elapsed_time = POSITION_SECONDS

    await dummy._replan_queue("queue-1")

    assert _terms(dummy)
    # the item path's set stays empty: nothing of the bus path is booked there
    assert dummy._dj_queues["queue-1"].decided_gap_ids == set()
    store = dummy._term_store("queue-1")
    assert store.state((tracks[0].queue_item_id, tracks[1].queue_item_id)) is (
        BoundaryState.DECIDED
    )


def _chance_host(chance: float) -> dict[str, Any]:
    """Return a host whose only break rolls the given chance at every boundary."""
    host = _must_host()
    host["section_order"] = [
        {
            "when": "between_songs",
            "flow": [{"OPTIONAL": {"section": "Song_Transition", "chance": chance}}],
        }
    ]
    return host


class _AlwaysLowRoll:
    """A chance roll that always lands under any chance above zero."""

    def random(self) -> float:
        """Return the lowest roll there is."""
        return 0.0


def _a_chance_that_always_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every chance roll of the planner win, so a re-evaluated slot always speaks."""
    monkeypatch.setattr(
        music_assistant.providers.ai_radio.runtime,
        "random",
        SimpleNamespace(Random=_AlwaysLowRoll),
    )


def _a_queue_whose_running_boundary_is_settled(
    tmp_path: Path, host: dict[str, Any] | None = None
) -> tuple[ReplanQueueDJ, list[FakeQueueItem]]:
    """Return a playing queue whose pass settled the boundary of the running item."""
    tracks = [_track(index) for index in range(21)]
    dummy = _make_replan_dj(tmp_path, tracks, host=host or _must_host())
    dummy.mass.activate_program_bus("queue-1")
    queue = dummy.player_queues._queue
    queue.current_item = tracks[0]
    queue.corrected_elapsed_time = POSITION_SECONDS
    # a pass that looked at the boundary the running item ends at found nothing to say there
    dummy._term_store("queue-1").settle([(tracks[0].queue_item_id, tracks[1].queue_item_id)])
    return dummy, tracks


async def test_a_settled_boundary_is_not_decided_again_by_pumps_and_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once a pass settled a boundary, neither a pump nor a pass rolls it a second time."""
    dummy, tracks = _a_queue_whose_running_boundary_is_settled(tmp_path)
    store = dummy._term_store("queue-1")
    key = (tracks[0].queue_item_id, tracks[1].queue_item_id)
    requests = _record_replans(dummy, monkeypatch)

    for _ in range(10):
        await dummy._pump_terms("queue-1")
        await _drain_tasks(dummy.mass)
        await dummy._replan_queue("queue-1")

    assert store.state(key) is BoundaryState.DECIDED
    assert store.term_of(key) is None
    assert not [term for term in _terms(dummy) if term.queue_item_id == tracks[0].queue_item_id]
    # the pump asked for no pass of its own: nothing was missed, so nothing is reopened
    assert requests == []


def _record_decided_ids(
    dummy: ReplanQueueDJ, monkeypatch: pytest.MonkeyPatch, seen: list[set[str] | None]
) -> None:
    """Record the settled boundaries every pass hands to the planner."""
    plan_sections = dummy._plan_sections

    def _record(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        seen.append(kwargs.get("decided_next_item_ids"))
        return plan_sections(*args, **kwargs)

    monkeypatch.setattr(dummy, "_plan_sections", _record)


async def test_a_boundary_the_host_declined_keeps_its_no(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A boundary a pass declined is never rolled again, even with a chance that would win."""
    _a_chance_that_always_wins(monkeypatch)
    dummy, tracks = _a_queue_whose_running_boundary_is_settled(tmp_path, _chance_host(0.3))
    store = dummy._term_store("queue-1")
    key = (tracks[0].queue_item_id, tracks[1].queue_item_id)
    decided: list[set[str] | None] = []
    _record_decided_ids(dummy, monkeypatch, decided)

    for _ in range(10):
        await dummy._replan_queue("queue-1")

    assert store.state(key) is BoundaryState.DECIDED
    assert store.term_of(key) is None
    # the planner is told what the store decided, so no pass rolls that boundary's chance again
    assert len(decided) == 10
    assert all(item_ids is not None and tracks[1].queue_item_id in item_ids for item_ids in decided)


async def test_a_transient_drop_opens_its_boundary_for_the_next_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A break lost to a missing bus is decided again, the way round a pump is there for."""
    dummy, tracks = _overlay_dj(tmp_path)
    await dummy._replan_queue("queue-1")
    store = dummy._term_store("queue-1")
    term = next(term for term in _terms(dummy) if term.queue_item_id == tracks[0].queue_item_id)
    dummy._drop_term("queue-1", store, term, TermDropReason.NO_BUS)
    requests = _record_replans(dummy, monkeypatch)

    await dummy._pump_terms("queue-1")

    assert store.state(term.key) is BoundaryState.OPEN
    assert requests == ["queue-1"]

    await _drain_tasks(dummy.mass)

    # the pass the pump asked for speaks the boundary again
    assert store.state(term.key) is BoundaryState.DECIDED
    assert store.term_of(term.key) is not term


async def test_a_resume_opens_the_boundary_the_pause_leaves_settled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pause is no reason to decide again; the play that follows it is."""
    dummy, tracks = _a_queue_whose_running_boundary_is_settled(tmp_path)
    store = dummy._term_store("queue-1")
    queue = dummy.player_queues._queue
    key = (tracks[0].queue_item_id, tracks[1].queue_item_id)
    requests = _record_replans(dummy, monkeypatch)

    queue.state = PlaybackState.PAUSED
    await dummy._pump_terms("queue-1")

    assert store.state(key) is BoundaryState.DECIDED
    assert requests == []

    queue.state = PlaybackState.PLAYING
    await dummy._pump_terms("queue-1")

    assert store.state(key) is BoundaryState.OPEN
    assert requests == ["queue-1"]


async def test_with_the_switch_off_the_item_path_keeps_its_own_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the switch off a settled gap stays settled in the item path's own ledger."""
    _a_chance_that_always_wins(monkeypatch)
    tracks = [_track(index) for index in range(4)]
    for track in tracks:
        track.duration = TRACK_SECONDS
    dummy = _make_replan_dj(tmp_path, tracks, host=_chance_host(0.3))
    dummy.mass.streams.program_bus_enabled_flag = False
    queue = dummy.player_queues._queue
    queue.current_item = tracks[0]
    queue.corrected_elapsed_time = POSITION_SECONDS
    dummy._dj_queues["queue-1"].decided_gap_ids = {tracks[1].queue_item_id}

    await dummy._replan_queue("queue-1")

    announced = _announced_gaps(dummy)
    # the gap the item path settled was not rolled again, the ones after it were
    assert tracks[1].queue_item_id not in announced
    assert tracks[2].queue_item_id in announced
    assert dummy._term_store("queue-1").active() == ()


async def test_a_switched_host_opens_the_boundaries_the_old_one_left_settled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another host's sections say nothing about the silence the previous host decided."""
    dummy, tracks = _a_queue_whose_running_boundary_is_settled(tmp_path)
    store = dummy._term_store("queue-1")
    key = (tracks[0].queue_item_id, tracks[1].queue_item_id)
    await dummy._replan_queue("queue-1")
    requests = _record_replans(dummy, monkeypatch)

    await dummy._pump_terms("queue-1")
    # the program is the one the boundary was settled under, so nothing is opened
    assert store.state(key) is BoundaryState.DECIDED
    assert requests == []

    dummy._dj_queues["queue-1"].host_id = "amy"
    await dummy._pump_terms("queue-1")

    assert store.state(key) is BoundaryState.OPEN
    assert requests == ["queue-1"]


async def test_switched_sections_open_what_the_old_ones_left_settled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same host with other sections plans another show, so its silence is re-decided."""
    dummy, tracks = _a_queue_whose_running_boundary_is_settled(tmp_path)
    store = dummy._term_store("queue-1")
    key = (tracks[0].queue_item_id, tracks[1].queue_item_id)
    await dummy._replan_queue("queue-1")
    requests = _record_replans(dummy, monkeypatch)
    await dummy._pump_terms("queue-1")
    assert requests == []

    program = store.program or {}
    store.program = {
        **program,
        "sections": [*(program.get("sections") or []), {"id": "Weather"}],
    }
    await dummy._pump_terms("queue-1")

    assert store.state(key) is BoundaryState.OPEN
    assert requests == ["queue-1"]

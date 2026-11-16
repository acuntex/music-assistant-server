"""
Golden bytes of the flag-off flow, and the anchor drift a clip's lead reports.

The scenario is two tracks with a voice-over clip between them, carrying the pre the
program bus places over the outgoing track's tail and the post it plays into the next
one. With the experiment off that clip is just an item and the flow is the item path's:
the bytes it emits are frozen here, twice - on a queue that cuts its items at their
boundaries, and on one that crossfades them, where the clip's head is the boundary's voice
over, the track's tail is held back for the whole lead the pre may reach over, and the
track after the clip arrives through the prefetcher instead of a second decode. A change
to either flag-off path then shows up as a changed hash instead of only as a differently
sounding stream. The frozen streams pin this branch with the bus off, not `dev`, for two
reasons. The first is the item path itself, which never held a tail for a lead on `dev`:
the "nothing changed" comparison left is the bus path's, and it compares against this
branch's own flag-off run. The second is `delivered_media_covers_item`: `dev` wrote the
streamed length back as the item's duration after every pass, and here a delivery that
stops short of the declared length no longer stands in for it, which the item path
observes as well. With the experiment on and nothing planned, the bus may count frames
but must not change one of them. And the placement a lead derives from declared durations
is held against the audio the item really served, drift included.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any, NamedTuple, cast
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import CrossfadeMode, MediaType
from music_assistant_models.errors import QueueEmpty
from music_assistant_models.streamdetails import TailOverlap

from music_assistant.controllers.streams.audio import StreamsAudio
from music_assistant.controllers.streams.constants import (
    CONF_STREAM_PROGRAM_BUS,
    LEAD_HOLD_CEILING,
    PRE_OVERLAP_ATTR,
    PRE_OVERLAP_FROM_ATTR,
    VOICE_OVER_RAMP,
)
from tests.controllers.streams.test_crossfade_transition import (
    STANDARD_CROSSFADE_DURATION,
    TEST_PCM_FORMAT,
    _flow_audio,
    _queue_item,
)
from tests.controllers.streams.test_flow_program_bus_identity import _summing_mix
from tests.controllers.streams.test_flow_program_bus_output import (
    SECOND,
    _chunks_of,
    _drain,
    _flow,
    _frame_size,
    _music,
    _publish,
    _reference_sum,
    _voice,
)

FRAME_SIZE = _frame_size(TEST_PCM_FORMAT)
TRACK_SECONDS = 40
CLIP_SECONDS = 4
HEAD_SECONDS = 2.0
POST_SECONDS = 2.0
AFTER_SECONDS = 8
# the flag-off flow: its items, less the overlap one side of the boundary is heard in
# (the mix replaces both sides of the boundary instead of adding to them)
FLOW_SECONDS = TRACK_SECONDS + CLIP_SECONDS + AFTER_SECONDS - POST_SECONDS
# The SHA-256 of the frozen flag-off PCM. Regenerate it only together with a deliberate
# change to what the experiment-off flow sounds like (run the test, read the actual
# digest from the failure). It is a digest rather than a fixture file because the
# reference is ~1.6 MB of PCM and would dwarf every other file in the suite.
FLAG_OFF_SHA256 = "176a1076d64104c85529dd3484b89e027947d1010775b4ea1bf70087e2f618a2"
# The same, for the flag-off flow on a crossfading queue, where the lead is a boundary
# voice over instead of a program the bus places.
FLAG_OFF_CROSSFADE_SHA256 = "70e44be1e4fa7b18d84568bc80998ee1c6c6ff2e54da636b790c638269f73505"
DRIFT_LINE = "PROGRAM-BUS anchor drift=%.0fms label=%s anchor=%d produced_end=%d declared_end=%d"
CANCEL_LINE = "PROGRAM-BUS cancel reason=%s label=%s"


class _Drift(NamedTuple):
    """One anchor drift report, as the flow logged it."""

    drift_ms: float
    label: str
    anchor: int
    produced_end: int
    declared_end: int


def _scenario(
    monkeypatch: pytest.MonkeyPatch,
    *,
    program_bus: bool,
    planned: bool = True,
    delivered: dict[str, int] | None = None,
    crossfade_mode: CrossfadeMode = CrossfadeMode.DISABLED,
) -> tuple[
    StreamsAudio,
    SimpleNamespace,
    MagicMock,
    dict[str, bytes],
    SimpleNamespace,
    SimpleNamespace,
    SimpleNamespace,
]:
    """
    Build track -> clip (pre/post) -> track, serving each item what it really delivers.

    :param monkeypatch: Fixture that installs the fake sources.
    :param program_bus: Whether the flow runs with the experiment switched on.
    :param planned: Whether the clip declares the lead the bus would place over the track.
    :param delivered: Seconds each item's source really serves, when it differs from the
        item's declared duration.
    :param crossfade_mode: Crossfade mode the queue plays with.
    """
    track = _queue_item("item-1", "Track", duration=TRACK_SECONDS)
    clip = _queue_item("item-2", "Transition", duration=CLIP_SECONDS)
    clip.media_type = MediaType.SOUND_EFFECT
    clip.streamdetails.tail_overlap = TailOverlap(
        duration=POST_SECONDS, next_queue_item_id="item-3"
    )
    if planned:
        clip.extra_attributes[PRE_OVERLAP_ATTR] = HEAD_SECONDS
        clip.extra_attributes[PRE_OVERLAP_FROM_ATTR] = "item-1"
    after = _queue_item("item-3", "After", duration=AFTER_SECONDS)
    audio, queue, mass = _flow_audio(
        monkeypatch,
        next_item=clip,
        load_next=[clip, after, QueueEmpty],
        crossfade_mode=crossfade_mode,
        program_bus=program_bus,
        build_result=SimpleNamespace(
            timing_info=SimpleNamespace(
                fadein_trimmed_duration=0.0,
                crossfade_duration=POST_SECONDS,
                pre_crossfade_duration=0.0,
            )
        ),
        # a declared lead is what holds the outgoing tail wider than one crossfade
        lead_hold=planned,
    )
    audio.logger = MagicMock()
    _summing_mix(monkeypatch, audio)
    # the real rule decides whether a declared lead applies
    del audio.crossfade_allowed
    next_items = {"item-1": clip, "item-2": after}
    mass.player_queues.get_next_item.side_effect = lambda _queue_id, item_id: next_items.get(
        item_id
    )
    pcm = {
        "item-1": _music(TRACK_SECONDS),
        "item-2": _voice(CLIP_SECONDS),
        "item-3": _music(AFTER_SECONDS),
    }
    if delivered is not None:
        pcm = {
            item_id: body[: delivered[item_id] * SECOND * FRAME_SIZE]
            for item_id, body in pcm.items()
        }

    async def _item_stream(
        queue_item: SimpleNamespace, *_args: object, **kwargs: object
    ) -> AsyncGenerator[bytes]:
        item_id = queue_item.queue_item_id
        seek = int(cast("float", kwargs.get("seek_position", 0)))
        body = pcm[item_id][seek * SECOND * FRAME_SIZE :]
        for chunk in _chunks_of(body, TEST_PCM_FORMAT):
            yield chunk
            await asyncio.sleep(0)

    monkeypatch.setattr(audio, "get_queue_item_stream", _item_stream)
    return audio, queue, mass, pcm, track, clip, after


async def _run(
    audio: StreamsAudio, queue: SimpleNamespace, start: SimpleNamespace
) -> tuple[bytes, list[int]]:
    """Drain one flow session, returning its PCM and the sizes it was emitted in."""
    chunks, _anchor = await _drain(_flow(audio, queue, start), [])
    return b"".join(chunks), [len(chunk) for chunk in chunks]


def _log_records(audio: StreamsAudio) -> list[tuple[Any, ...]]:
    """Return the debug records the flow wrote, format string first."""
    return [call.args for call in cast("Any", audio.logger).debug.call_args_list]


def _program_bus_lines(audio: StreamsAudio) -> list[str]:
    """Return the format strings of the records the program bus wrote."""
    return [
        record[0]
        for record in _log_records(audio)
        if isinstance(record[0], str) and "PROGRAM-BUS" in record[0]
    ]


def _drift_reports(audio: StreamsAudio) -> list[_Drift]:
    """Return the drift reports, taken once from the line the flow logs them in."""
    return [_Drift(*record[1:]) for record in _log_records(audio) if record[0] == DRIFT_LINE]


async def test_the_switch_off_emits_the_same_bytes_on_every_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frozen flag-off flow is the old one: no bus, no overlay, the same bytes."""
    runs: list[bytes] = []
    emitted: list[list[int]] = []
    pcm: dict[str, bytes] = {}
    for _ in range(3):
        audio, queue, mass, pcm, track, _clip, _after = _scenario(monkeypatch, program_bus=False)
        output, chunk_sizes = await _run(audio, queue, track)
        runs.append(output)
        emitted.append(chunk_sizes)
        # the experiment was asked for and answered no, at least once per session
        cast("Any", mass.streams.get_config_value).assert_any_call(
            CONF_STREAM_PROGRAM_BUS, False, return_type=bool
        )
        mass.streams.register_program_bus.assert_not_called()
        mass.streams.release_program_bus.assert_not_called()
        # and nothing of the bus ran: no session, no program, no lead
        assert _program_bus_lines(audio) == []

    # the flow has the length of its items minus the overlap the mix replaces
    assert len(runs[0]) == FLOW_SECONDS * SECOND * FRAME_SIZE
    # the track before the clip plays untouched, and the track after it from its own start
    assert runs[0][: TRACK_SECONDS * SECOND * FRAME_SIZE] == pcm["item-1"]
    tail = int((AFTER_SECONDS - POST_SECONDS - VOICE_OVER_RAMP) * SECOND) * FRAME_SIZE
    assert runs[0][-tail:] == pcm["item-3"][-tail:]
    # frozen: any change to the flag-off path changes this digest
    assert hashlib.sha256(runs[0]).hexdigest() == FLAG_OFF_SHA256
    # and every run is the same run, down to the sizes it went out in
    assert all(run == runs[0] for run in runs[1:])
    assert all(chunk_sizes == emitted[0] for chunk_sizes in emitted[1:])


async def test_the_switch_off_with_crossfade_emits_the_same_bytes_on_every_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The frozen flag-off flow on a crossfading queue: the lead is a boundary voice over.

    Nothing of the bus runs here either, but the boundary paths are the ones this branch
    touched: the clip's head is the fade-in the mixer builds over the outgoing track's
    tail, that tail is held back for the whole lead ceiling instead of one crossfade, and
    the incoming track is prefetched and handed over to cover the wait.
    """
    runs: list[bytes] = []
    emitted: list[list[int]] = []
    pcm: dict[str, bytes] = {}
    for _ in range(2):
        audio, queue, mass, pcm, track, _clip, _after = _scenario(
            monkeypatch, program_bus=False, crossfade_mode=CrossfadeMode.STANDARD_CROSSFADE
        )
        output, chunk_sizes = await _run(audio, queue, track)
        runs.append(output)
        emitted.append(chunk_sizes)
        cast("Any", mass.streams.get_config_value).assert_any_call(
            CONF_STREAM_PROGRAM_BUS, False, return_type=bool
        )
        mass.streams.register_program_bus.assert_not_called()
        mass.streams.release_program_bus.assert_not_called()
        assert _program_bus_lines(audio) == []
        # the clip is an item of the queue here, read once from its own start
        flow_log = mass.player_queues.queue_data.return_value.flow_mode_stream_log
        assert [entry.queue_item_id for entry in flow_log] == ["item-1", "item-2", "item-3"]
        # its head is the boundary's voice over, and its own tail the next boundary's
        builds = cast("Any", audio.smart_fades_mixer.build).await_args_list
        assert [(call.kwargs["mode"], call.kwargs["lead"]) for call in builds] == [
            (CrossfadeMode.VOICE_OVER, True),
            (CrossfadeMode.VOICE_OVER, False),
        ]
        # the track's tail is held back for the whole lead the clip may reach over, not for
        # one standard crossfade; what the boundary's wait played out first is not in the mix
        held_seconds = len(builds[0].kwargs["fade_out_data"]) / (SECOND * FRAME_SIZE)
        assert STANDARD_CROSSFADE_DURATION < held_seconds <= LEAD_HOLD_CEILING
        # the track after the clip was prefetched while the clip held its tail, and handed
        # over to the boundary rather than decoded twice
        lines = [record[0] for record in _log_records(audio) if isinstance(record[0], str)]
        assert lines.count("Prefetching %.0f seconds of %s while the tail of %s is held back") == 1
        in_hand = [
            record
            for record in _log_records(audio)
            if isinstance(record[0], str) and "of its overlap in hand" in record[0]
        ]
        # args: format, name, in hand, prefetched, build, resident - the handover's audio
        assert [record[3] for record in in_hand][-1] > 0

    # frozen: any change to the flag-off crossfading path changes this digest
    assert hashlib.sha256(runs[0]).hexdigest() == FLAG_OFF_CROSSFADE_SHA256
    # the track is still heard whole and the track after the clip from its own start
    assert runs[0][: TRACK_SECONDS * SECOND * FRAME_SIZE] == pcm["item-1"]
    assert runs[0][-AFTER_SECONDS * SECOND * FRAME_SIZE :] == pcm["item-3"]
    # and every run is the same run, down to the sizes it went out in
    assert all(run == runs[0] for run in runs[1:])
    assert all(chunk_sizes == emitted[0] for chunk_sizes in emitted[1:])


async def test_the_switch_on_without_a_planned_program_emits_the_same_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A bus that has nothing to place counts frames and leaves every byte alone.

    The clip declares no lead here, so the bus has nothing to place over the track. The
    flag-off path never reads that declaration, which is why the same run is the golden
    run: the bus may count frames, but it must not touch one of them.
    """
    reference_audio, reference_queue, _reference_mass, _reference_pcm, reference_track, _, _ = (
        _scenario(monkeypatch, program_bus=False)
    )
    reference, reference_sizes = await _run(reference_audio, reference_queue, reference_track)
    audio, queue, mass, _pcm, track, _clip, _after = _scenario(
        monkeypatch, program_bus=True, planned=False
    )
    published = _publish(mass)

    output, chunk_sizes = await _run(audio, queue, track)

    assert output == reference
    assert chunk_sizes == reference_sizes
    assert hashlib.sha256(output).hexdigest() == FLAG_OFF_SHA256
    # the session did open a bus, and every frame of the emitted session went through it
    assert len(published) == 1
    bus = published[0]
    assert bus.active is False
    # the bus counts what it emitted, so the post overlap merged into the boundary is not
    # part of its frame count
    assert bus.produced_frames == len(output) // FRAME_SIZE == FLOW_SECONDS * SECOND
    # but it placed nothing: a session line, and no schedule, drift or carry after it
    assert _program_bus_lines(audio) == ["PROGRAM-BUS session %s opened for queue %s"]
    assert [
        record[0]
        for record in _log_records(audio)
        if isinstance(record[0], str) and "LEAD-OVERLAY" in record[0]
    ] == []


async def test_a_lead_reports_the_drift_from_what_its_item_really_delivered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An item that stops short of its declared duration leaves a positive drift behind."""
    # declared 40s, delivered 39s: the head is placed where the declared durations put
    # the item's end, and the last second of it has no item left to play over
    delivered = {
        "item-1": TRACK_SECONDS - 1,
        "item-2": CLIP_SECONDS,
        "item-3": AFTER_SECONDS,
    }
    audio, queue, mass, pcm, track, _clip, _after = _scenario(
        monkeypatch, program_bus=True, delivered=delivered
    )
    published = _publish(mass)

    output, _chunk_sizes = await _run(audio, queue, track)

    assert len(published) == 1
    reports = _drift_reports(audio)
    assert len(reports) == 1
    report = reports[0]
    fed_frames = len(pcm["item-1"]) // FRAME_SIZE
    # the placement is the declared end minus the head the clip declared
    assert report.label == "lead:item-2"
    assert report.declared_end == TRACK_SECONDS * SECOND
    assert report.anchor == report.declared_end - int(HEAD_SECONDS * SECOND)
    # while the item ended on exactly the frames its source served
    assert report.produced_end == fed_frames
    # the drift is that difference, computed from the fed lengths alone
    expected_drift_ms = (report.declared_end - fed_frames) / TEST_PCM_FORMAT.sample_rate * 1000
    assert expected_drift_ms == 1000.0
    assert report.drift_ms == expected_drift_ms
    # the number, the reported ends and the fed audio all say the same thing
    assert report.drift_ms == pytest.approx(
        (report.declared_end - report.produced_end) / TEST_PCM_FORMAT.sample_rate * 1000
    )
    # the head is heard over the last second the item really had ...
    head = _voice(CLIP_SECONDS)[: int(HEAD_SECONDS * SECOND) * FRAME_SIZE]
    heard_frames = report.produced_end - report.anchor
    assert output == _reference_sum(
        pcm["item-1"] + pcm["item-3"],
        head[: heard_frames * FRAME_SIZE],
        TEST_PCM_FORMAT,
        start_frame=report.anchor,
        stop_frame=report.produced_end,
    )
    assert (
        output[report.anchor * FRAME_SIZE : report.produced_end * FRAME_SIZE]
        != (
            (pcm["item-1"] + pcm["item-3"])[
                report.anchor * FRAME_SIZE : report.produced_end * FRAME_SIZE
            ]
        )
    )
    # ... while the rest of it is dropped, and reported as dropped
    assert (CANCEL_LINE, "head_outlasted", "lead:item-2") in _log_records(audio)
    # and the item after the boundary is its own audio from its first frame
    assert output[report.produced_end * FRAME_SIZE :] == pcm["item-3"]


async def test_the_clip_the_guard_kept_carries_its_lead_across_the_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The one clip item the bus still meets: the one the guard kept when the switch came on.

    A queue planned with the experiment off carries the DJ's break as an item, and a takeover
    only takes out the clips the player does not own yet, so such a clip can still reach the
    bus. Its head is placed over the outgoing item's tail and its declared post carries the
    rest of it into the boundary, so the clip is no item of this flow: the flow never opens
    it and the play log has no entry for it.
    """
    audio, queue, mass, pcm, track, clip, after = _scenario(monkeypatch, program_bus=True)
    published = _publish(mass)

    output, _chunk_sizes = await _run(audio, queue, track)

    assert len(published) == 1
    # the lead is the bus's, anchored where the declared durations put the track's end
    reports = _drift_reports(audio)
    assert [report.label for report in reports] == [f"lead:{clip.queue_item_id}"]
    assert reports[0].anchor == reports[0].declared_end - int(HEAD_SECONDS * SECOND)
    # the rest of the clip is summed into the boundary instead of being served
    carried = [
        record
        for record in _log_records(audio)
        if isinstance(record[0], str) and "LEAD-OVERLAY carry" in record[0]
    ]
    assert len(carried) == 1
    # the clip is no entry of the play log, so the queue never reports it as what plays
    flow_log = mass.player_queues.queue_data.return_value.flow_mode_stream_log
    assert [entry.queue_item_id for entry in flow_log] == [
        track.queue_item_id,
        after.queue_item_id,
    ]
    # the track plays untouched up to the frame the head starts on, and the item after the
    # boundary from its own start
    assert (
        output[: reports[0].anchor * FRAME_SIZE] == pcm["item-1"][: reports[0].anchor * FRAME_SIZE]
    )
    assert output[-AFTER_SECONDS * SECOND * FRAME_SIZE :] == pcm["item-3"]

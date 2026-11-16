"""Tests for the flow session's program bus: scheduled programs, and the lead as an anchor."""

from __future__ import annotations

import asyncio
import struct
import time
from collections import deque
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request
from music_assistant_models.enums import (
    ContentType,
    CrossfadeMode,
    MediaType,
    PlaybackState,
    StreamType,
    VolumeNormalizationMode,
)
from music_assistant_models.errors import QueueEmpty
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.queue_item import QueueItem
from music_assistant_models.streamdetails import StreamDetails, TailOverlap

from music_assistant.constants import FLOW_MODE_SAMPLE_RATE_SMART
from music_assistant.controllers.player_queues.controller import PlayerQueuesController
from music_assistant.controllers.streams.audio import (
    FlowProgramBus,
    StreamsAudio,
    _IncomingFadePrefetcher,
    _ProgramLead,
)
from music_assistant.controllers.streams.constants import (
    CONF_STREAM_PROGRAM_BUS,
    LEAD_HOLD_CEILING,
    LEAD_MIN_LEAD_SECONDS,
    MAX_LEAD_OVERLAP,
    PRE_OVERLAP_ATTR,
    PRE_OVERLAP_FROM_ATTR,
    VOICE_OVER_RAMP,
)
from music_assistant.controllers.streams.controller import StreamsController
from music_assistant.controllers.streams.program_ledger import ProgramLedger
from music_assistant.controllers.streams.program_schedule import (
    ProgramKind,
    ProgramPriority,
    ProgramRequest,
    ScheduleReason,
    place_program,
)
from music_assistant.controllers.streams.smart_fades.filters import (
    MIX_CEILING_DB,
    VOICE_OVER_DUCK_DEPTH,
)
from tests.controllers.streams.test_crossfade_transition import (
    STANDARD_CROSSFADE_DURATION,
    TEST_PCM_FORMAT,
    _flow_audio,
    _queue_item,
)
from tests.controllers.streams.test_flow_format import _make_player, _make_streams_audio
from tests.controllers.streams.test_flow_program_bus_output import _reference_sum

SAMPLE_SIZE = TEST_PCM_FORMAT.pcm_sample_size
SECOND = TEST_PCM_FORMAT.sample_rate
LEAD = 2.0
LEAD_CLIP_SECONDS = 4
LEAD_TRACK_SECONDS = 40
NON_MIXABLE_PCM_FORMAT = AudioFormat(
    content_type=ContentType.PCM_S16BE,
    sample_rate=TEST_PCM_FORMAT.sample_rate,
    bit_depth=16,
    channels=2,
)


def _bus() -> FlowProgramBus:
    """Build a bus on the test PCM format."""
    return FlowProgramBus(TEST_PCM_FORMAT, cast("Any", MagicMock()))


def _music(seconds: float, level: int = 100) -> bytes:
    """Build music at the given level for the given number of seconds."""
    return struct.pack("<hh", level, level) * int(seconds * SECOND)


def _voice(seconds: float, level: int = 8) -> bytes:
    """Build a voice at the given level for the given number of seconds."""
    return struct.pack("<hh", level, level) * int(seconds * SECOND)


def _levels(pcm: bytes) -> list[int]:
    """Return the first channel's samples of a PCM chunk."""
    return list(struct.unpack(f"<{len(pcm) // 2}h", pcm))[::2]


async def test_the_bus_passes_bytes_through_untouched_without_a_program() -> None:
    """A session without a due program hands every chunk back exactly as it came in."""
    bus = _bus()
    chunks = [bytes([index, index + 1, index + 2, index + 3]) * 100 for index in range(4)]

    output = [await bus.feed(chunk) for chunk in chunks]

    assert all(result is chunk for result, chunk in zip(output, chunks, strict=True))
    assert bus.produced_frames == sum(len(chunk) for chunk in chunks) // 4
    assert not bus.active


async def test_a_program_is_mixed_from_its_anchor_on_without_changing_the_length() -> None:
    """Everything before the anchor is untouched, everything inside is ducked and summed."""
    bus = _bus()
    anchor = bus.schedule(_voice(2.0), int(1.0 * SECOND), label="test")
    assert anchor == SECOND

    output = [await bus.feed(_music(1.0)) for _ in range(5)]

    # the length and the chunking are exactly the flow's own
    assert [len(chunk) for chunk in output] == [len(_music(1.0))] * 5
    # the second before the anchor is untouched, the two seconds under the voice are not
    assert output[0] == _music(1.0)
    assert output[1] != _music(1.0)
    assert output[2] != _music(1.0)
    mixed = _levels(output[1])
    # the music is pulled down over the ramp, then ducked with the voice summed on top
    assert mixed[0] > int(100 * (1 - VOICE_OVER_DUCK_DEPTH)) + 8
    ducked_from = int(VOICE_OVER_RAMP * SECOND)
    assert mixed[ducked_from] == int(100 * (1 - VOICE_OVER_DUCK_DEPTH)) + 8
    assert max(mixed[ducked_from:]) == int(100 * (1 - VOICE_OVER_DUCK_DEPTH)) + 8
    # and once the voice has ramped out, the music is untouched again
    assert output[4] == _music(1.0)
    # the tail ramp makes the program outlast its voice, then it falls out of the bus
    assert not bus.active


async def test_a_program_is_dropped_when_its_window_already_passed() -> None:
    """A program that would have played entirely in the past can not be placed."""
    bus = _bus()
    await bus.feed(_music(3.0))

    assert bus.schedule(_voice(1.0), int(0.5 * SECOND), label="late") is None
    assert not bus.active


async def test_a_program_that_is_only_partly_in_the_past_starts_late() -> None:
    """A program whose anchor already went by is moved to the current frame instead."""
    bus = _bus()
    await bus.feed(_music(2.5))

    anchor = bus.schedule(_voice(2.0), int(1.0 * SECOND), label="late")

    assert anchor == int(2.5 * SECOND)
    assert bus.active


def test_a_program_is_moved_past_one_that_is_already_scheduled() -> None:
    """A second program never ducks on top of the first: it starts where that one ends."""
    bus = _bus()

    first = bus.schedule(_voice(2.0), SECOND, label="first")
    second = bus.schedule(_voice(1.0), int(1.5 * SECOND), label="second")

    assert first == SECOND
    # the first program's window includes its ramp back to the music
    first_end = first + int((2.0 + VOICE_OVER_RAMP) * SECOND)
    assert second == first_end
    # a program inside the second one is moved past it as well
    second_end = second + int((1.0 + VOICE_OVER_RAMP) * SECOND)
    assert bus.schedule(_voice(1.0), first_end, label="third") == second_end
    # a program clear of the scheduled ones is left where it asked to be
    clear = 10 * SECOND
    assert bus.schedule(_voice(1.0), clear, label="fourth") == clear


def test_a_program_pushed_past_its_stop_window_is_dropped() -> None:
    """A program that only fits before a scheduled one can not be placed inside its own item."""
    bus = _bus()

    assert bus.schedule(_voice(2.0), SECOND, label="first") == SECOND

    assert (
        bus.schedule(
            _voice(2.0),
            int(1.5 * SECOND),
            label="crowded",
            stop_frame=int(2.0 * SECOND),
        )
        is None
    )


async def test_cancel_all_drops_a_program_at_once_and_leaves_the_audio_alone() -> None:
    """Cancelling mid-program returns to the untouched bytes and leaves no program behind."""
    bus = _bus()
    bus.schedule(_voice(2.0), int(1.0 * SECOND), label="test")
    await bus.feed(_music(1.5))

    bus.cancel_all("test stop")

    assert not bus.active
    assert await bus.feed(_music(1.0)) == _music(1.0)


async def test_cancel_all_leaves_no_task_behind() -> None:
    """Cancelling a program does not leave work of its own running."""

    async def _run() -> bytes:
        bus = _bus()
        bus.schedule(_voice(2.0), 0, label="test")
        mixed = await bus.feed(_music(2.0))
        bus.cancel_all("done")
        return mixed

    before = asyncio.all_tasks()
    await _run()

    assert asyncio.all_tasks() == before


def _lead_clip(outgoing_item_id: str = "item-1") -> SimpleNamespace:
    """Build a DJ clip that starts over the tail of the given queue item."""
    clip = _queue_item("item-2", "Clip", duration=LEAD_CLIP_SECONDS)
    clip.media_type = MediaType.SOUND_EFFECT
    clip.streamdetails.tail_overlap = None
    clip.extra_attributes[PRE_OVERLAP_ATTR] = LEAD
    clip.extra_attributes[PRE_OVERLAP_FROM_ATTR] = outgoing_item_id
    return clip


def _install_lead_item_streams(
    monkeypatch: pytest.MonkeyPatch, audio: StreamsAudio
) -> list[tuple[str, float]]:
    """Serve each item for its duration, recording where every stream of it was opened."""
    opened: list[tuple[str, float]] = []

    async def _item_stream(
        queue_item: SimpleNamespace, *_args: object, **kwargs: object
    ) -> AsyncGenerator[bytes]:
        item_id = queue_item.queue_item_id
        opened.append((item_id, cast("float", kwargs.get("seek_position", 0))))
        marker = 0x10 if item_id == "item-1" else 0x40
        remaining = queue_item.streamdetails.duration - int(
            cast("float", kwargs.get("seek_position", 0))
        )
        for _ in range(remaining):
            yield bytes([marker]) * SAMPLE_SIZE
            await asyncio.sleep(0)

    monkeypatch.setattr(audio, "get_queue_item_stream", _item_stream)
    return opened


def _lead_flow(
    monkeypatch: pytest.MonkeyPatch,
    *,
    program_bus: bool,
    crossfade_mode: CrossfadeMode = CrossfadeMode.STANDARD_CROSSFADE,
) -> tuple[StreamsAudio, SimpleNamespace, MagicMock, list[tuple[str, float]], SimpleNamespace]:
    """Build a flow whose next item is a clip that starts over the running track's tail."""
    track = _queue_item("item-1", "Track", duration=LEAD_TRACK_SECONDS)
    clip = _lead_clip()
    audio, queue, mass = _flow_audio(
        monkeypatch,
        next_item=clip,
        load_next=[clip, QueueEmpty],
        crossfade_mode=crossfade_mode,
        program_bus=program_bus,
        # the clip starts over the outgoing tail, so its tail is held for it
        lead_hold=True,
    )
    # the real rule decides whether the declared lead applies
    del audio.crossfade_allowed
    opened = _install_lead_item_streams(monkeypatch, audio)
    return audio, queue, mass, opened, track


async def _drain(stream: AsyncGenerator[bytes]) -> bytes:
    """Collect a flow stream, giving the loop a turn after every chunk."""
    output = bytearray()
    async for chunk in stream:
        output.extend(chunk)
        await asyncio.sleep(0)
    return bytes(output)


def _flow(audio: StreamsAudio, queue: SimpleNamespace, start: SimpleNamespace) -> Any:
    """Open the flow stream for the queue, starting at the given item."""
    return audio.get_queue_flow_stream(
        cast("Any", queue),
        cast("Any", start),
        TEST_PCM_FORMAT,
        session_id="session-1",
    )


async def test_the_lead_is_mixed_at_the_boundary_when_the_program_bus_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    With the experiment off the clip's head is the boundary's fade-in and the clip stays an item.

    The declared lead is not dropped: the flow builds the same voice over it builds for a
    crossfade, with the head as its fade-in, so the track's own bytes stay untouched in the
    stream while the clip is still served as the item it is.
    """
    audio, queue, mass, opened, track = _lead_flow(monkeypatch, program_bus=False)

    output = await _drain(_flow(audio, queue, track))

    track_bytes = int(LEAD_TRACK_SECONDS * SAMPLE_SIZE)
    # the track's stream is untouched: nothing was taken out of it for the head
    assert output[:track_bytes] == bytes([0x10]) * track_bytes
    # the lead is mixed all the same, as the boundary's voice over fade-in
    build = cast("Any", audio.smart_fades_mixer.build).await_args
    assert build is not None
    assert build.kwargs["mode"] == CrossfadeMode.VOICE_OVER
    assert build.kwargs["lead"] is True
    assert build.kwargs["fade_in_streamdetails"] is _clip_of(mass).streamdetails
    # and the clip is an item of the queue, read once from its own start
    flow_log = mass.player_queues.queue_data.return_value.flow_mode_stream_log
    assert [entry.queue_item_id for entry in flow_log] == ["item-1", "item-2"]
    assert [entry for entry in opened if entry[0] == "item-2"] == [("item-2", 0)]
    # no bus was opened, let alone mixed into
    cast("Any", mass.streams.register_program_bus).assert_not_called()
    cast("Any", mass.streams.get_config_value).assert_any_call(
        CONF_STREAM_PROGRAM_BUS, False, return_type=bool
    )


async def test_the_lead_is_mixed_over_the_tail_when_the_program_bus_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the experiment on the clip's head is placed on the bus at the track's end."""
    audio, queue, _mass, opened, track = _lead_flow(monkeypatch, program_bus=True)

    output = await _drain(_flow(audio, queue, track))

    track_bytes = int(LEAD_TRACK_SECONDS * SAMPLE_SIZE)
    head_bytes = int(LEAD * SAMPLE_SIZE)
    # the track keeps its length; only its last lead seconds carry the voice
    assert output[: track_bytes - head_bytes] == bytes([0x10]) * (track_bytes - head_bytes)
    tail = output[track_bytes - head_bytes : track_bytes]
    assert tail != bytes([0x10]) * len(tail)
    assert max(struct.unpack(f"<{len(tail) // 2}h", tail)) > 0x1010
    # the head is read once, out of the stream the clip itself continues on
    assert [entry for entry in opened if entry[0] == "item-2"] == [("item-2", 0)]
    # and the boundary into the clip adds nothing on top of it
    cast("Any", audio.smart_fades_mixer.build).assert_not_awaited()


@pytest.mark.parametrize(
    "crossfade_mode",
    [CrossfadeMode.STANDARD_CROSSFADE, CrossfadeMode.DISABLED],
    ids=["crossfade on", "crossfade off"],
)
async def test_a_planned_lead_is_mixed_whatever_the_queue_crossfade_says(
    monkeypatch: pytest.MonkeyPatch, crossfade_mode: CrossfadeMode
) -> None:
    """
    A declared lead is the clip's own transition, so switching the crossfade off keeps it.

    Nothing of the track is held back for a crossfade without one, but a lead is placed on
    the bus as the track streams, so it is mixed all the same: the plan is never silently
    dropped because the queue plays without a crossfade.
    """
    audio, queue, mass, opened, track = _lead_flow(
        monkeypatch, program_bus=True, crossfade_mode=crossfade_mode
    )
    logger = MagicMock()
    audio.logger = logger

    output = await _drain(_flow(audio, queue, track))

    track_bytes = int(LEAD_TRACK_SECONDS * SAMPLE_SIZE)
    head_bytes = int(LEAD * SAMPLE_SIZE)
    # the track keeps its length, and only its last lead seconds carry the voice
    assert output[: track_bytes - head_bytes] == bytes([0x10]) * (track_bytes - head_bytes)
    tail = output[track_bytes - head_bytes : track_bytes]
    assert tail != bytes([0x10]) * len(tail)
    assert max(struct.unpack(f"<{len(tail) // 2}h", tail)) > 0x1010
    # the clip continues on its own stream past the head: it is an item, not a second blend
    assert [entry for entry in opened if entry[0] == "item-2"] == [("item-2", 0)]
    flow_log = mass.player_queues.queue_data.return_value.flow_mode_stream_log
    assert [entry.queue_item_id for entry in flow_log] == ["item-1", "item-2"]
    assert [entry.seek_position for entry in flow_log] == [0, LEAD]
    cast("Any", audio.smart_fades_mixer.build).assert_not_awaited()
    # only the queue that plays without a crossfade gets the line that explains it
    reasons = [
        call.args
        for call in logger.debug.call_args_list
        if call.args[0] == "LEAD-OVERLAY lead %s mixes without a crossfade: the clip's pre is "
        "its own transition"
    ]
    assert bool(reasons) is (crossfade_mode == CrossfadeMode.DISABLED)
    assert [args[1] for args in reasons] == (
        ["lead:item-2"] if crossfade_mode == CrossfadeMode.DISABLED else []
    )


async def test_a_clips_head_plays_once_over_the_tail_it_was_planned_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clip's head plays over the track's tail and its own stream carries on after it."""
    audio, queue, mass, opened, track = _lead_flow(monkeypatch, program_bus=True)

    output = await _drain(_flow(audio, queue, track))

    # the track plays whole and the clip adds only what it did not already play
    assert len(output) == int((LEAD_TRACK_SECONDS + LEAD_CLIP_SECONDS - LEAD) * SAMPLE_SIZE)
    body = int((LEAD_CLIP_SECONDS - LEAD) * SAMPLE_SIZE)
    assert output[-body:] == bytes([0x40]) * body
    # the head was read once, out of the stream the clip itself continues on
    assert [entry for entry in opened if entry[0] == "item-2"] == [("item-2", 0)]
    # and the boundary into the clip adds nothing on top of it
    cast("Any", audio.smart_fades_mixer.build).assert_not_awaited()

    # the queue reports the clip from where it is really heard
    flow_log = mass.player_queues.queue_data.return_value.flow_mode_stream_log
    assert [entry.seek_position for entry in flow_log] == [0, LEAD]
    assert flow_log[1].seconds_streamed == LEAD_CLIP_SECONDS - LEAD
    assert _clip_of(mass).streamdetails.seek_position == LEAD
    assert _clip_of(mass).streamdetails.duration == LEAD_CLIP_SECONDS


def _clip_of(mass: MagicMock) -> SimpleNamespace:
    """Return the clip item the flow helper built as the queue's next item."""
    return cast("SimpleNamespace", mass.player_queues.get_next_item.return_value)


async def test_a_clips_body_is_reported_in_media_time_at_its_playback_speed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The head is taken in stream time, so the reported position scales by the clip's speed."""
    audio, queue, mass, _opened, track = _lead_flow(monkeypatch, program_bus=True)
    clip = _clip_of(mass)
    clip.extra_attributes["playback_speed"] = 2.0
    clip.streamdetails.duration = 8
    clip.duration = 8

    await _drain(_flow(audio, queue, track))

    flow_log = mass.player_queues.queue_data.return_value.flow_mode_stream_log
    assert flow_log[1].seek_position == LEAD * 2
    assert _clip_of(mass).streamdetails.seek_position == LEAD * 2


async def test_a_non_mixable_flow_leaves_the_head_to_the_boundary_fade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session that passes a source format through never takes the head out of the clip."""
    audio, queue, mass, opened, track = _lead_flow(monkeypatch, program_bus=True)

    output = await _drain(
        audio.get_queue_flow_stream(
            cast("Any", queue),
            cast("Any", track),
            NON_MIXABLE_PCM_FORMAT,
            session_id="session-1",
        )
    )

    bus = mass.streams.register_program_bus.call_args.args[1]
    assert not bus.can_mix
    assert not bus.active
    # the head stays in the clip and is played by the boundary's voice over instead
    cast("Any", audio.smart_fades_mixer.build).assert_awaited()
    assert [entry for entry in opened if entry[0] == "item-2"] == [("item-2", 0)]
    # nothing of the clip is skipped: its whole body is still there after the head
    assert len(output) == int((LEAD_TRACK_SECONDS + LEAD_CLIP_SECONDS) * SAMPLE_SIZE)


def test_a_lead_whose_window_is_taken_is_not_placed() -> None:
    """A head moved past the item's end is left to the clip instead of cut short."""
    logger = MagicMock()
    bus = FlowProgramBus(TEST_PCM_FORMAT, logger)
    # a program covers the frame the head is anchored on and runs past the item's end
    assert bus.schedule(_voice(4.0), int(9.0 * SECOND), label="announcement") is not None
    lead = _ProgramLead(
        bus,
        logger,
        _voice(LEAD),
        label="lead:clip",
        declared_anchor=10 * SECOND,
    )

    assert lead.arm() is None
    assert lead.anchor_report is None
    # only the announcement is on the bus, and the head is late for the end it was written for
    assert bus.active
    assert bus.last_refusal is ScheduleReason.TOO_LATE
    assert any(
        call.args[0] == "PROGRAM-BUS schedule label=%s dropped: its window was closed before it"
        for call in logger.debug.call_args_list
    )


def test_a_cancelled_lead_reports_a_reason_from_the_vocabulary() -> None:
    """Every reason the audio path withdraws a head with is a ScheduleReason, never free text."""
    logger = MagicMock()
    bus = FlowProgramBus(TEST_PCM_FORMAT, logger)
    vocabulary = {str(reason) for reason in ScheduleReason}

    abandoned = _ProgramLead(
        bus,
        logger,
        _voice(LEAD),
        label="lead:abandoned",
        declared_anchor=int(5 * SECOND),
    )
    assert abandoned.arm() is not None
    abandoned.abandon()

    ended = _ProgramLead(
        bus,
        logger,
        _voice(LEAD),
        label="lead:ended",
        declared_anchor=SECOND,
    )
    assert ended.arm() == SECOND
    # the item ends before the head reaches its anchor, so the head is withdrawn
    ended.finish(pending_frames=0)

    reasons = [
        call.args[1]
        for call in logger.debug.call_args_list
        if call.args[0] == "PROGRAM-BUS cancel reason=%s label=%s"
    ]
    assert len(reasons) == 2
    assert all(str(reason) in vocabulary for reason in reasons)
    assert {str(reason) for reason in reasons} == {
        str(ScheduleReason.ITEM_ABANDONED),
        str(ScheduleReason.HEAD_OUTLASTED),
    }


async def test_the_flow_session_end_cancels_with_the_flow_ended_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The session hands its programs to the ledger with the reason the design names."""
    audio, queue, mass, _opened, track = _lead_flow(monkeypatch, program_bus=True)
    logger = MagicMock()
    audio.logger = logger
    published: list[FlowProgramBus] = []

    def _register(_queue_id: str, program_bus: FlowProgramBus) -> None:
        published.append(program_bus)
        # longer than the session, so the end of the flow is what drops it
        program_bus.schedule(_voice(LEAD_TRACK_SECONDS + 60), 0, label="announcement")

    mass.streams.register_program_bus.side_effect = _register

    await _drain(_flow(audio, queue, track))

    reasons = [
        call.args[1]
        for call in logger.debug.call_args_list
        if str(call.args[0]).startswith("PROGRAM-BUS cancel reason=")
    ]
    assert reasons, published[0]
    assert str(ScheduleReason.FLOW_ENDED) in {str(reason) for reason in reasons}
    assert {str(reason) for reason in reasons} <= {str(reason) for reason in ScheduleReason}


async def test_the_program_bus_is_read_once_per_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The switch is read when the flow session starts, not for every chunk."""
    audio, queue, mass, _opened, track = _lead_flow(monkeypatch, program_bus=True)

    await _drain(_flow(audio, queue, track))

    assert mass.streams.get_config_value.call_count == 1


async def test_the_flow_session_publishes_and_withdraws_its_program_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A running session is reachable for overlays and gone once it ends."""
    audio, queue, mass, _opened, track = _lead_flow(monkeypatch, program_bus=True)

    await _drain(_flow(audio, queue, track))

    published = mass.streams.register_program_bus.call_args.args
    assert published[0] == "queue-1"
    mass.streams.release_program_bus.assert_called_once_with("queue-1", published[1])


async def test_no_program_bus_is_published_when_the_switch_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the experiment off there is nothing an overlay could be mixed into."""
    audio, queue, mass, _opened, track = _lead_flow(monkeypatch, program_bus=False)

    await _drain(_flow(audio, queue, track))

    mass.streams.register_program_bus.assert_not_called()
    mass.streams.release_program_bus.assert_not_called()


async def test_an_abandoned_flow_session_still_withdraws_its_program_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session the player closes early leaves no bus reachable behind."""
    audio, queue, mass, _opened, track = _lead_flow(monkeypatch, program_bus=True)
    stream = _flow(audio, queue, track)
    assert await anext(stream)

    await stream.aclose()

    published = mass.streams.register_program_bus.call_args.args
    mass.streams.release_program_bus.assert_called_once_with("queue-1", published[1])


async def test_a_program_still_playing_is_dropped_when_the_session_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The session ends the programs it still had to play instead of leaving them behind."""
    audio, queue, mass, _opened, track = _lead_flow(monkeypatch, program_bus=True)
    published: list[FlowProgramBus] = []

    def _register(_queue_id: str, program_bus: FlowProgramBus) -> None:
        published.append(program_bus)
        program_bus.schedule(_voice(LEAD_TRACK_SECONDS), 0, label="announcement")

    mass.streams.register_program_bus.side_effect = _register

    await _drain(_flow(audio, queue, track))

    assert not published[0].active


async def test_the_voice_ramps_back_to_the_music_after_the_program() -> None:
    """The ducked music ramps up over the program's tail instead of jumping back."""
    bus = _bus()
    bus.schedule(_voice(1.0), int(1.0 * SECOND), label="test")

    output = await bus.feed(_music(3.0))

    levels = _levels(output)
    ramp_frames = max(1, int(VOICE_OVER_RAMP * SECOND))
    voice_end = int(2.0 * SECOND)
    # the voice ends on full duck, a hair later the music is part way back ...
    assert levels[voice_end - 1] == int(100 * (1 - VOICE_OVER_DUCK_DEPTH)) + 8
    part_way = levels[voice_end + ramp_frames // 2]
    assert 100 * (1 - VOICE_OVER_DUCK_DEPTH) < part_way < 100
    # ... and once the ramp is over the music is untouched again
    assert levels[voice_end + ramp_frames + 1] == 100


async def test_the_program_bus_leaves_no_task_behind_when_the_flow_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished flow session owns no running work of the bus."""
    audio, queue, _mass, _opened, track = _lead_flow(monkeypatch, program_bus=True)
    before = asyncio.all_tasks()

    await _drain(_flow(audio, queue, track))

    assert asyncio.all_tasks() == before


async def test_sixteen_bit_big_endian_can_not_be_mixed() -> None:
    """A big-endian session is passed through instead of being mixed as little-endian."""
    bus = FlowProgramBus(NON_MIXABLE_PCM_FORMAT, cast("Any", MagicMock()))

    assert not bus.can_mix
    assert bus.schedule(_voice(1.0), 0, label="test") is None
    assert await bus.feed(_music(1.0)) == _music(1.0)


async def test_a_voice_shorter_than_the_ramp_is_fully_ducked_before_it_ends() -> None:
    """The duck-in ramp is bounded by the voice, so a short one does not step back up mid-word."""
    bus = _bus()
    voice = _voice(0.1)
    assert bus.schedule(voice, 0, label="test") is not None

    levels = _levels(await bus.feed(_music(1.0)))
    voice_end = int(0.1 * SECOND)
    ducked = int(100 * (1 - VOICE_OVER_DUCK_DEPTH)) + 8
    # the last frame of the voice is already at full duck, so its end does not step back up
    assert levels[voice_end - 1] == ducked
    assert levels[voice_end] == ducked - 8


async def test_the_overlay_keeps_its_frame_size_for_float_pcm() -> None:
    """32-bit float is mixed as well as the 16-bit the flow uses by default."""
    pcm_format = AudioFormat(
        content_type=ContentType.PCM_F32LE,
        sample_rate=8000,
        bit_depth=32,
        channels=2,
    )
    bus = FlowProgramBus(pcm_format, cast("Any", MagicMock()))
    bus.schedule(struct.pack("<ff", 0.1, 0.1) * 8000, 0, label="test")

    mixed = await bus.feed(struct.pack("<ff", 0.5, 0.5) * 8000)

    # past the duck-in ramp the music is at full duck, with the voice summed on top
    offset = int(VOICE_OVER_RAMP * pcm_format.sample_rate) * 8
    samples = struct.unpack("<f", mixed[offset : offset + 4])
    assert samples[0] == pytest.approx(0.5 * (1 - VOICE_OVER_DUCK_DEPTH) + 0.1, abs=1e-6)


async def test_a_lead_shorter_than_the_measured_minimum_is_not_placed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lead too short to reach the player in time is left as a plain cut."""
    track = _queue_item("item-1", "Track", duration=LEAD_TRACK_SECONDS)
    clip = _lead_clip()
    clip.extra_attributes[PRE_OVERLAP_ATTR] = 1.0
    audio, queue, _mass = _flow_audio(
        monkeypatch,
        next_item=clip,
        load_next=[clip, QueueEmpty],
        program_bus=True,
    )
    del audio.crossfade_allowed
    _install_lead_item_streams(monkeypatch, audio)

    output = await _drain(_flow(audio, queue, track))

    track_bytes = int(LEAD_TRACK_SECONDS * SAMPLE_SIZE)
    assert output[:track_bytes] == bytes([0x10]) * track_bytes


async def test_the_music_is_pulled_down_over_the_duck_ramp() -> None:
    """The music fades in to the duck over the same ramp the boundary's voice-over uses."""
    bus = _bus()
    bus.schedule(_voice(1.0), 0, label="test")

    levels = _levels(await bus.feed(_music(2.0)))

    full = 100
    ducked = int(100 * (1 - VOICE_OVER_DUCK_DEPTH)) + 8
    ramp = int(VOICE_OVER_RAMP * SECOND)
    # the first frame is still (nearly) at full level, the ramp ends on the ducked level
    assert ducked < levels[ramp // 2] < levels[0] <= full + 8
    assert levels[ramp] == ducked
    assert levels[ramp + 1] == ducked


async def test_a_full_scale_voice_over_full_scale_music_does_not_clip() -> None:
    """Summing a full-scale voice over full-scale music saturates below full scale."""
    pcm_format = AudioFormat(
        content_type=ContentType.PCM_F32LE,
        sample_rate=8000,
        bit_depth=32,
        channels=2,
    )
    bus = FlowProgramBus(pcm_format, cast("Any", MagicMock()))
    music = struct.pack("<ff", 1.0, 1.0) * 8000
    bus.schedule(struct.pack("<ff", 1.0, 1.0) * 8000, 0, label="test")

    mixed = await bus.feed(music)

    samples = struct.unpack(f"<{len(mixed) // 4}f", mixed)
    # not a single sample leaves full scale, so nothing wraps or turns into a NaN
    assert max(samples) <= 1.0
    assert min(samples) >= -1.0
    # the sum is held at the ceiling the boundary's voice-over limiter uses
    assert max(samples) == pytest.approx(10 ** (MIX_CEILING_DB / 20), abs=1e-6)
    assert len(mixed) == len(music)


async def test_a_full_scale_voice_over_full_scale_music_saturates_in_sixteen_bit() -> None:
    """The 16-bit mix saturates instead of wrapping around at full scale."""
    bus = _bus()
    music = struct.pack("<hh", 32767, 32767) * SECOND
    bus.schedule(struct.pack("<hh", 32767, 32767) * SECOND, 0, label="test")

    mixed = await bus.feed(music)

    levels = list(struct.unpack(f"<{len(mixed) // 2}h", mixed))
    assert max(levels) == 32767
    assert min(levels) >= -32768
    assert len(mixed) == len(music)


def _patterned(seconds: float, pcm_format: AudioFormat, *, base: float, step: float) -> bytes:
    """Build PCM whose level changes every frame and reaches past the mix ceiling."""
    frames = int(seconds * pcm_format.sample_rate)
    levels = base + (np.arange(frames) % 97) * step
    dtype = np.float32 if pcm_format.content_type == ContentType.PCM_F32LE else np.int16
    return np.repeat(levels.astype(dtype), pcm_format.channels).tobytes()


@pytest.mark.parametrize(
    "pcm_format",
    [
        TEST_PCM_FORMAT,
        AudioFormat(
            content_type=ContentType.PCM_F32LE,
            sample_rate=8000,
            bit_depth=32,
            channels=2,
        ),
    ],
    ids=["s16", "f32"],
)
async def test_the_mixed_chunk_is_the_frame_by_frame_sum_of_before(
    pcm_format: AudioFormat,
) -> None:
    """
    The mix emits the bytes the frame-by-frame sum emitted, ducking and ceiling included.

    The reference is the formula the bus mixed with one frame at a time: the music is ducked
    over the ramps, the program is summed on top of it and the sum is held at the ceiling.
    A bulk mix that reorders a multiplication, rounds a gain at a different place or clamps
    before the sum shows up here as a changed byte, in both PCM formats the bus mixes.
    """
    float_pcm = pcm_format.content_type == ContentType.PCM_F32LE
    music = _patterned(
        3.0, pcm_format, base=0.5 if float_pcm else 20000, step=0.01 if float_pcm else 100
    )
    voice = _patterned(
        2.0, pcm_format, base=0.4 if float_pcm else 18000, step=0.02 if float_pcm else 200
    )
    bus = FlowProgramBus(pcm_format, cast("Any", MagicMock()))
    anchor = bus.schedule(voice, int(1.0 * pcm_format.sample_rate), label="test")
    assert anchor is not None

    mixed = await bus.feed(music)

    assert mixed == _reference_sum(music, voice, pcm_format, start_frame=anchor)
    # the same frames, the same length: the mix only changes what is inside them
    assert len(mixed) == len(music)


async def test_a_long_chunk_is_mixed_in_slices_and_lets_the_loop_run_between_them() -> None:
    """A second of audio is not mixed in one go: the event loop runs while the bus works."""
    bus = _bus()
    assert bus.schedule(_voice(1.0), 0, label="test") is not None
    produced: list[int] = []

    async def _watch() -> None:
        while True:
            produced.append(bus.produced_frames)
            await asyncio.sleep(0)

    watcher = asyncio.create_task(_watch())
    await asyncio.sleep(0)
    mixed = await bus.feed(_music(1.0))
    watcher.cancel()
    await asyncio.gather(watcher, return_exceptions=True)

    frames = len(mixed) // 4
    assert frames == SECOND
    # the bus reported frames between its slices, so the loop ran while it was still mixing
    assert any(0 < counted < frames for counted in produced)
    # in slices of about a tenth of a second, not one frame and not the whole chunk
    assert max(produced) >= frames // 2
    assert len([counted for counted in produced if counted]) >= 5


async def test_a_second_of_mixed_audio_costs_a_fraction_of_a_chunk() -> None:
    """The mix stays far below the 63 ms the frame-by-frame loop needed for one second."""
    bus = _bus()
    assert bus.schedule(_voice(1.0), 0, label="test") is not None
    music = _music(1.0)

    timings = []
    for _ in range(3):
        start = time.perf_counter()
        await bus.feed(music)
        timings.append(time.perf_counter() - start)
        bus.produced_frames = 0
        assert bus.schedule(_voice(1.0), 0, label="test") is not None

    assert min(timings) < 0.025


def _capture_leads(monkeypatch: pytest.MonkeyPatch, audio: StreamsAudio) -> list[_ProgramLead]:
    """Record every lead the flow arms, so its anchor report can be read afterwards."""
    leads: list[_ProgramLead] = []
    schedule_lead = audio._schedule_program_lead

    async def _capture(*args: Any, **kwargs: Any) -> _ProgramLead | None:
        lead = await schedule_lead(*args, **kwargs)
        if lead is not None:
            leads.append(lead)
        return lead

    monkeypatch.setattr(audio, "_schedule_program_lead", _capture)
    return leads


DRIFT_REAL_SECONDS = LEAD_TRACK_SECONDS + 0.219


async def test_the_lead_never_holds_the_item_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A head is placed on the bus only: the item runs on to its last frame, without a gap."""
    audio, queue, mass, _opened, track = _lead_flow(monkeypatch, program_bus=True)
    leads = _capture_leads(monkeypatch, audio)
    published: list[FlowProgramBus] = []
    mass.streams.register_program_bus.side_effect = lambda _queue_id, bus: published.append(bus)
    frame_size = SAMPLE_SIZE // SECOND
    track_frames = LEAD_TRACK_SECONDS * SECOND
    track_bytes = track_frames * frame_size
    produced_at_end: dict[str, int] = {}

    async def _item_stream(
        queue_item: SimpleNamespace, *_args: object, **kwargs: object
    ) -> AsyncGenerator[bytes]:
        marker = 0x10 if queue_item.queue_item_id == "item-1" else 0x40
        remaining = queue_item.streamdetails.duration - int(
            cast("float", kwargs.get("seek_position", 0))
        )
        for _ in range(remaining):
            yield bytes([marker]) * SAMPLE_SIZE
        if queue_item.queue_item_id == "item-1":
            # the source is exhausted here; the crossfade tail may still be held, but the
            # bus itself must not keep any of the item's frames behind
            produced_at_end["item-1"] = published[0].produced_frames

    monkeypatch.setattr(audio, "get_queue_item_stream", _item_stream)

    output = await _drain(_flow(audio, queue, track))

    report = leads[-1].anchor_report
    assert report is not None
    # the bus counted what it emitted, and the head's report puts the item's end where the
    # item really ended: nothing of it was held back for the head
    # the boundary window (the crossfade widened to LEAD_HOLD_CEILING, held in whole
    # seconds) is still behind; nothing else of the item was kept back
    assert produced_at_end["item-1"] == (LEAD_TRACK_SECONDS - int(LEAD_HOLD_CEILING)) * SECOND
    assert report.produced_end == track_frames
    # ... so its last frames really are in the stream, with the head ducked over them
    assert output[track_bytes - SAMPLE_SIZE : track_bytes] != bytes([0x10]) * SAMPLE_SIZE
    assert len(output) == track_bytes + int((LEAD_CLIP_SECONDS - LEAD) * SAMPLE_SIZE)
    anchor = track_frames - int(LEAD * SECOND)
    # the head starts on the frame it was anchored at, not later
    assert report.anchor == anchor
    assert _levels(output[(anchor - 1) * frame_size : anchor * frame_size]) == [0x1010]
    assert max(_levels(output[anchor * frame_size : (anchor + 1) * frame_size])) > 0x1010
    # the queue's stream time keeps running forward over the boundary
    flow_log = mass.player_queues.queue_data.return_value.flow_mode_stream_log
    stream_time = 0.0
    elapsed: list[float] = []
    for entry in flow_log:
        stream_time += entry.seconds_streamed or 0.0
        elapsed.append(stream_time)
    assert elapsed == sorted(elapsed)
    clip_body = LEAD_CLIP_SECONDS - LEAD
    assert elapsed == pytest.approx([LEAD_TRACK_SECONDS, LEAD_TRACK_SECONDS + clip_body])


SECOND_ITEM_TRACK_SECONDS = 20


@pytest.mark.parametrize(
    ("lead_hold", "widened"),
    [(True, True), (False, False)],
    ids=["lead-claimed", "post-only"],
)
async def test_only_a_queue_that_may_carry_a_lead_holds_its_tail_wider(
    monkeypatch: pytest.MonkeyPatch, lead_hold: bool, widened: bool
) -> None:
    """The wide hold is the lead's: a queue that just mixes its boundaries keeps one fade."""
    clip = _lead_clip()
    audio, queue, _mass = _flow_audio(
        monkeypatch,
        next_item=clip,
        load_next=[clip, QueueEmpty],
        crossfade_mode=CrossfadeMode.STANDARD_CROSSFADE,
        lead_hold=lead_hold,
    )
    del audio.crossfade_allowed
    _install_lead_item_streams(monkeypatch, audio)
    track = _queue_item("item-1", "Track", duration=LEAD_TRACK_SECONDS)

    await _drain(_flow(audio, queue, track))

    builds = cast("Any", audio.smart_fades_mixer.build).await_args_list
    assert len(builds) == 1
    held_seconds = len(builds[0].kwargs["fade_out_data"]) / SAMPLE_SIZE
    if widened:
        assert STANDARD_CROSSFADE_DURATION < held_seconds <= LEAD_HOLD_CEILING
    else:
        assert held_seconds == pytest.approx(STANDARD_CROSSFADE_DURATION, abs=0.01)


async def test_a_lead_over_the_second_item_is_anchored_at_its_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lead over a later item ends where that item really ends, hold and mix included."""
    track = _queue_item("item-1", "Track", duration=SECOND_ITEM_TRACK_SECONDS)
    second = _queue_item("item-2", "Second", duration=SECOND_ITEM_TRACK_SECONDS)
    clip = _queue_item("item-3", "Clip", duration=LEAD_CLIP_SECONDS)
    clip.media_type = MediaType.SOUND_EFFECT
    clip.streamdetails.tail_overlap = None
    clip.extra_attributes[PRE_OVERLAP_ATTR] = LEAD
    clip.extra_attributes[PRE_OVERLAP_FROM_ATTR] = "item-2"
    audio, queue, mass = _flow_audio(
        monkeypatch,
        next_item=second,
        load_next=[second, clip, QueueEmpty],
        program_bus=True,
        # the clip starts over the second item's tail, so its tail is held for it
        lead_hold=True,
    )
    del audio.crossfade_allowed
    next_items = {"item-1": second, "item-2": clip}
    mass.player_queues.get_next_item.side_effect = lambda _queue_id, item_id: next_items.get(
        item_id
    )
    # the boundary into the second item merges its head into the first item's tail, the
    # way the real mixer does: a concatenating stand-in would keep those frames as media
    mixes = _summing_mix(monkeypatch, audio)
    leads = _capture_leads(monkeypatch, audio)
    opened: list[tuple[str, float]] = []
    markers = {"item-1": 0x10, "item-2": 0x20}
    head_seconds = int(LEAD)

    async def _item_stream(
        queue_item: SimpleNamespace, *_args: object, **kwargs: object
    ) -> AsyncGenerator[bytes]:
        item_id = queue_item.queue_item_id
        seek_position = int(cast("float", kwargs.get("seek_position", 0)))
        opened.append((item_id, seek_position))
        remaining = queue_item.streamdetails.duration - seek_position
        for index in range(remaining):
            if item_id == "item-3":
                # the clip's head is the part the bus plays over the tail, its body follows
                marker = 0x08 if index < head_seconds else 0x40
            else:
                marker = markers[item_id]
            yield bytes([marker]) * SAMPLE_SIZE
            await asyncio.sleep(0)

    monkeypatch.setattr(audio, "get_queue_item_stream", _item_stream)

    output = await _drain(_flow(audio, queue, track))

    frame_size = SAMPLE_SIZE // SECOND
    report = leads[-1].anchor_report
    assert report is not None
    # the two 20 s items with an 8 s crossfade end the second one at 32 s of the emitted
    # session: its 10 s predecessor tail is emitted at the boundary, and the mix takes its
    # own 8 s head out of its stream
    item_end = (SECOND_ITEM_TRACK_SECONDS * 2 - STANDARD_CROSSFADE_DURATION) * SECOND
    anchor = item_end - int(LEAD * SECOND)
    assert report.anchor == anchor
    assert report.produced_end == item_end
    assert report.declared_end == item_end
    assert report.drift_ms == pytest.approx(0, abs=1)
    # the head sits on the item's own last frames, and only there: one second before it
    # the item still plays untouched
    assert output[(anchor - 1) * frame_size : anchor * frame_size] == bytes([0x20]) * frame_size
    head = output[anchor * frame_size : item_end * frame_size]
    assert head != bytes([0x20]) * len(head)
    # the clip's body continues where the mixed head ended, so its head is heard once
    assert output[item_end * frame_size :] == bytes([0x40]) * int(
        (LEAD_CLIP_SECONDS - LEAD) * SAMPLE_SIZE
    )
    # the placement moved the head inside the session, it did not change its length: the
    # two items up to that end, then the clip's body
    assert len(output) == (item_end // SECOND + LEAD_CLIP_SECONDS - int(LEAD)) * SAMPLE_SIZE
    # the clip was read once, from its start: no second stream to replay the head
    assert [entry for entry in opened if entry[0] == "item-3"] == [("item-3", 0)]
    # the boundary really merged the two sides: the first item's tail is the mix's only side
    assert len(mixes) == 1
    # the 20 s item keeps half of itself as the boundary window
    assert len(mixes[0][0]) == (SECOND_ITEM_TRACK_SECONDS // 2) * SAMPLE_SIZE


async def test_a_declared_duration_miss_only_leaves_the_anchor_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An item longer than its declared duration keeps flowing; the head is simply placed early."""
    audio, queue, track, produced_at_end = _drift_flow(monkeypatch)
    leads = _capture_leads(monkeypatch, audio)

    output = await _drain(_flow(audio, queue, track))

    frame_size = SAMPLE_SIZE // SECOND
    real_end = int(DRIFT_REAL_SECONDS * SECOND)
    voice = int(LEAD * SECOND)
    declared_end = LEAD_TRACK_SECONDS * SECOND
    report = leads[-1].anchor_report
    assert report is not None
    # the head is placed where the declared durations put it, and the item ended later
    assert report.anchor == declared_end - voice
    assert report.produced_end == real_end
    assert report.declared_end == declared_end
    # the item produced all of its audio, so the 219 ms the durations missed are in the
    # stream; the bus only counted what it emitted before the tail was handed over
    assert produced_at_end["item-1"] <= real_end
    assert report.produced_end == real_end
    assert output[: report.anchor * frame_size] == bytes([0x10]) * (report.anchor * frame_size)
    missed = output[declared_end * frame_size : real_end * frame_size]
    assert missed == bytes([0x10]) * len(missed)
    assert output[real_end * frame_size :] == bytes([0x40]) * int(
        (LEAD_CLIP_SECONDS - LEAD) * SAMPLE_SIZE
    )
    # the residue is what the report says: the head ends where the durations said it would
    assert report.drift_ms == pytest.approx(-(real_end - declared_end) / SECOND * 1000, abs=1)


async def test_a_still_streaming_source_is_not_held_back_either(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source that keeps trickling is mixed through like any other, to its last frame."""
    audio, queue, track, produced_at_end = _drift_flow(monkeypatch)
    leads = _capture_leads(monkeypatch, audio)
    cast("Any", track.streamdetails.buffer).eof = False

    await _drain(_flow(audio, queue, track))

    real_end = int(DRIFT_REAL_SECONDS * SECOND)
    report = leads[-1].anchor_report
    assert report is not None
    assert report.produced_end == real_end
    assert produced_at_end["item-1"] <= real_end
    assert report.anchor == LEAD_TRACK_SECONDS * SECOND - int(LEAD * SECOND)


async def test_a_late_anchor_is_clamped_forward_and_reported() -> None:
    """A frame that already went by is moved to where the flow is, never waited for."""
    logger = MagicMock()
    bus = FlowProgramBus(TEST_PCM_FORMAT, logger)
    await bus.feed(_music(1.0))
    lead = _ProgramLead(
        bus,
        logger,
        _voice(1.0),
        label="lead:clip",
        declared_anchor=int(0.5 * SECOND),
    )

    assert lead.arm() == SECOND
    assert any(
        call.args[0] == "PROGRAM-BUS schedule label=%s started late by %.3fs"
        for call in logger.debug.call_args_list
    )
    # the item's own audio is not rewound: it goes through as it comes, the head on top
    mixed = await bus.feed(_music(1.0))
    lead.finish()

    assert mixed != _music(1.0)
    report = lead.anchor_report
    assert report is not None
    assert (report.anchor, report.produced_end, report.declared_end) == (
        SECOND,
        2 * SECOND,
        int(1.5 * SECOND),
    )
    # the clamp dropped the head's first half second, so only half of it played and its end
    # sits half a second before the item's end: the drift is that residue, not zero
    assert report.drift_ms == pytest.approx(-500, abs=1)


async def test_a_head_that_played_out_is_not_cancelled_as_unfinished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A head the clamp trimmed is fully heard once the frames it has left have gone by."""
    logger = MagicMock()
    bus = FlowProgramBus(TEST_PCM_FORMAT, logger)
    cancel = MagicMock(wraps=bus.cancel)
    monkeypatch.setattr(bus, "cancel", cancel)
    await bus.feed(_music(1.0))
    lead = _ProgramLead(
        bus,
        logger,
        _voice(1.0),
        label="lead:clip",
        declared_anchor=int(0.5 * SECOND),
    )
    assert lead.arm() == SECOND

    # exactly the trimmed half second is left, and the item ends with it
    mixed = await bus.feed(_music(0.5))
    lead.finish()

    assert mixed != _music(0.5)
    report = lead.anchor_report
    assert report is not None
    # the whole played head sits on the item's end: an onset at 0.5s and an end at 1.5s
    assert (report.anchor, report.produced_end, report.declared_end) == (
        SECOND,
        int(1.5 * SECOND),
        int(1.5 * SECOND),
    )
    assert report.drift_ms == pytest.approx(0, abs=1)
    # it played out on its own frames, so the bus was never asked to drop it: what is left
    # of the program is only its duck ramp, which a later chunk ends by playing on
    cancel.assert_not_called()
    assert bus.scheduled_voice_frames("lead:clip") is not None
    await bus.feed(_music(1.0))
    await bus.feed(_music(1.0))
    assert bus.active is False
    cancel.assert_not_called()


def _drift_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[StreamsAudio, SimpleNamespace, SimpleNamespace, dict[str, int]]:
    """Build a flow whose item delivers 219 ms more audio than its declared duration."""
    track = _queue_item("item-1", "Track", duration=LEAD_TRACK_SECONDS)
    clip = _lead_clip()
    audio, queue, mass = _flow_audio(
        monkeypatch, next_item=clip, load_next=[clip, QueueEmpty], program_bus=True
    )
    del audio.crossfade_allowed
    published: list[FlowProgramBus] = []
    mass.streams.register_program_bus.side_effect = lambda _queue_id, bus: published.append(bus)
    produced_at_end: dict[str, int] = {}

    async def _item_stream(
        queue_item: SimpleNamespace, *_args: object, **_kwargs: object
    ) -> AsyncGenerator[bytes]:
        # a seeked source delivers a little more than the whole-second metadata duration
        marker = 0x10 if queue_item.queue_item_id == "item-1" else 0x40
        remaining = (
            DRIFT_REAL_SECONDS
            if queue_item.queue_item_id == "item-1"
            else queue_item.streamdetails.duration
        )
        while remaining > 0:
            yield bytes([marker]) * int(min(1.0, remaining) * SAMPLE_SIZE)
            remaining -= 1.0
        if queue_item.queue_item_id == "item-1" and published:
            produced_at_end["item-1"] = published[0].produced_frames

    monkeypatch.setattr(audio, "get_queue_item_stream", _item_stream)
    return audio, queue, track, produced_at_end


async def test_a_cancelled_lead_never_sounds_over_the_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A skip drops the head from the bus: the track plays out untouched."""
    audio, queue, mass, _opened, track = _lead_flow(monkeypatch, program_bus=True)
    published: list[FlowProgramBus] = []
    mass.streams.register_program_bus.side_effect = lambda _queue_id, bus: published.append(bus)
    logger = MagicMock()
    audio.logger = logger

    output = bytearray()
    cancelled = False
    async for chunk in _flow(audio, queue, track):
        output.extend(chunk)
        if published and not cancelled:
            published[0].cancel_all("next")
            cancelled = True
        await asyncio.sleep(0)

    # the track plays untouched: its tail never got the head
    track_bytes = int(LEAD_TRACK_SECONDS * SAMPLE_SIZE)
    assert output[:track_bytes] == bytes([0x10]) * track_bytes
    assert any(
        call.args[0] == "PROGRAM-BUS cancel reason=%s" and call.args[1] == "next"
        for call in logger.debug.call_args_list
    )


async def test_a_group_flow_stream_mixes_its_program_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flow stream of a two-member group runs one bus, so both members share one mix."""
    track = _queue_item("item-1", "Track", duration=4)
    audio, queue, mass = _flow_audio(
        monkeypatch,
        next_item=None,
        load_next=[QueueEmpty],
        crossfade_mode=CrossfadeMode.DISABLED,
        program_bus=True,
    )
    del audio.crossfade_allowed
    _install_lead_item_streams(monkeypatch, audio)
    player = cast("Any", mass.players.get_player.return_value)
    player.state.group_members = ["member-a", "member-b"]
    published: list[FlowProgramBus] = []
    mass.streams.register_program_bus.side_effect = lambda _queue_id, bus: published.append(bus)

    output = bytearray()
    anchor = 0
    async for chunk in _flow(audio, queue, track):
        if published and not output:
            anchor = published[0].produced_frames
            assert published[0].schedule(_voice(1.0), anchor, label="announcement") is not None
        output.extend(chunk)
        await asyncio.sleep(0)

    # the group's session runs a single bus ...
    assert len(published) == 1
    # ... and its mix carries the program exactly once, over the frame it was placed at
    frame_size = SAMPLE_SIZE // SECOND
    marker = bytes([0x10]) * frame_size
    changed = [
        index
        for index in range(len(output) // frame_size)
        if output[index * frame_size : (index + 1) * frame_size] != marker
    ]
    assert changed[0] == anchor
    assert changed[-1] == anchor + SECOND + int(VOICE_OVER_RAMP * SECOND) - 1
    # no second copy of the program elsewhere (a frame or two can round onto the
    # untouched level inside the ramp)
    assert len(changed) >= changed[-1] - changed[0] + 1 - 2


FLOW_MATCH_INFO = {
    "session_id": "session-1",
    "queue_id": "queue-1",
    "queue_item_id": "item-1",
    "player_id": "player-1",
    "fmt": "pcm",
}


def _flow_start_item() -> QueueItem:
    """Return a track item with resolved stream details."""
    item = QueueItem(queue_id="queue-1", queue_item_id="item-1", name="Track", duration=180)
    item.streamdetails = StreamDetails(
        provider="local--1",
        item_id="item-1",
        audio_format=AudioFormat(
            content_type=ContentType.FLAC, sample_rate=44100, bit_depth=16, channels=2
        ),
        media_type=MediaType.TRACK,
        stream_type=StreamType.LOCAL_FILE,
        path="/music/item1.flac",
        duration=180,
    )
    item.streamdetails.volume_normalization_mode = VolumeNormalizationMode.DISABLED
    return item


def _flow_request_controller(*, program_bus: bool) -> MagicMock:
    """
    Return a streams controller answering the flow stream request of one track.

    :param program_bus: Whether the experimental program bus is switched on.
    """
    controller = MagicMock()
    controller.get_crossfade_mode.return_value = CrossfadeMode.DISABLED
    controller.flow_mode_required.return_value = False
    controller.boundary_mix_required.return_value = False
    controller.program_bus_enabled.return_value = program_bus
    controller._get_flow_start_item.return_value = _flow_start_item()
    queue = controller.mass.player_queues.get.return_value
    queue.overlay_enabled = False
    queue.overlay_source = None
    controller.mass.player_queues.queue_data.return_value = SimpleNamespace(session_id="session-1")
    controller.mass.player_queues.flow_stream_finished.return_value = False
    controller.mass.players.get_output_protocol.return_value = _make_player(
        supported=[(44100, 16)], flow_mode=FLOW_MODE_SAMPLE_RATE_SMART
    )
    controller.mass.players.get_output_protocol.return_value.get_config_value = MagicMock(
        return_value=False
    )
    controller.mass.config.get_raw_player_config_value.return_value = "disabled"
    controller.audio = _make_streams_audio()
    controller.audio.select_flow_pcm_format = AsyncMock(
        wraps=controller.audio.select_flow_pcm_format
    )
    controller.audio.get_output_format = AsyncMock(return_value=TEST_PCM_FORMAT)
    return controller


@pytest.mark.parametrize(
    ("program_bus", "bit_depth"),
    [(True, 32), (False, 16)],
    ids=["bus-on", "bus-off"],
)
async def test_the_flow_runs_in_float_while_the_program_bus_can_mix(
    program_bus: bool, bit_depth: int
) -> None:
    """A session that may mix a program asks the flow for float PCM, so its sum has room."""
    controller = _flow_request_controller(program_bus=program_bus)
    request = make_mocked_request("HEAD", "/flow/item.pcm", match_info=FLOW_MATCH_INFO)

    response = await StreamsController.serve_queue_flow_stream(controller, request)

    assert isinstance(response, web.StreamResponse)
    format_request = controller.audio.select_flow_pcm_format.await_args
    assert format_request is not None
    # the overlay path decides for the whole session: a depth change would restart it
    assert format_request.kwargs["overlay_active"] is program_bus
    # the flow then really runs in float (or keeps the source's 16-bit without the bus)
    output_request = controller.audio.get_output_format.await_args
    assert output_request is not None
    assert output_request.kwargs["content_bit_depth"] == bit_depth


THREE_ITEM_TRACK_SECONDS = 200


def _clip_between_tracks_flow(
    monkeypatch: pytest.MonkeyPatch, stop_after: dict[str, int]
) -> tuple[StreamsAudio, SimpleNamespace, MagicMock, SimpleNamespace, SimpleNamespace]:
    """Build a flow of track -> clip -> track, serving each item up to the given second."""
    track = _queue_item("item-1", "Track", duration=LEAD_TRACK_SECONDS)
    clip = _lead_clip()
    after = _queue_item("item-3", "After", duration=THREE_ITEM_TRACK_SECONDS)
    audio, queue, mass = _flow_audio(
        monkeypatch,
        next_item=clip,
        load_next=[clip, after, QueueEmpty],
        program_bus=True,
    )
    # the real rule decides whether the declared lead applies
    del audio.crossfade_allowed
    next_items = {"item-1": clip, "item-2": after}
    mass.player_queues.get_next_item.side_effect = lambda _queue_id, item_id: next_items.get(
        item_id
    )

    async def _item_stream(
        queue_item: SimpleNamespace, *_args: object, **kwargs: object
    ) -> AsyncGenerator[bytes]:
        remaining = stop_after[queue_item.queue_item_id] - int(
            cast("float", kwargs.get("seek_position", 0))
        )
        for _ in range(max(0, remaining)):
            yield bytes([0x10]) * SAMPLE_SIZE
            await asyncio.sleep(0)

    monkeypatch.setattr(audio, "get_queue_item_stream", _item_stream)
    return audio, queue, mass, track, after


async def test_a_track_after_a_clip_keeps_its_own_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clip's seconds belong to the clip; the track behind it keeps its length."""
    audio, queue, mass, track, after = _clip_between_tracks_flow(
        monkeypatch, {"item-1": LEAD_TRACK_SECONDS, "item-2": LEAD_CLIP_SECONDS, "item-3": 200}
    )
    clip = _clip_of(mass)

    await _drain(_flow(audio, queue, track))

    assert clip.streamdetails.duration == LEAD_CLIP_SECONDS
    assert after.streamdetails.duration == THREE_ITEM_TRACK_SECONDS
    assert after.duration == THREE_ITEM_TRACK_SECONDS
    assert after.streamdetails.seconds_streamed == THREE_ITEM_TRACK_SECONDS
    flow_log = mass.player_queues.queue_data.return_value.flow_mode_stream_log
    assert [entry.seek_position for entry in flow_log] == [0, LEAD, 0]
    assert flow_log[2].duration == THREE_ITEM_TRACK_SECONDS


async def test_a_track_after_a_clip_keeps_its_length_when_its_source_stops_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source that stops mid-item must not hand the item its short streamed length."""
    audio, queue, mass, track, after = _clip_between_tracks_flow(
        monkeypatch, {"item-1": LEAD_TRACK_SECONDS, "item-2": LEAD_CLIP_SECONDS, "item-3": 5}
    )
    clip = _clip_of(mass)

    await _drain(_flow(audio, queue, track))

    # the clip keeps its own length and the track behind it keeps its 200 seconds
    assert clip.streamdetails.duration == LEAD_CLIP_SECONDS
    assert after.streamdetails.duration == THREE_ITEM_TRACK_SECONDS
    assert after.duration == THREE_ITEM_TRACK_SECONDS
    # what really reached the player stays honest, so the queue can still advance
    assert after.streamdetails.seconds_streamed == 5
    flow_log = mass.player_queues.queue_data.return_value.flow_mode_stream_log
    assert [entry.seconds_streamed for entry in flow_log] == [
        LEAD_TRACK_SECONDS,
        LEAD_CLIP_SECONDS - LEAD,
        5,
    ]


async def test_a_prefetched_stream_is_handed_over_after_the_item_was_resolved_again() -> None:
    """The head was taken out of one stream; a re-resolve must not replay that head here."""
    prefetcher = _IncomingFadePrefetcher(cast("Any", MagicMock()), TEST_PCM_FORMAT, "session")
    clip = cast("Any", _queue_item("clip", "Transition", duration=LEAD_CLIP_SECONDS))
    taken = StreamDetails(provider="ai_radio", item_id="clip_1", audio_format=TEST_PCM_FORMAT)
    clip.streamdetails = taken

    async def _body() -> AsyncGenerator[bytes]:
        yield _voice(0.5)

    prefetcher._task = asyncio.create_task(asyncio.sleep(0))
    prefetcher._queue_item_id = "clip"
    prefetcher._seek_position = 0
    prefetcher._streamdetails = taken
    prefetcher._stream = _body()
    prefetcher._chunks = deque([_voice(0.25)])
    await prefetcher._task

    # preparing the clip's buffer resolves the same media into a new object
    clip.streamdetails = StreamDetails(
        provider="ai_radio", item_id="clip_1", audio_format=TEST_PCM_FORMAT
    )
    handed = await prefetcher.take(clip, 0)

    assert handed is not None
    assert b"".join([chunk async for chunk in handed]) == _voice(0.25) + _voice(0.5)


def test_a_taken_head_still_counts_once_the_item_is_resolved_again() -> None:
    """The queue resolves an item's details again for its buffer; the head stays that item's."""
    prefetcher = _IncomingFadePrefetcher(cast("Any", MagicMock()), TEST_PCM_FORMAT, "session")
    clip = cast("Any", _queue_item("clip", "Transition", duration=LEAD_CLIP_SECONDS))
    taken = StreamDetails(provider="ai_radio", item_id="clip_1", audio_format=TEST_PCM_FORMAT)
    clip.streamdetails = taken
    prefetcher._task = cast("Any", MagicMock())
    prefetcher._queue_item_id = "clip"
    prefetcher._head_seconds = LEAD
    prefetcher._streamdetails = taken

    assert prefetcher.head_mixed(clip) == LEAD
    # preparing the clip's buffer resolves the same media into a new object
    clip.streamdetails = StreamDetails(
        provider="ai_radio", item_id="clip_1", audio_format=TEST_PCM_FORMAT
    )
    assert prefetcher.head_mixed(clip) == LEAD
    # a stream of another item at the same queue position is not the head that was taken
    clip.streamdetails = StreamDetails(
        provider="ai_radio", item_id="clip_2", audio_format=TEST_PCM_FORMAT
    )
    assert prefetcher.head_mixed(clip) is None


COVERED_CLIP_SECONDS = 4
COVERED_HEAD = 2.0
COVERED_POST = 2.0
# the marker each item's audio carries, so the output can be read per item
ITEM_SECONDS = {"item-1": LEAD_TRACK_SECONDS, "item-2": COVERED_CLIP_SECONDS, "item-3": 200}
ITEM_MARKERS = {"item-1": 0x10, "item-2": 0x40, "item-3": 0x50}


def _covered_clip() -> SimpleNamespace:
    """Build a DJ clip whose head and tail overlap together cover all of it."""
    clip = _queue_item("item-2", "Transition", duration=COVERED_CLIP_SECONDS)
    clip.media_type = MediaType.SOUND_EFFECT
    clip.extra_attributes[PRE_OVERLAP_ATTR] = COVERED_HEAD
    clip.extra_attributes[PRE_OVERLAP_FROM_ATTR] = "item-1"
    clip.streamdetails.tail_overlap = TailOverlap(
        duration=COVERED_POST, next_queue_item_id="item-3"
    )
    return clip


def _summing_mix(monkeypatch: pytest.MonkeyPatch, audio: StreamsAudio) -> list[tuple[bytes, bytes]]:
    """
    Replace the mixer with the shape a voice over really has: one side is summed in.

    The side that carries the music keeps its own length, so the flow's length is exactly what
    it would really be, and both sides of every boundary can be read afterwards.
    """
    mixes: list[tuple[bytes, bytes]] = []

    async def _mix(
        _smart_fade: object,
        *,
        fade_in_part: bytes | AsyncGenerator[bytes],
        fade_out_part: bytes,
        **_kwargs: object,
    ) -> AsyncGenerator[bytes]:
        if isinstance(fade_in_part, bytes):
            fade_in_bytes = fade_in_part
        else:
            fade_in_bytes = b"".join([chunk async for chunk in fade_in_part])
        mixes.append((fade_out_part, fade_in_bytes))
        # the music side of the blend is the longer one: the voice is summed into it
        yield fade_out_part if len(fade_out_part) >= len(fade_in_bytes) else fade_in_bytes

    monkeypatch.setattr(audio.smart_fades_mixer, "mix", _mix)
    return mixes


def _covered_clip_flow(
    monkeypatch: pytest.MonkeyPatch, *, program_bus: bool
) -> tuple[
    StreamsAudio,
    SimpleNamespace,
    MagicMock,
    SimpleNamespace,
    SimpleNamespace,
    SimpleNamespace,
    list[Any],
]:
    """Build a flow of track -> covered clip -> track, marking each item's audio."""
    track = _queue_item("item-1", "Track", duration=LEAD_TRACK_SECONDS)
    clip = _covered_clip()
    after = _queue_item("item-3", "After", duration=ITEM_SECONDS["item-3"])
    audio, queue, mass = _flow_audio(
        monkeypatch,
        next_item=clip,
        load_next=[clip, after, QueueEmpty],
        program_bus=program_bus,
        build_result=SimpleNamespace(
            timing_info=SimpleNamespace(
                fadein_trimmed_duration=0.0,
                crossfade_duration=COVERED_POST,
                pre_crossfade_duration=0.0,
            )
        ),
    )
    mixes = _summing_mix(monkeypatch, audio)
    # the real rule decides whether a declared lead applies
    del audio.crossfade_allowed
    next_items = {"item-1": clip, "item-2": after}
    mass.player_queues.get_next_item.side_effect = lambda _queue_id, item_id: next_items.get(
        item_id
    )
    opened: list[tuple[str, float]] = []

    async def _item_stream(
        queue_item: SimpleNamespace, *_args: object, **kwargs: object
    ) -> AsyncGenerator[bytes]:
        item_id = queue_item.queue_item_id
        opened.append((item_id, cast("float", kwargs.get("seek_position", 0))))
        remaining = ITEM_SECONDS[item_id] - int(cast("float", kwargs.get("seek_position", 0)))
        for _ in range(max(0, remaining)):
            yield bytes([ITEM_MARKERS[item_id]]) * SAMPLE_SIZE
            await asyncio.sleep(0)

    monkeypatch.setattr(audio, "get_queue_item_stream", _item_stream)
    return audio, queue, mass, track, clip, after, [opened, mixes]


async def test_a_covered_clip_is_carried_over_the_boundary_without_a_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clip that its head and post cover whole is summed into the boundary, not served."""
    audio, queue, mass, track, clip, _after, captured = _covered_clip_flow(
        monkeypatch, program_bus=True
    )
    opened, mixes = captured

    output = await _drain(_flow(audio, queue, track))

    second = SAMPLE_SIZE
    track_bytes = LEAD_TRACK_SECONDS * second
    head_bytes = int(COVERED_HEAD * second)
    # the track plays whole, with the clip's head over its last seconds ...
    assert output[: track_bytes - head_bytes] == bytes([0x10]) * (track_bytes - head_bytes)
    assert output[track_bytes - head_bytes : track_bytes] != bytes([0x10]) * head_bytes
    # ... and the track after it owns every byte from the boundary on: no hole, no silence
    assert output[track_bytes:] == bytes([0x50]) * (len(output) - track_bytes)
    # the flow is exactly the two tracks long; the clip's voice is summed into them
    assert len(output) == (LEAD_TRACK_SECONDS + ITEM_SECONDS["item-3"]) * second
    # the clip was read once, as a whole, and never opened a second time as an item
    assert [entry for entry in opened if entry[0] == "item-2"] == [("item-2", 0)]
    # its remaining audio is the mixed side of the boundary into the track after it
    carried = COVERED_CLIP_SECONDS - int(COVERED_HEAD)
    assert len(mixes) == 1
    assert mixes[0][0] == bytes([0x40]) * (carried * second)
    build = cast("Any", audio.smart_fades_mixer.build).await_args
    assert build.kwargs["mode"] == CrossfadeMode.VOICE_OVER
    assert build.kwargs["lead"] is False
    # and it is no item of this flow: the play log names only the two tracks
    flow_log = mass.player_queues.queue_data.return_value.flow_mode_stream_log
    assert [entry.queue_item_id for entry in flow_log] == ["item-1", "item-3"]
    assert [entry.seconds_streamed for entry in flow_log] == [
        LEAD_TRACK_SECONDS,
        ITEM_SECONDS["item-3"],
    ]
    # the clip keeps its own length; nothing of it was written back as another item's
    assert clip.streamdetails.duration == COVERED_CLIP_SECONDS
    assert clip.duration == COVERED_CLIP_SECONDS


async def test_the_now_playing_stays_the_track_while_a_covered_clip_plays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The queue never reports a clip that is only heard over the tracks around it."""
    audio, queue, mass, track, clip, after, _captured = _covered_clip_flow(
        monkeypatch, program_bus=True
    )

    await _drain(_flow(audio, queue, track))

    data = mass.player_queues.queue_data.return_value
    data.items = [track, clip, after]
    queue.current_index = 0
    controller = PlayerQueuesController.__new__(PlayerQueuesController)
    controller._queue_data = {queue.queue_id: data}

    def _position(elapsed: float) -> tuple[int | None, float]:
        player = SimpleNamespace(
            corrected_elapsed_time=elapsed, playback_state=PlaybackState.PLAYING
        )
        return controller._get_flow_queue_stream_index(cast("Any", queue), cast("Any", player))

    # the clip's head plays over the track, and the track is what the player shows
    assert _position(LEAD_TRACK_SECONDS - COVERED_HEAD)[0] == 0
    # at the boundary the position moves on to the track after it, never to the clip
    assert _position(LEAD_TRACK_SECONDS + 0.5)[0] == 2
    assert _position(LEAD_TRACK_SECONDS + 5)[0] == 2
    # the position runs forward inside each item, and no item is ever skipped backwards
    indexes = [_position(elapsed)[0] for elapsed in (39.0, 39.5, 40.5, 41.0, 42.0)]
    assert indexes == [0, 0, 2, 2, 2]
    within_after = [_position(elapsed)[1] for elapsed in (40.5, 41.0, 42.0)]
    assert within_after == sorted(within_after)


async def test_a_covered_clip_is_served_as_an_item_when_the_program_bus_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the bus there is no head, so the clip is an item of the queue as before."""
    audio, queue, mass, track, _clip, _after, captured = _covered_clip_flow(
        monkeypatch, program_bus=False
    )
    opened, mixes = captured

    output = await _drain(_flow(audio, queue, track))

    # nothing was carried over the boundary; the clip is opened and served as its own item
    assert [entry for entry in opened if entry[0] == "item-2"] == [("item-2", 0)]
    flow_log = mass.player_queues.queue_data.return_value.flow_mode_stream_log
    assert [entry.queue_item_id for entry in flow_log] == ["item-1", "item-2", "item-3"]
    assert flow_log[1].seconds_streamed
    # both of its boundaries are mixed as they always were: its head is the fade-in over the
    # track before it, and its held tail the fade-out into the track after it
    second = SAMPLE_SIZE
    head = int(COVERED_HEAD * second)
    assert len(mixes) == 2
    assert mixes[0][1] == bytes([0x40]) * head
    assert mixes[1][0] == bytes([0x40]) * (COVERED_CLIP_SECONDS * second - head)
    # and the flow is exactly the two tracks long: the clip is summed into both boundaries
    assert len(output) == (LEAD_TRACK_SECONDS + ITEM_SECONDS["item-3"]) * second


PRE_ONLY_CLIP_SECONDS = 4
PRE_ONLY_AFTER_SECONDS = 6
PRE_ONLY_SECONDS = {
    "item-1": LEAD_TRACK_SECONDS,
    "item-2": PRE_ONLY_CLIP_SECONDS,
    "item-3": PRE_ONLY_AFTER_SECONDS,
}


def _pre_only_clip(seconds: int = PRE_ONLY_CLIP_SECONDS) -> SimpleNamespace:
    """Build a break whose head alone covers all of it: a pre without a post."""
    clip = _queue_item("item-2", "Break", duration=seconds)
    clip.media_type = MediaType.SOUND_EFFECT
    # no tail overlap: the section does not allow a post, so the head takes the whole break
    clip.streamdetails.tail_overlap = None
    clip.extra_attributes[PRE_OVERLAP_ATTR] = float(seconds)
    clip.extra_attributes[PRE_OVERLAP_FROM_ATTR] = "item-1"
    return clip


def _pre_only_clip_flow(
    monkeypatch: pytest.MonkeyPatch,
    *,
    overrun: float = 0.0,
    clip_seconds: int = PRE_ONLY_CLIP_SECONDS,
) -> tuple[StreamsAudio, SimpleNamespace, MagicMock, SimpleNamespace, SimpleNamespace, list[Any]]:
    """
    Build a flow of track -> pre-only break -> track, marking each item's audio.

    :param overrun: Seconds the break's real audio outlasts its declared duration.
    :param clip_seconds: Declared duration of the break, and the head it declares.
    """
    track = _queue_item("item-1", "Track", duration=LEAD_TRACK_SECONDS)
    clip = _pre_only_clip(clip_seconds)
    after = _queue_item("item-3", "After", duration=PRE_ONLY_AFTER_SECONDS)
    real_seconds: dict[str, float] = dict(PRE_ONLY_SECONDS)
    real_seconds["item-2"] = clip_seconds + overrun
    audio, queue, mass = _flow_audio(
        monkeypatch,
        next_item=clip,
        load_next=[clip, after, QueueEmpty],
        program_bus=True,
    )
    audio.logger = MagicMock()
    mixes = _summing_mix(monkeypatch, audio)
    # the real rule decides whether a declared lead applies
    del audio.crossfade_allowed
    next_items = {"item-1": clip, "item-2": after}
    mass.player_queues.get_next_item.side_effect = lambda _queue_id, item_id: next_items.get(
        item_id
    )
    opened: list[tuple[str, float]] = []

    async def _item_stream(
        queue_item: SimpleNamespace, *_args: object, **kwargs: object
    ) -> AsyncGenerator[bytes]:
        item_id = queue_item.queue_item_id
        seek_position = cast("float", kwargs.get("seek_position", 0))
        opened.append((item_id, seek_position))
        # the source delivers in tenths of a second, as a decoded clip really does, so a
        # declared duration that is short of the real one is representable
        remaining = round((real_seconds[item_id] - seek_position) * 10)
        for _ in range(max(0, remaining)):
            yield bytes([ITEM_MARKERS[item_id]]) * (SAMPLE_SIZE // 10)
            await asyncio.sleep(0)

    monkeypatch.setattr(audio, "get_queue_item_stream", _item_stream)
    return audio, queue, mass, track, clip, [opened, mixes]


async def test_a_pre_only_break_is_heard_over_the_track_and_never_served_as_an_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A break its head covers whole is no item either, even though it has no post.

    Nothing of its own stream is left once the head played over the track's tail, so
    serving it would hand the player an item without audio: the flow skips it, plays the
    held-back tail and moves straight on to the next track.
    """
    audio, queue, mass, track, clip, captured = _pre_only_clip_flow(monkeypatch)
    opened, mixes = captured

    output = await _drain(_flow(audio, queue, track))

    second = SAMPLE_SIZE
    track_bytes = LEAD_TRACK_SECONDS * second
    head_bytes = PRE_ONLY_CLIP_SECONDS * second
    # the track plays whole, with the break's head over its last seconds ...
    assert output[: track_bytes - head_bytes] == bytes([0x10]) * (track_bytes - head_bytes)
    assert output[track_bytes - head_bytes : track_bytes] != bytes([0x10]) * head_bytes
    # ... and the track after it owns every byte from the boundary on: no hole, no silence
    assert output[track_bytes:] == bytes([0x50]) * (len(output) - track_bytes)
    # the flow is exactly the two tracks long: the break is heard once, over the track
    assert len(output) == (LEAD_TRACK_SECONDS + PRE_ONLY_AFTER_SECONDS) * second
    # it was read once, as a whole, and never opened a second time as an item
    assert [entry for entry in opened if entry[0] == "item-2"] == [("item-2", 0)]
    # nothing of it is left to mix into the boundary after it
    assert mixes == []
    cast("Any", audio.smart_fades_mixer.build).assert_not_awaited()
    # the item it was no item of is not reported as broken or as played
    logger = cast("Any", audio.logger)
    assert not [
        call
        for call in logger.warning.call_args_list
        if isinstance(call.args[0], str) and "produced no audio data" in call.args[0]
    ]
    assert clip.streamdetails.stream_error is False
    # and the queue moved past it: the play log names only the two tracks
    flow_log = mass.player_queues.queue_data.return_value.flow_mode_stream_log
    assert [entry.queue_item_id for entry in flow_log] == ["item-1", "item-3"]
    assert [entry.seconds_streamed for entry in flow_log] == [
        LEAD_TRACK_SECONDS,
        PRE_ONLY_AFTER_SECONDS,
    ]
    assert [call.args[1] for call in mass.player_queues.load_next_queue_item.await_args_list] == [
        "item-1",
        "item-2",
        "item-3",
    ]
    # the skip is logged as the cover it is, not as a lost clip
    assert any("LEAD-OVERLAY cover" in call.args[0] for call in logger.debug.call_args_list)


async def test_a_pre_only_break_that_outlasts_its_declared_length_plays_its_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A whole-second declared duration must not cut the break's real end off.

    The probed duration is floored to whole seconds, so the break's audio outlasts the head
    planned from it by up to a second. Treating the head as covering it because the declared
    length is covered would drop that speech: what the stream still delivers after the head
    is served as the item's own body, after the music held back under it played out.
    """
    overrun = 0.6
    audio, queue, mass, track, clip, captured = _pre_only_clip_flow(monkeypatch, overrun=overrun)
    opened, mixes = captured

    output = await _drain(_flow(audio, queue, track))

    second = SAMPLE_SIZE
    track_bytes = LEAD_TRACK_SECONDS * second
    head_bytes = PRE_ONLY_CLIP_SECONDS * second
    rest_bytes = round(overrun * second)
    # the track plays whole, with the break's head over its last seconds ...
    assert output[: track_bytes - head_bytes] == bytes([0x10]) * (track_bytes - head_bytes)
    assert output[track_bytes - head_bytes : track_bytes] != bytes([0x10]) * head_bytes
    # ... and the break's real end follows it, heard once, as its own body
    assert output[track_bytes : track_bytes + rest_bytes] == bytes([0x40]) * rest_bytes
    # the track after it owns every byte from there on: no hole, no silence
    assert output[track_bytes + rest_bytes :] == bytes([0x50]) * (
        len(output) - track_bytes - rest_bytes
    )
    assert len(output) == (LEAD_TRACK_SECONDS + overrun + PRE_ONLY_AFTER_SECONDS) * second
    # the clip was read once, from its start, and never opened a second time as an item
    assert [entry for entry in opened if entry[0] == "item-2"] == [("item-2", 0)]
    # the body continues where the mixed head ended, so no word of the break is heard twice
    flow_log = mass.player_queues.queue_data.return_value.flow_mode_stream_log
    assert flow_log[1].seek_position == float(PRE_ONLY_CLIP_SECONDS)
    # nothing of it is left to mix into the boundary after it
    assert mixes == []
    cast("Any", audio.smart_fades_mixer.build).assert_not_awaited()
    # the item is not reported as broken, and its head is not cancelled as unheard
    logger = cast("Any", audio.logger)
    assert not [
        call
        for call in logger.warning.call_args_list
        if isinstance(call.args[0], str) and "produced no audio data" in call.args[0]
    ]
    cancelled = [
        call.args[1]
        for call in logger.debug.call_args_list
        if call.args[0] == "PROGRAM-BUS cancel reason=%s label=%s"
    ]
    assert "head_outlasted" not in cancelled
    assert clip.streamdetails.stream_error is False
    # the queue reports the break from where it is really heard, for the rest it still had
    assert [entry.queue_item_id for entry in flow_log] == ["item-1", "item-2", "item-3"]
    assert [entry.seconds_streamed for entry in flow_log] == [
        LEAD_TRACK_SECONDS,
        overrun,
        PRE_ONLY_AFTER_SECONDS,
    ]
    # and it keeps its declared length: the floor is not written back as a longer one
    assert clip.streamdetails.duration == PRE_ONLY_CLIP_SECONDS


LONG_PRE_ONLY_SECONDS = int(MAX_LEAD_OVERLAP) + 8


async def test_a_pre_only_break_longer_than_the_lead_cap_keeps_its_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A break whose head is capped below its length still has speech left to serve.

    The plan only ever takes ``MAX_LEAD_OVERLAP`` of a declared pre, so a longer break
    keeps a body. That body is served as the item it is, from past the head, instead of
    being carried whole and losing the speech the cap left in its stream.
    """
    audio, queue, mass, track, clip, captured = _pre_only_clip_flow(
        monkeypatch, clip_seconds=LONG_PRE_ONLY_SECONDS
    )
    opened, mixes = captured

    output = await _drain(_flow(audio, queue, track))

    second = SAMPLE_SIZE
    track_bytes = LEAD_TRACK_SECONDS * second
    head_bytes = int(MAX_LEAD_OVERLAP * second)
    body_bytes = (LONG_PRE_ONLY_SECONDS - int(MAX_LEAD_OVERLAP)) * second
    # the track plays whole, with only the capped head over its last seconds ...
    assert output[: track_bytes - head_bytes] == bytes([0x10]) * (track_bytes - head_bytes)
    assert output[track_bytes - head_bytes : track_bytes] != bytes([0x10]) * head_bytes
    # ... and the break's body follows it, served as the item it still is
    assert output[track_bytes : track_bytes + body_bytes] == bytes([0x40]) * body_bytes
    # the track after it owns every byte from there on: no hole, no silence
    assert output[track_bytes + body_bytes :] == bytes([0x50]) * (
        len(output) - track_bytes - body_bytes
    )
    assert (
        len(output)
        == (
            LEAD_TRACK_SECONDS
            + LONG_PRE_ONLY_SECONDS
            - int(MAX_LEAD_OVERLAP)
            + PRE_ONLY_AFTER_SECONDS
        )
        * second
    )
    # the break was read once, from its start, and never opened a second time as an item
    assert [entry for entry in opened if entry[0] == "item-2"] == [("item-2", 0)]
    # the body continues exactly where the capped head ended
    flow_log = mass.player_queues.queue_data.return_value.flow_mode_stream_log
    assert [entry.queue_item_id for entry in flow_log] == ["item-1", "item-2", "item-3"]
    assert flow_log[1].seek_position == float(int(MAX_LEAD_OVERLAP))
    assert flow_log[1].seconds_streamed == pytest.approx(
        LONG_PRE_ONLY_SECONDS - int(MAX_LEAD_OVERLAP), abs=0.01
    )
    # no second boundary mix is needed: the break was an item, not a carried cover
    assert mixes == []
    # the item is not reported as broken, and its head is not cancelled as unheard
    logger = cast("Any", audio.logger)
    assert not [
        call
        for call in logger.warning.call_args_list
        if isinstance(call.args[0], str) and "produced no audio data" in call.args[0]
    ]
    assert clip.streamdetails.stream_error is False
    assert clip.streamdetails.duration == LONG_PRE_ONLY_SECONDS


async def test_a_stalled_source_does_not_hold_the_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source that stops delivering owes nothing, so looking for its rest may not block."""
    prefetcher = _IncomingFadePrefetcher(cast("Any", MagicMock()), TEST_PCM_FORMAT, "session")
    clip = cast("Any", _queue_item("clip", "Transition", duration=LEAD_CLIP_SECONDS))
    taken = StreamDetails(provider="ai_radio", item_id="clip_1", audio_format=TEST_PCM_FORMAT)
    clip.streamdetails = taken

    async def _stalled() -> AsyncGenerator[bytes]:
        await asyncio.Event().wait()
        yield b""  # pragma: no cover - never reached

    monkeypatch.setattr("music_assistant.controllers.streams.audio.REST_PEEK_TIMEOUT", 0.05)
    prefetcher._task = asyncio.create_task(asyncio.sleep(0))
    prefetcher._queue_item_id = "clip"
    prefetcher._head_seconds = LEAD
    prefetcher._streamdetails = taken
    prefetcher._stream = _stalled()
    await prefetcher._task

    started = asyncio.get_running_loop().time()
    assert await prefetcher.rest_pending(clip) is False
    assert asyncio.get_running_loop().time() - started < 0.5


POST_CLIP_SECONDS = 4
POST_OVERLAP = 2.0
POST_ONLY_SECONDS = {
    "item-1": LEAD_TRACK_SECONDS,
    "item-2": POST_CLIP_SECONDS,
    "item-3": 8,
}


def _post_only_clip() -> SimpleNamespace:
    """Build a clip that declares a post but no pre: the common case, the pre is opt-in."""
    clip = _queue_item("item-2", "Transition", duration=POST_CLIP_SECONDS)
    clip.media_type = MediaType.SOUND_EFFECT
    clip.streamdetails.tail_overlap = TailOverlap(
        duration=POST_OVERLAP, next_queue_item_id="item-3"
    )
    return clip


def _post_only_flow(
    monkeypatch: pytest.MonkeyPatch, *, program_bus: bool
) -> tuple[StreamsAudio, SimpleNamespace, MagicMock, SimpleNamespace, list[Any]]:
    """Build a flow of track -> post-only clip -> track, marking each item's audio."""
    track = _queue_item("item-1", "Track", duration=LEAD_TRACK_SECONDS)
    clip = _post_only_clip()
    after = _queue_item("item-3", "After", duration=POST_ONLY_SECONDS["item-3"])
    audio, queue, mass = _flow_audio(
        monkeypatch,
        next_item=clip,
        load_next=[clip, after, QueueEmpty],
        crossfade_mode=CrossfadeMode.STANDARD_CROSSFADE,
        program_bus=program_bus,
        build_result=SimpleNamespace(
            timing_info=SimpleNamespace(
                fadein_trimmed_duration=0.0,
                crossfade_duration=POST_OVERLAP,
                pre_crossfade_duration=0.0,
            )
        ),
    )
    audio.logger = MagicMock()
    mixes = _summing_mix(monkeypatch, audio)
    # the real rule decides whether a declared overlap applies
    del audio.crossfade_allowed
    next_items = {"item-1": clip, "item-2": after}
    mass.player_queues.get_next_item.side_effect = lambda _queue_id, item_id: next_items.get(
        item_id
    )
    opened: list[tuple[str, float]] = []

    async def _item_stream(
        queue_item: SimpleNamespace, *_args: object, **kwargs: object
    ) -> AsyncGenerator[bytes]:
        item_id = queue_item.queue_item_id
        opened.append((item_id, cast("float", kwargs.get("seek_position", 0))))
        remaining = POST_ONLY_SECONDS[item_id] - int(cast("float", kwargs.get("seek_position", 0)))
        for _ in range(max(0, remaining)):
            yield bytes([ITEM_MARKERS[item_id]]) * SAMPLE_SIZE
            await asyncio.sleep(0)

    monkeypatch.setattr(audio, "get_queue_item_stream", _item_stream)
    return audio, queue, mass, track, [opened, mixes]


async def test_a_post_only_clip_mixes_its_post_once_at_the_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    With the bus on and a crossfade on, a clip that declares only a post stays an item.

    Its post is the boundary's voice over, mixed exactly once at the clip's end; the bus
    itself adds nothing anywhere, so the session is byte-identical to the same flow with
    the experiment switched off.
    """
    reference_audio, reference_queue, _reference_mass, reference_track, _reference = (
        _post_only_flow(monkeypatch, program_bus=False)
    )
    reference = await _drain(_flow(reference_audio, reference_queue, reference_track))
    audio, queue, mass, track, captured = _post_only_flow(monkeypatch, program_bus=True)
    opened, mixes = captured
    published: list[FlowProgramBus] = []
    mass.streams.register_program_bus.side_effect = lambda _queue_id, bus: published.append(bus)

    output = await _drain(_flow(audio, queue, track))

    second = SAMPLE_SIZE
    track_bytes = LEAD_TRACK_SECONDS * second
    body_bytes = int((POST_CLIP_SECONDS - POST_OVERLAP) * second)
    # the bus is a pass-through here: every byte outside the mixed window is the reference's
    assert output == reference
    # the track plays whole and untouched - no head was planned over its tail
    assert output[:track_bytes] == bytes([0x10]) * track_bytes
    # the clip is served as the item it is, from its start, up to the post it holds back
    assert output[track_bytes : track_bytes + body_bytes] == bytes([0x40]) * body_bytes
    assert [entry for entry in opened if entry[0] == "item-2"] == [("item-2", 0)]
    # the post is mixed exactly once, at the boundary into the track it was planned against
    assert len(mixes) == 1
    assert mixes[0][0] == bytes([0x40]) * int(POST_OVERLAP * second)
    build = cast("Any", audio.smart_fades_mixer.build).await_args_list
    assert [call.kwargs["mode"] for call in build] == [CrossfadeMode.VOICE_OVER]
    assert build[0].kwargs["lead"] is False
    assert len(build[0].kwargs["fade_out_data"]) == int(POST_OVERLAP * second)
    using = [
        call.args[1:]
        for call in cast("Any", audio.logger).debug.call_args_list
        if call.args[0] == "Using a %.1f second %s for %s"
    ]
    assert using == [(POST_OVERLAP, "voice_over", "test://item-3")]
    # the clip is an item of the flow, credited with its own seconds; the track after it
    # loses exactly the intro the post was mixed over
    flow_log = mass.player_queues.queue_data.return_value.flow_mode_stream_log
    assert [entry.queue_item_id for entry in flow_log] == ["item-1", "item-2", "item-3"]
    assert [entry.seconds_streamed for entry in flow_log] == [
        LEAD_TRACK_SECONDS,
        POST_CLIP_SECONDS,
        POST_ONLY_SECONDS["item-3"] - POST_OVERLAP,
    ]
    # the bus counts emitted frames: the session is one post overlap shorter than its items
    assert published[0].produced_frames == len(output) // (SAMPLE_SIZE // SECOND)
    assert not published[0].active


PRE_POST_CLIP_SECONDS = 6
PRE_POST_REAL_SECONDS = 6.6
PRE_POST_SECONDS = {
    "item-1": LEAD_TRACK_SECONDS,
    "item-2": PRE_POST_CLIP_SECONDS,
    "item-3": 8,
}


def _pre_post_clip() -> SimpleNamespace:
    """Build a clip whose head and post leave a body of its own in between."""
    clip = _queue_item("item-2", "Transition", duration=PRE_POST_CLIP_SECONDS)
    clip.media_type = MediaType.SOUND_EFFECT
    clip.extra_attributes[PRE_OVERLAP_ATTR] = COVERED_HEAD
    clip.extra_attributes[PRE_OVERLAP_FROM_ATTR] = "item-1"
    clip.streamdetails.tail_overlap = TailOverlap(
        duration=COVERED_POST, next_queue_item_id="item-3"
    )
    return clip


def _pre_post_clip_flow(
    monkeypatch: pytest.MonkeyPatch,
    *,
    overrun: float = PRE_POST_REAL_SECONDS - PRE_POST_CLIP_SECONDS,
) -> tuple[StreamsAudio, SimpleNamespace, MagicMock, SimpleNamespace, SimpleNamespace, list[Any]]:
    """
    Build a flow of track -> pre+post clip -> track, marking each item's audio.

    The clip's real audio outlasts its declared duration by ``overrun`` seconds, read in
    tenths so the sub-second rest is representable.

    :param overrun: Seconds the clip's real audio outlasts its declared duration.
    """
    track = _queue_item("item-1", "Track", duration=LEAD_TRACK_SECONDS)
    clip = _pre_post_clip()
    after = _queue_item("item-3", "After", duration=PRE_POST_SECONDS["item-3"])
    real_seconds: dict[str, float] = dict(PRE_POST_SECONDS)
    real_seconds["item-2"] = PRE_POST_CLIP_SECONDS + overrun
    audio, queue, mass = _flow_audio(
        monkeypatch,
        next_item=clip,
        load_next=[clip, after, QueueEmpty],
        program_bus=True,
    )

    async def _build(**kwargs: Any) -> object:
        # a voice over blends exactly the tail that was held back, so the timing the flow
        # credits has to be that length rather than a fixed window
        return SimpleNamespace(
            timing_info=SimpleNamespace(
                fadein_trimmed_duration=0.0,
                crossfade_duration=len(kwargs["fade_out_data"]) / SAMPLE_SIZE,
                pre_crossfade_duration=0.0,
            )
        )

    audio.smart_fades_mixer.build.side_effect = _build  # type: ignore[attr-defined]
    audio.logger = MagicMock()
    mixes = _summing_mix(monkeypatch, audio)
    # the real rule decides whether a declared lead applies
    del audio.crossfade_allowed
    next_items = {"item-1": clip, "item-2": after}
    mass.player_queues.get_next_item.side_effect = lambda _queue_id, item_id: next_items.get(
        item_id
    )
    opened: list[tuple[str, float]] = []

    async def _item_stream(
        queue_item: SimpleNamespace, *_args: object, **kwargs: object
    ) -> AsyncGenerator[bytes]:
        item_id = queue_item.queue_item_id
        seek_position = cast("float", kwargs.get("seek_position", 0))
        opened.append((item_id, seek_position))
        # a decoded clip delivers in tenths of a second, so a declared duration that is
        # short of the real one is representable
        remaining = round((real_seconds[item_id] - seek_position) * 10)
        for _ in range(max(0, remaining)):
            yield bytes([ITEM_MARKERS[item_id]]) * (SAMPLE_SIZE // 10)
            await asyncio.sleep(0)

    monkeypatch.setattr(audio, "get_queue_item_stream", _item_stream)
    return audio, queue, mass, track, clip, [opened, mixes]


async def test_a_pre_and_post_clip_serves_the_body_its_declared_length_hides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A clip with a body still plays that body, however much longer its real audio is.

    Its head goes over the track's tail, its own body is served as the item from past the
    head, and its post sits at the boundary into the track it was planned against. The
    whole-second declared length is not the end of the clip: the rest of its stream is
    served too, without it ever being reported as an item that produced no audio.
    """
    audio, queue, mass, track, clip, captured = _pre_post_clip_flow(monkeypatch)
    opened, mixes = captured

    output = await _drain(_flow(audio, queue, track))

    second = SAMPLE_SIZE
    track_bytes = LEAD_TRACK_SECONDS * second
    head_bytes = int(COVERED_HEAD * second)
    declared_body = PRE_POST_CLIP_SECONDS - COVERED_HEAD - COVERED_POST
    # the track plays whole, with the clip's head over its last seconds ...
    assert output[: track_bytes - head_bytes] == bytes([0x10]) * (track_bytes - head_bytes)
    assert output[track_bytes - head_bytes : track_bytes] != bytes([0x10]) * head_bytes
    # ... and the clip's body follows it on its own, from past the head
    body_end = output.find(bytes([0x50]) * 4, track_bytes)
    assert body_end > track_bytes
    body_seconds = (body_end - track_bytes) / second
    # the body is served whole: more than the declared length left to it, up to what the
    # stream really holds
    assert declared_body < body_seconds <= PRE_POST_REAL_SECONDS - COVERED_HEAD
    # the track after the clip owns every byte from there on: no hole, no silence
    assert output[body_end:] == bytes([0x50]) * (len(output) - body_end)
    # the whole clip was read once, and its body continues on that same stream
    assert [entry for entry in opened if entry[0] == "item-2"] == [("item-2", 0)]
    body_log = [call.args for call in cast("Any", audio.logger).debug.call_args_list]
    assert (
        "LEAD-OVERLAY body clip=%s(%s) lead=%.2fs",
        "item-2",
        MediaType.SOUND_EFFECT,
        COVERED_HEAD,
    ) in body_log
    assert not [call for call in body_log if call[0] == "LEAD-OVERLAY cover clip=%s(%s)"]
    # the post sits at the boundary into the next track, mixed there exactly once: the
    # clipped tail of the body is its fade-out side, the next track's intro its fade-in
    assert len(mixes) == 1
    assert set(mixes[0][0]) == {0x40}
    assert set(mixes[0][1]) == {0x50}
    assert len(mixes[0][1]) == len(mixes[0][0]) + int(VOICE_OVER_RAMP * second)
    build = cast("Any", audio.smart_fades_mixer.build).await_args
    assert build is not None
    assert build.kwargs["mode"] == CrossfadeMode.VOICE_OVER
    assert build.kwargs["lead"] is False
    # the body is the item's own audio, reported from where it is really heard
    flow_log = mass.player_queues.queue_data.return_value.flow_mode_stream_log
    assert [entry.queue_item_id for entry in flow_log] == ["item-1", "item-2", "item-3"]
    assert flow_log[1].seek_position == COVERED_HEAD
    assert flow_log[1].seconds_streamed == pytest.approx(
        PRE_POST_REAL_SECONDS - COVERED_HEAD, abs=0.01
    )
    # nothing of the clip is lost or reported as an item without audio
    assert clip.streamdetails.stream_error is False
    assert clip.streamdetails.duration == PRE_POST_CLIP_SECONDS
    cast("Any", audio.logger).warning.assert_not_called()


async def test_a_refused_placement_reports_why() -> None:
    """The bus says whether a program lost its window or arrived too late."""
    program_bus = _bus()
    await program_bus.feed(_music(3.0))
    assert (
        program_bus.schedule(
            _voice(2.0), int(5 * SECOND), label="first", priority=ProgramPriority.TERM
        )
        is not None
    )

    # the anchor has to move past the window of the first program, and the frame this one
    # may not pass is already behind it: it did not lose a window, its own had closed
    refused_closed = program_bus.schedule(
        _voice(2.0),
        int(5 * SECOND),
        label="second",
        priority=ProgramPriority.TERM,
        stop_frame=int(5.5 * SECOND),
    )
    assert refused_closed is None
    assert program_bus.last_refusal is ScheduleReason.TOO_LATE

    refused_late = program_bus.schedule(_voice(2.0), 0, label="late", priority=ProgramPriority.TERM)
    assert refused_late is None
    assert program_bus.last_refusal is ScheduleReason.TOO_LATE


async def test_a_program_its_stop_frame_leaves_no_room_for_is_late() -> None:
    """A program the frame it may not pass shrinks to nothing is too late, not displaced."""
    program_bus = _bus()
    await program_bus.feed(_music(3.0))

    assert (
        program_bus.schedule(
            _voice(2.0), int(1 * SECOND), label="closed", stop_frame=int(1 * SECOND)
        )
        is None
    )

    assert program_bus.last_refusal is ScheduleReason.TOO_LATE


async def test_a_chunk_that_ends_mid_frame_carries_its_bytes_into_the_next_one() -> None:
    """A partial frame is no frame yet: its bytes go out first in the chunk after it."""
    logger = MagicMock()
    program_bus = FlowProgramBus(TEST_PCM_FORMAT, cast("Any", logger))

    kept = await program_bus.feed(_music(1.0) + b"\x11")
    carried = await program_bus.feed(b"\x22\x33\x44" + _music(0.5))

    # nothing of the first chunk is lost, and its last byte is held back for the frame
    assert kept == _music(1.0)
    # the three bytes of the next chunk complete the frame the first one left over, and
    # the rest of that chunk follows it in order
    assert carried[:4] == b"\x11\x22\x33\x44"
    assert carried[4:] == _music(0.5)
    assert len(carried) == len(_music(0.5)) + 4
    # the byte count is the stream's, not the chunk's: nothing rests once the frames of
    # the two chunks add up again
    assert await program_bus.feed(_music(0.5)) == _music(0.5)
    assert program_bus.produced_frames == 2 * SECOND + 1
    logger.warning.assert_called_once()
    message = str(logger.warning.call_args.args[0])
    assert "carried" in message
    assert "frame" in message


async def test_a_partial_frame_is_only_held_back_for_the_next_chunk() -> None:
    """A chunk that ends mid-frame counts its whole frames and holds its rest back, once."""
    bus = FlowProgramBus(TEST_PCM_FORMAT, cast("Any", MagicMock()))
    stream = _music(1.0) + b"\x01\x02" + _music(1.0) + b"\x03" + _music(1.0)

    emitted = b"".join(
        [
            await bus.feed(_music(1.0) + b"\x01\x02"),
            await bus.feed(_music(1.0) + b"\x03"),
            await bus.feed(_music(1.0)),
        ]
    )

    # the stream goes out whole and in its order, up to the last frame it makes: the
    # carried bytes are no loss, they are the front of the chunk after them
    assert emitted == stream[: len(stream) - len(stream) % bus.frame_size]
    assert bus.produced_frames == len(emitted) // bus.frame_size
    # the three bytes that make no frame yet wait for the fourth, and go out with it
    tail = await bus.feed(b"\x04")
    assert tail == stream[len(emitted) :] + b"\x04"
    assert bus.produced_frames == len(stream) // bus.frame_size + 1


async def test_a_program_placed_at_a_frame_is_mixed_into_unaligned_chunks() -> None:
    """Frames, not chunks, decide where a program starts, even when no chunk ends on one."""
    bus = _bus()
    assert bus.schedule(_voice(100 / SECOND), 600, label="term") == 600

    stream = _music(1.0) + b"\x02\x03"
    chunks = [stream[offset : offset + 1002] for offset in range(0, len(stream), 1002)]
    emitted = b"".join([await bus.feed(chunk) for chunk in chunks])

    # what left the bus is whole frames and the stream's own bytes: nothing is lost at the
    # chunk boundaries, only the partial frame at the very end waits for its rest
    assert len(emitted) == len(stream) - len(stream) % bus.frame_size
    assert bus.produced_frames == len(emitted) // bus.frame_size
    levels = _levels(emitted)
    music_levels = _levels(stream[: len(emitted)])
    assert levels[:600] == music_levels[:600]
    # the program covers 600-699; at 700 the voice is over and only the duck ramp is left
    ducked = int(100 * (1 - VOICE_OVER_DUCK_DEPTH))
    assert levels[599] == music_levels[599]
    assert levels[699] == ducked + 8
    assert levels[700] == ducked


async def test_a_partial_frame_left_at_the_end_of_a_session_is_given_up_with_a_log() -> None:
    """A stream that ends inside a frame holds bytes nothing completes: the bus says so."""
    logger = MagicMock()
    program_bus = FlowProgramBus(TEST_PCM_FORMAT, cast("Any", logger))
    await program_bus.feed(_music(1.0) + b"\x01\x02")

    program_bus.close()

    # the two bytes are no frame and no chunk follows to complete them
    assert program_bus.produced_frames == SECOND
    assert logger.debug.call_args.args[1] == 2


async def test_a_session_that_ends_on_a_frame_closes_without_a_report() -> None:
    """Nothing is given up when the last chunk made whole frames."""
    logger = MagicMock()
    program_bus = FlowProgramBus(TEST_PCM_FORMAT, cast("Any", logger))
    await program_bus.feed(_music(0.5))

    program_bus.close()

    assert program_bus.produced_frames == SECOND // 2
    logger.debug.assert_not_called()


async def test_leftover_bytes_are_reported_once_in_a_burst() -> None:
    """A stream that keeps emitting a partial frame is reported, not warned about per chunk."""
    logger = MagicMock()
    program_bus = FlowProgramBus(TEST_PCM_FORMAT, cast("Any", logger))

    for _ in range(5):
        await program_bus.feed(_music(1.0) + b"\x01")

    # the five carried bytes make one more frame than five whole seconds of chunks
    assert program_bus.produced_frames == 5 * SECOND + 1
    logger.warning.assert_called_once()


async def test_a_successful_placement_clears_the_last_refusal() -> None:
    """The refusal is only about the placement that was just refused."""
    program_bus = _bus()
    await program_bus.feed(_music(3.0))
    assert program_bus.schedule(_voice(2.0), 0, label="late", priority=ProgramPriority.TERM) is None
    assert program_bus.last_refusal is ScheduleReason.TOO_LATE

    assert (
        program_bus.schedule(
            _voice(1.0), int(5 * SECOND), label="on-time", priority=ProgramPriority.TERM
        )
        is not None
    )
    assert program_bus.last_refusal is None


def test_a_program_whose_window_is_taken_hands_its_audio_back() -> None:
    """A program dropped before its first frame tells whoever holds its audio to keep it."""
    bus = _bus()
    given_back: list[str] = []
    assert (
        bus.schedule(
            _voice(2.0),
            10 * SECOND,
            label="lead",
            on_dropped=lambda: given_back.append("lead"),
        )
        == 10 * SECOND
    )

    dropped = bus.drop_window(10 * SECOND, 2 * SECOND, below=ProgramPriority.ANNOUNCEMENT)

    assert dropped == [(10 * SECOND, "lead")]
    assert given_back == ["lead"]
    assert not bus.active


async def test_a_program_that_already_started_is_never_handed_back() -> None:
    """What the listener already heard would be spoken twice if it were given back."""
    bus = _bus()
    given_back: list[str] = []
    assert (
        bus.schedule(
            _voice(4.0),
            int(1.0 * SECOND),
            label="lead",
            on_dropped=lambda: given_back.append("lead"),
        )
        is not None
    )
    await bus.feed(_music(2.0))

    bus.drop_window(int(1.0 * SECOND), int(2.0 * SECOND), below=ProgramPriority.ANNOUNCEMENT)

    assert not bus.active
    assert given_back == []


async def test_an_announcement_that_takes_a_leads_window_keeps_the_clips_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A head that never reached the bus is no skipped audio: the clip starts at its first frame."""
    audio, queue, mass, opened, track = _lead_flow(monkeypatch, program_bus=True)
    logger = MagicMock()
    audio.logger = logger
    ledger = ProgramLedger(logger=logger)
    lead_anchor = int((LEAD_TRACK_SECONDS - LEAD) * SECOND)
    announcement: list[Any] = []

    output = bytearray()
    async for chunk in _flow(audio, queue, track):
        output.extend(chunk)
        if not announcement and mass.streams.register_program_bus.called:
            program_bus = mass.streams.register_program_bus.call_args.args[1]
            if program_bus.produced_frames >= lead_anchor:
                # an announcement is immediate: it lands on the produced frame, which is
                # the frame the head was placed on, so it takes that window
                result = place_program(
                    program_bus,
                    _voice(1.0),
                    request=ProgramRequest(
                        queue_id="queue-1",
                        label="announcement",
                        start_in_seconds=0.0,
                        priority=ProgramPriority.ANNOUNCEMENT,
                        kind=ProgramKind.ANNOUNCEMENT,
                    ),
                    audible_frame=program_bus.produced_frames - int(LEAD_MIN_LEAD_SECONDS * SECOND),
                    ledger=ledger,
                    logger=logger,
                    report_no_position=lambda *_args: None,
                )
                announcement.append(result)
        await asyncio.sleep(0)

    assert len(announcement) == 1
    assert announcement[0].scheduled is True
    # the head was taken out of the clip's stream and put back, so the clip is read from its
    # first frame twice: once for the head, once for the body that now starts at 0
    assert [entry for entry in opened if entry[0] == "item-2"] == [("item-2", 0), ("item-2", 0)]
    flow_log = mass.player_queues.queue_data.return_value.flow_mode_stream_log
    assert [entry.queue_item_id for entry in flow_log] == ["item-1", "item-2"]
    assert [entry.seek_position for entry in flow_log] == [0, 0]
    # nothing of the clip's speech is missing: its whole length follows the track
    body = int(LEAD_CLIP_SECONDS * SAMPLE_SIZE)
    assert output[-body:] == bytes([0x40]) * body

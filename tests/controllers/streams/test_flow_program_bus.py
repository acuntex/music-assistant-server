"""Tests for the flow session's program bus: scheduled programs, and the lead as an anchor."""

from __future__ import annotations

import asyncio
import struct
from collections import deque
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

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
    StreamsAudio,
    _FlowProgramBus,
    _IncomingFadePrefetcher,
    _ProgramLead,
)
from music_assistant.controllers.streams.constants import (
    CONF_STREAM_PROGRAM_BUS,
    PRE_OVERLAP_ATTR,
    PRE_OVERLAP_FROM_ATTR,
    VOICE_OVER_RAMP,
)
from music_assistant.controllers.streams.controller import StreamsController
from music_assistant.controllers.streams.smart_fades.filters import (
    MIX_CEILING_DB,
    VOICE_OVER_DUCK_DEPTH,
)
from tests.controllers.streams.test_crossfade_transition import (
    TEST_PCM_FORMAT,
    _flow_audio,
    _queue_item,
)
from tests.controllers.streams.test_flow_format import _make_player, _make_streams_audio

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


def _bus() -> _FlowProgramBus:
    """Build a bus on the test PCM format."""
    return _FlowProgramBus(TEST_PCM_FORMAT, cast("Any", MagicMock()))


def _music(seconds: float, level: int = 100) -> bytes:
    """Build music at the given level for the given number of seconds."""
    return struct.pack("<hh", level, level) * int(seconds * SECOND)


def _voice(seconds: float, level: int = 8) -> bytes:
    """Build a voice at the given level for the given number of seconds."""
    return struct.pack("<hh", level, level) * int(seconds * SECOND)


def _levels(pcm: bytes) -> list[int]:
    """Return the first channel's samples of a PCM chunk."""
    return list(struct.unpack(f"<{len(pcm) // 2}h", pcm))[::2]


def test_the_bus_passes_bytes_through_untouched_without_a_program() -> None:
    """A session without a due program hands every chunk back exactly as it came in."""
    bus = _bus()
    chunks = [bytes([index, index + 1, index + 2, index + 3]) * 100 for index in range(4)]

    output = [bus.feed(chunk) for chunk in chunks]

    assert all(result is chunk for result, chunk in zip(output, chunks, strict=True))
    assert bus.produced_frames == sum(len(chunk) for chunk in chunks) // 4
    assert not bus.active


def test_a_program_is_mixed_from_its_anchor_on_without_changing_the_length() -> None:
    """Everything before the anchor is untouched, everything inside is ducked and summed."""
    bus = _bus()
    anchor = bus.schedule(_voice(2.0), int(1.0 * SECOND), label="test")
    assert anchor == SECOND

    output = [bus.feed(_music(1.0)) for _ in range(5)]

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


def test_a_program_is_dropped_when_its_window_already_passed() -> None:
    """A program that would have played entirely in the past can not be placed."""
    bus = _bus()
    bus.feed(_music(3.0))

    assert bus.schedule(_voice(1.0), int(0.5 * SECOND), label="late") is None
    assert not bus.active


def test_a_program_that_is_only_partly_in_the_past_starts_late() -> None:
    """A program whose anchor already went by is moved to the current frame instead."""
    bus = _bus()
    bus.feed(_music(2.5))

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


def test_cancel_all_drops_a_program_at_once_and_leaves_the_audio_alone() -> None:
    """Cancelling mid-program returns to the untouched bytes and leaves no program behind."""
    bus = _bus()
    bus.schedule(_voice(2.0), int(1.0 * SECOND), label="test")
    bus.feed(_music(1.5))

    bus.cancel_all("test stop")

    assert not bus.active
    assert bus.feed(_music(1.0)) == _music(1.0)


async def test_cancel_all_leaves_no_task_behind() -> None:
    """Cancelling a program does not leave work of its own running."""

    async def _run() -> bytes:
        bus = _bus()
        bus.schedule(_voice(2.0), 0, label="test")
        mixed = await asyncio.to_thread(bus.feed, _music(2.0))
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
    monkeypatch: pytest.MonkeyPatch, *, program_bus: bool
) -> tuple[StreamsAudio, SimpleNamespace, MagicMock, list[tuple[str, float]], SimpleNamespace]:
    """Build a flow whose next item is a clip that starts over the running track's tail."""
    track = _queue_item("item-1", "Track", duration=LEAD_TRACK_SECONDS)
    clip = _lead_clip()
    audio, queue, mass = _flow_audio(
        monkeypatch,
        next_item=clip,
        load_next=[clip, QueueEmpty],
        program_bus=program_bus,
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


async def test_the_lead_is_not_mixed_when_the_program_bus_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the experiment off the flow plays the track untouched and opens no bus."""
    audio, queue, mass, _opened, track = _lead_flow(monkeypatch, program_bus=False)

    output = await _drain(_flow(audio, queue, track))

    track_bytes = int(LEAD_TRACK_SECONDS * SAMPLE_SIZE)
    assert output[:track_bytes] == bytes([0x10]) * track_bytes
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
    """A head moved out of its own window is left to the clip instead of cut short."""
    logger = MagicMock()
    bus = _FlowProgramBus(TEST_PCM_FORMAT, logger)
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
    # the head is not on the bus; only the announcement is
    assert bus.active
    assert any(
        call.args[0]
        == "PROGRAM-BUS schedule label=%s dropped: its window is taken by another program"
        for call in logger.debug.call_args_list
    )


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
    published: list[_FlowProgramBus] = []

    def _register(_queue_id: str, program_bus: _FlowProgramBus) -> None:
        published.append(program_bus)
        program_bus.schedule(_voice(LEAD_TRACK_SECONDS), 0, label="announcement")

    mass.streams.register_program_bus.side_effect = _register

    await _drain(_flow(audio, queue, track))

    assert not published[0].active


def test_the_voice_ramps_back_to_the_music_after_the_program() -> None:
    """The ducked music ramps up over the program's tail instead of jumping back."""
    bus = _bus()
    bus.schedule(_voice(1.0), int(1.0 * SECOND), label="test")

    output = bus.feed(_music(3.0))

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


def test_sixteen_bit_big_endian_can_not_be_mixed() -> None:
    """A big-endian session is passed through instead of being mixed as little-endian."""
    bus = _FlowProgramBus(NON_MIXABLE_PCM_FORMAT, cast("Any", MagicMock()))

    assert not bus.can_mix
    assert bus.schedule(_voice(1.0), 0, label="test") is None
    assert bus.feed(_music(1.0)) == _music(1.0)


def test_a_voice_shorter_than_the_ramp_is_fully_ducked_before_it_ends() -> None:
    """The duck-in ramp is bounded by the voice, so a short one does not step back up mid-word."""
    bus = _bus()
    voice = _voice(0.1)
    assert bus.schedule(voice, 0, label="test") is not None

    levels = _levels(bus.feed(_music(1.0)))
    voice_end = int(0.1 * SECOND)
    ducked = int(100 * (1 - VOICE_OVER_DUCK_DEPTH)) + 8
    # the last frame of the voice is already at full duck, so its end does not step back up
    assert levels[voice_end - 1] == ducked
    assert levels[voice_end] == ducked - 8


def test_the_overlay_keeps_its_frame_size_for_float_pcm() -> None:
    """32-bit float is mixed as well as the 16-bit the flow uses by default."""
    pcm_format = AudioFormat(
        content_type=ContentType.PCM_F32LE,
        sample_rate=8000,
        bit_depth=32,
        channels=2,
    )
    bus = _FlowProgramBus(pcm_format, cast("Any", MagicMock()))
    bus.schedule(struct.pack("<ff", 0.1, 0.1) * 8000, 0, label="test")

    mixed = bus.feed(struct.pack("<ff", 0.5, 0.5) * 8000)

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


def test_the_music_is_pulled_down_over_the_duck_ramp() -> None:
    """The music fades in to the duck over the same ramp the boundary's voice-over uses."""
    bus = _bus()
    bus.schedule(_voice(1.0), 0, label="test")

    levels = _levels(bus.feed(_music(2.0)))

    full = 100
    ducked = int(100 * (1 - VOICE_OVER_DUCK_DEPTH)) + 8
    ramp = int(VOICE_OVER_RAMP * SECOND)
    # the first frame is still (nearly) at full level, the ramp ends on the ducked level
    assert ducked < levels[ramp // 2] < levels[0] <= full + 8
    assert levels[ramp] == ducked
    assert levels[ramp + 1] == ducked


def test_a_full_scale_voice_over_full_scale_music_does_not_clip() -> None:
    """Summing a full-scale voice over full-scale music saturates below full scale."""
    pcm_format = AudioFormat(
        content_type=ContentType.PCM_F32LE,
        sample_rate=8000,
        bit_depth=32,
        channels=2,
    )
    bus = _FlowProgramBus(pcm_format, cast("Any", MagicMock()))
    music = struct.pack("<ff", 1.0, 1.0) * 8000
    bus.schedule(struct.pack("<ff", 1.0, 1.0) * 8000, 0, label="test")

    mixed = bus.feed(music)

    samples = struct.unpack(f"<{len(mixed) // 4}f", mixed)
    # not a single sample leaves full scale, so nothing wraps or turns into a NaN
    assert max(samples) <= 1.0
    assert min(samples) >= -1.0
    # the sum is held at the ceiling the boundary's voice-over limiter uses
    assert max(samples) == pytest.approx(10 ** (MIX_CEILING_DB / 20), abs=1e-6)
    assert len(mixed) == len(music)


def test_a_full_scale_voice_over_full_scale_music_saturates_in_sixteen_bit() -> None:
    """The 16-bit mix saturates instead of wrapping around at full scale."""
    bus = _bus()
    music = struct.pack("<hh", 32767, 32767) * SECOND
    bus.schedule(struct.pack("<hh", 32767, 32767) * SECOND, 0, label="test")

    mixed = bus.feed(music)

    levels = list(struct.unpack(f"<{len(mixed) // 2}h", mixed))
    assert max(levels) == 32767
    assert min(levels) >= -32768
    assert len(mixed) == len(music)


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
    published: list[_FlowProgramBus] = []
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
            # the source is exhausted here: nothing of the item may still wait in the bus
            produced_at_end["item-1"] = published[0].produced_frames

    monkeypatch.setattr(audio, "get_queue_item_stream", _item_stream)

    output = await _drain(_flow(audio, queue, track))

    # the item's whole audio was mixed through as it came, not held back for the head
    assert produced_at_end["item-1"] == track_frames
    # ... so its last frames really are in the stream, with the head ducked over them
    assert output[track_bytes - SAMPLE_SIZE : track_bytes] != bytes([0x10]) * SAMPLE_SIZE
    assert len(output) == track_bytes + int((LEAD_CLIP_SECONDS - LEAD) * SAMPLE_SIZE)
    report = leads[-1].anchor_report
    assert report is not None
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
    # the item produced all of its audio, so the 219 ms the durations missed are in the stream
    assert produced_at_end["item-1"] == real_end
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
    assert produced_at_end["item-1"] == real_end
    assert report.anchor == LEAD_TRACK_SECONDS * SECOND - int(LEAD * SECOND)


def test_a_late_anchor_is_clamped_forward_and_reported() -> None:
    """A frame that already went by is moved to where the flow is, never waited for."""
    logger = MagicMock()
    bus = _FlowProgramBus(TEST_PCM_FORMAT, logger)
    bus.feed(_music(1.0))
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
    mixed = bus.feed(_music(1.0))
    lead.finish()

    assert mixed != _music(1.0)
    report = lead.anchor_report
    assert report is not None
    assert (report.anchor, report.produced_end, report.declared_end) == (
        SECOND,
        2 * SECOND,
        int(1.5 * SECOND),
    )
    # the clamp dropped the head's first half second, so its end sits on the item's end
    assert report.drift_ms == pytest.approx(0, abs=1)


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
    published: list[_FlowProgramBus] = []
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
    published: list[_FlowProgramBus] = []
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
    published: list[_FlowProgramBus] = []
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
    controller.mass.players.get_player.return_value = _make_player(
        supported=[(44100, 16)], flow_mode=FLOW_MODE_SAMPLE_RATE_SMART
    )
    controller.mass.players.get_player.return_value.get_config_value = MagicMock(return_value=False)
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
            state=SimpleNamespace(
                corrected_elapsed_time=elapsed, playback_state=PlaybackState.PLAYING
            )
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

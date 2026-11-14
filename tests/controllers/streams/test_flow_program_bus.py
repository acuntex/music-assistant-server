"""Tests for the flow session's program bus: scheduled programs, and the lead as an anchor."""

from __future__ import annotations

import asyncio
import struct
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
    StreamType,
    VolumeNormalizationMode,
)
from music_assistant_models.errors import QueueEmpty
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.queue_item import QueueItem
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.constants import FLOW_MODE_SAMPLE_RATE_SMART
from music_assistant.controllers.streams.audio import (
    StreamsAudio,
    _FlowProgramBus,
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


def test_a_lead_whose_end_is_taken_by_a_program_is_not_placed() -> None:
    """A head that can not end on the item's end is left to the clip instead of cut short."""
    logger = MagicMock()
    bus = _bus()
    bus.feed(_music(0.5))
    lead = _ProgramLead(
        bus,
        logger,
        _voice(LEAD),
        label="lead:clip",
        declared_anchor=10 * SECOND,
    )
    assert lead.arm() == 10 * SECOND
    # the item's last seconds are held back for its own end ...
    for _ in range(10):
        bus.feed(lead.feed(_music(1.0)))
    assert bus.produced_frames == 8.5 * SECOND
    # ... when another program takes the window the head would move onto
    assert bus.schedule(_voice(1.0), bus.produced_frames, label="announcement") is not None

    tail = lead.finish()

    assert not lead.placed
    assert lead.anchor_report is None
    assert sum(len(chunk) for chunk in tail) == int(LEAD * SAMPLE_SIZE)
    # the head is not on the bus; only the announcement is
    assert bus.active


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


async def test_the_head_ends_where_the_item_really_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An item that delivers more audio than its declared duration still ends the head on it."""
    audio, queue, track = _drift_flow(monkeypatch)
    leads = _capture_leads(monkeypatch, audio)

    output = await _drain(_flow(audio, queue, track))

    frame_size = SAMPLE_SIZE // SECOND
    real_end = int(DRIFT_REAL_SECONDS * SECOND)
    voice = int(LEAD * SECOND)
    report = leads[-1].anchor_report
    assert report is not None
    # the anchor follows the audio the item really produced, not duration - seek
    assert report.produced_end == real_end
    assert report.anchor == real_end - voice
    assert report.declared_end == LEAD_TRACK_SECONDS * SECOND
    # the declared durations put the head 219 ms early, which is what the fix moved out
    assert report.anchor - (report.declared_end - voice) == real_end - report.declared_end
    # everything before the head is untouched, the head is heard up to the item's end
    before = report.anchor * frame_size
    assert output[:before] == bytes([0x10]) * before
    mixed_tail = output[before : real_end * frame_size]
    assert mixed_tail != bytes([0x10]) * len(mixed_tail)
    assert output[(real_end - 1) * frame_size : real_end * frame_size] != bytes([0x10]) * frame_size
    # and what is left of that drift is inaudible
    assert abs(report.drift_ms) < 50


async def test_a_lead_whose_source_still_streams_is_held_the_same_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source that keeps trickling is held back like any other: the head ends on its end."""
    audio, queue, track = _drift_flow(monkeypatch)
    leads = _capture_leads(monkeypatch, audio)
    # holding a still-streaming tail is what the player's buffer is there for
    cast("Any", track.streamdetails.buffer).eof = False

    output = await _drain(_flow(audio, queue, track))

    frame_size = SAMPLE_SIZE // SECOND
    real_end = int(DRIFT_REAL_SECONDS * SECOND)
    voice = int(LEAD * SECOND)
    report = leads[-1].anchor_report
    assert report is not None
    assert (report.anchor, report.produced_end, report.declared_end) == (
        real_end - voice,
        real_end,
        LEAD_TRACK_SECONDS * SECOND,
    )
    assert abs(report.drift_ms) < 50
    before = report.anchor * frame_size
    assert output[:before] == bytes([0x10]) * before


def test_a_lead_that_lost_its_window_reports_the_drift_it_kept() -> None:
    """A head that lost its window can not be held, so the report measures what is left."""
    logger = MagicMock()
    bus = _bus()
    bus.feed(_music(1.0))
    lead = _ProgramLead(
        bus,
        logger,
        _voice(1.0),
        label="lead:clip",
        declared_anchor=SECOND,
    )
    # the item is already at the frame the declared durations put the head's end on
    assert lead.arm() == SECOND

    for _ in range(3):
        released = lead.feed(_music(1.0))
        assert released == _music(1.0)
        bus.feed(released)
    tail = lead.finish()

    # nothing was held back: the item's own audio went through as it came
    assert tail == ()
    report = lead.anchor_report
    assert report is not None
    assert report.anchor == SECOND
    assert (report.produced_end, report.declared_end) == (4 * SECOND, 2 * SECOND)
    # the head's end sits 2 s before the item's end, and the report says exactly that
    assert report.drift_ms == pytest.approx(-2000, abs=1)


def _drift_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[StreamsAudio, SimpleNamespace, SimpleNamespace]:
    """Build a flow whose item delivers 219 ms more audio than its declared duration."""
    track = _queue_item("item-1", "Track", duration=LEAD_TRACK_SECONDS)
    clip = _lead_clip()
    audio, queue, _mass = _flow_audio(
        monkeypatch, next_item=clip, load_next=[clip, QueueEmpty], program_bus=True
    )
    del audio.crossfade_allowed

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

    monkeypatch.setattr(audio, "get_queue_item_stream", _item_stream)
    return audio, queue, track


async def test_a_cancelled_lead_is_not_placed_behind_the_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A skip drops the head even after its tail was already held back on the bus."""
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
        call.args and call.args[0] == "PROGRAM-BUS anchor label=%s dropped: cancelled"
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

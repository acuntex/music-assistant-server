"""Tests for the flow session's program bus: scheduled programs, and the lead as an anchor."""

from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import ContentType, MediaType
from music_assistant_models.errors import QueueEmpty
from music_assistant_models.media_items import AudioFormat

from music_assistant.controllers.streams.audio import (
    StreamsAudio,
    _FlowProgramBus,
)
from music_assistant.controllers.streams.constants import (
    CONF_STREAM_PROGRAM_BUS,
    PRE_OVERLAP_ATTR,
    PRE_OVERLAP_FROM_ATTR,
    VOICE_OVER_RAMP,
)
from music_assistant.controllers.streams.smart_fades.filters import VOICE_OVER_DUCK_DEPTH
from tests.controllers.streams.test_crossfade_transition import (
    TEST_PCM_FORMAT,
    _flow_audio,
    _queue_item,
)

SAMPLE_SIZE = TEST_PCM_FORMAT.pcm_sample_size
SECOND = TEST_PCM_FORMAT.sample_rate
LEAD = 2.0
LEAD_CLIP_SECONDS = 4
LEAD_TRACK_SECONDS = 40


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
    # the music is ducked and the voice summed on top of it
    assert max(mixed) == int(100 * (1 - VOICE_OVER_DUCK_DEPTH)) + 8
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


async def test_the_program_bus_is_read_once_per_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The switch is read when the flow session starts, not for every chunk."""
    audio, queue, mass, _opened, track = _lead_flow(monkeypatch, program_bus=True)

    await _drain(_flow(audio, queue, track))

    assert mass.streams.get_config_value.call_count == 1


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

    samples = struct.unpack("<f", mixed[:4])
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

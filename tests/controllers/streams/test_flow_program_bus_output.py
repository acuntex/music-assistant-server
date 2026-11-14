"""
End-to-end tests for the program bus: an overlay mixed into a running flow session.

The flow runs for real (``get_queue_flow_stream`` with its own item streams and the bus
switched on); only the item source is served by the test. A program is placed on the bus
the session published while the stream is running, and the emitted PCM is held against a
frame-by-frame reference sum, so a program placed one frame off, a dropped or duplicated
passage, a changed length or a changed chunk boundary shows up in the comparison.
"""

from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncGenerator, Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from music_assistant_models.enums import ContentType, CrossfadeMode
from music_assistant_models.errors import QueueEmpty
from music_assistant_models.media_items import AudioFormat

from music_assistant.controllers.streams import StreamsController
from music_assistant.controllers.streams.audio import StreamsAudio, _FlowProgramBus
from music_assistant.controllers.streams.constants import (
    CONF_STREAM_PROGRAM_BUS,
    LEAD_MIN_LEAD_SECONDS,
    VOICE_OVER_RAMP,
)
from music_assistant.controllers.streams.smart_fades.filters import (
    MIX_CEILING_DB,
    VOICE_OVER_DUCK_DEPTH,
)
from tests.controllers.streams.test_crossfade_transition import (
    TEST_PCM_FORMAT,
    _flow_audio,
    _queue_item,
)

SECOND = TEST_PCM_FORMAT.sample_rate
FIRST_SECONDS = 12
SECOND_SECONDS = 8
# frames per chunk, deliberately not a whole second, so chunk edges land inside the
# program's window and inside the items
CHUNK_FRAMES = 3000
# the music carries a different level on every frame, so a program placed one frame
# off would not match the reference sum
MUSIC_BASE = 100
MUSIC_STEP = 1
PATTERN_FRAMES = 100
VOICE_BASE = 8
VOICE_STEP = 1
F32_PCM_FORMAT = AudioFormat(
    content_type=ContentType.PCM_F32LE,
    sample_rate=TEST_PCM_FORMAT.sample_rate,
    bit_depth=32,
    channels=2,
)


def _frame_size(pcm_format: AudioFormat) -> int:
    """Return the size in bytes of one frame of the given PCM format."""
    return (pcm_format.bit_depth // 8) * pcm_format.channels


def _pcm(
    seconds: float,
    pcm_format: AudioFormat,
    *,
    base: float,
    step: float,
) -> bytes:
    """Build PCM whose level changes from frame to frame, so a shift would be heard."""
    frames = int(seconds * pcm_format.sample_rate)
    levels = base + (np.arange(frames) % PATTERN_FRAMES) * step
    dtype = np.float32 if pcm_format.content_type == ContentType.PCM_F32LE else np.int16
    return np.repeat(levels.astype(dtype), pcm_format.channels).tobytes()


def _music(seconds: float, pcm_format: AudioFormat = TEST_PCM_FORMAT) -> bytes:
    """Build music for the given number of seconds."""
    if pcm_format.content_type == ContentType.PCM_F32LE:
        return _pcm(seconds, pcm_format, base=0.5, step=0.001)
    return _pcm(seconds, pcm_format, base=MUSIC_BASE, step=MUSIC_STEP)


def _voice(seconds: float) -> bytes:
    """Build a program for the given number of seconds."""
    return _pcm(seconds, TEST_PCM_FORMAT, base=VOICE_BASE, step=VOICE_STEP)


def _chunks_of(pcm: bytes, pcm_format: AudioFormat) -> list[bytes]:
    """Split PCM the way the item streams in these tests deliver it."""
    step = CHUNK_FRAMES * _frame_size(pcm_format)
    return [pcm[start : start + step] for start in range(0, len(pcm), step)]


def _two_item_flow(
    monkeypatch: pytest.MonkeyPatch,
    *,
    program_bus: bool,
    pcm_format: AudioFormat = TEST_PCM_FORMAT,
) -> tuple[
    StreamsAudio,
    SimpleNamespace,
    MagicMock,
    dict[str, bytes],
    SimpleNamespace,
    SimpleNamespace,
]:
    """Build a flow of two tracks that are cut at their boundary, marking their audio."""
    first = _queue_item("item-1", "First", duration=FIRST_SECONDS)
    second = _queue_item("item-2", "Second", duration=SECOND_SECONDS)
    audio, queue, mass = _flow_audio(
        monkeypatch,
        next_item=second,
        load_next=[second, QueueEmpty],
        crossfade_mode=CrossfadeMode.DISABLED,
        program_bus=program_bus,
    )
    audio.logger = MagicMock()
    music = {
        "item-1": _music(FIRST_SECONDS, pcm_format),
        "item-2": _music(SECOND_SECONDS, pcm_format),
    }
    chunks = {item_id: _chunks_of(pcm, pcm_format) for item_id, pcm in music.items()}

    async def _item_stream(
        queue_item: SimpleNamespace, *_args: object, **_kwargs: object
    ) -> AsyncGenerator[bytes]:
        for chunk in chunks[queue_item.queue_item_id]:
            yield chunk
            await asyncio.sleep(0)

    monkeypatch.setattr(audio, "get_queue_item_stream", _item_stream)
    return audio, queue, mass, music, first, second


def _flow(
    audio: StreamsAudio,
    queue: SimpleNamespace,
    start: SimpleNamespace,
    pcm_format: AudioFormat = TEST_PCM_FORMAT,
) -> Any:
    """Open the flow stream for the queue, starting at the given item."""
    return audio.get_queue_flow_stream(
        cast("Any", queue), cast("Any", start), pcm_format, session_id="session-1"
    )


def _place_at(
    offset_seconds: float, voice: bytes, *, stop_frame: int | None = None
) -> Callable[[_FlowProgramBus], int]:
    """Place the program a given time ahead of the produced frame, returning that frame."""
    offset = int(offset_seconds * SECOND)

    def _place(bus: _FlowProgramBus) -> int:
        anchor = bus.produced_frames + offset
        assert bus.schedule(voice, anchor, label="announcement", stop_frame=stop_frame) == anchor
        return anchor

    return _place


async def _drain(
    stream: AsyncGenerator[bytes],
    published: list[_FlowProgramBus],
    *,
    place: Callable[[_FlowProgramBus], int | None] | None = None,
    at_seconds: float = 0.0,
    pcm_format: AudioFormat = TEST_PCM_FORMAT,
) -> tuple[list[bytes], int | None]:
    """
    Collect a flow stream, placing a program on its bus once it produced enough audio.

    :param stream: The flow stream to collect.
    :param published: Buses the session published, in the order it opened them.
    :param place: Places a program on the session's bus; returns the frame it was anchored at.
    :param at_seconds: Audio the session has to have produced before the program is placed.
    :param pcm_format: PCM format of the stream.
    """
    chunks: list[bytes] = []
    produced = 0
    anchor: int | None = None
    threshold = int(at_seconds * pcm_format.pcm_sample_size)
    async for chunk in stream:
        chunks.append(chunk)
        produced += len(chunk)
        if place is not None and anchor is None and published and produced >= threshold:
            anchor = place(published[0])
        await asyncio.sleep(0)
    return chunks, anchor


def _publish(mass: MagicMock) -> list[_FlowProgramBus]:
    """Record the bus every flow session publishes for the queue."""
    published: list[_FlowProgramBus] = []
    mass.streams.register_program_bus.side_effect = lambda _queue_id, program_bus: published.append(
        program_bus
    )
    return published


async def _reference_session(monkeypatch: pytest.MonkeyPatch) -> bytes:
    """Play the same two items without the bus, as the reference the output is held against."""
    reference, queue, _mass, _music_pcm, first, _second = _two_item_flow(
        monkeypatch, program_bus=False
    )
    chunks, _anchor = await _drain(_flow(reference, queue, first), [])
    return b"".join(chunks)


def _reference_sum(
    music: bytes,
    voice: bytes,
    pcm_format: AudioFormat,
    *,
    start_frame: int,
    stop_frame: int | None = None,
) -> bytes:
    """
    Sum the program over the music with the bus's formula, at frame resolution.

    This is the reference the flow's output is compared against byte for byte: every
    frame of the window is ducked and has the program added, everything else is untouched.
    """
    frame_size = _frame_size(pcm_format)
    float_pcm = pcm_format.content_type == ContentType.PCM_F32LE
    code = "f" if float_pcm else "h"
    unpack = struct.Struct(f"<{pcm_format.channels}{code}").unpack
    pack = struct.Struct(f"<{pcm_format.channels}{code}").pack
    ramp = max(1, int(VOICE_OVER_RAMP * pcm_format.sample_rate))
    voice_frames = len(voice) // frame_size
    duck_head = min(ramp, max(1, voice_frames))
    ceiling = 10 ** (MIX_CEILING_DB / 20)
    voice_end = start_frame + voice_frames
    end_frame = voice_end + ramp
    if stop_frame is not None:
        end_frame = max(start_frame, min(end_frame, stop_frame))
    out = bytearray(music)
    for position in range(start_frame, min(end_frame, len(music) // frame_size)):
        offset = position * frame_size
        voice_offset = -1
        if position < voice_end:
            ducked = VOICE_OVER_DUCK_DEPTH * min(1.0, (position - start_frame + 1) / duck_head)
            voice_offset = (position - start_frame) * frame_size
        else:
            ducked = VOICE_OVER_DUCK_DEPTH * max(0.0, (end_frame - position) / ramp)
        gain = 1.0 - ducked
        samples = list(unpack(music[offset : offset + frame_size]))
        if 0 <= voice_offset <= len(voice) - frame_size:
            spoken = unpack(voice[voice_offset : voice_offset + frame_size])
            if float_pcm:
                samples = [
                    max(-ceiling, min(ceiling, value * gain + extra))
                    for value, extra in zip(samples, spoken, strict=True)
                ]
            else:
                samples = [
                    max(-32768, min(32767, int(value * gain) + extra))
                    for value, extra in zip(samples, spoken, strict=True)
                ]
        elif float_pcm:
            samples = [max(-ceiling, min(ceiling, value * gain)) for value in samples]
        else:
            samples = [max(-32768, min(32767, int(value * gain))) for value in samples]
        out[offset : offset + frame_size] = pack(*samples)
    return bytes(out)


def _frame_pcm(pcm: bytes, frame: int, pcm_format: AudioFormat) -> bytes:
    """Return one frame of the PCM."""
    frame_size = _frame_size(pcm_format)
    return pcm[frame * frame_size : (frame + 1) * frame_size]


def _channel(pcm: bytes, pcm_format: AudioFormat) -> int | float:
    """Return the first channel's sample of a single frame."""
    if pcm_format.content_type == ContentType.PCM_F32LE:
        return cast("float", struct.unpack("<f", pcm[:4])[0])
    return cast("int", struct.unpack("<h", pcm[:2])[0])


async def test_the_switch_off_keeps_the_bus_out_of_the_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the experiment the items play untouched, chunk for chunk as they come."""
    audio, queue, mass, music, first, _second = _two_item_flow(monkeypatch, program_bus=False)

    chunks, _anchor = await _drain(_flow(audio, queue, first), [])

    # neither a bus nor anything it touched took part
    mass.streams.register_program_bus.assert_not_called()
    mass.streams.release_program_bus.assert_not_called()
    cast("Any", mass.streams.get_config_value).assert_any_call(
        CONF_STREAM_PROGRAM_BUS, False, return_type=bool
    )
    assert b"".join(chunks) == music["item-1"] + music["item-2"]
    assert chunks == _chunks_of(music["item-1"], TEST_PCM_FORMAT) + _chunks_of(
        music["item-2"], TEST_PCM_FORMAT
    )


async def test_an_overlay_changes_nothing_outside_its_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flow keeps its exact length and chunking, and is untouched around the program."""
    reference_pcm = await _reference_session(monkeypatch)
    audio, queue, mass, _music_pcm, first, _second = _two_item_flow(monkeypatch, program_bus=True)
    published = _publish(mass)
    voice = _voice(2.0)

    chunks, anchor = await _drain(
        _flow(audio, queue, first),
        published,
        place=_place_at(2.0, voice),
        at_seconds=3.0,
    )
    output = b"".join(chunks)
    assert anchor is not None
    frame_size = _frame_size(TEST_PCM_FORMAT)
    end_frame = anchor + len(voice) // frame_size + int(VOICE_OVER_RAMP * SECOND)

    # same total length, same chunk boundaries: nothing was added, split or joined
    assert len(output) == len(reference_pcm)
    assert [len(chunk) for chunk in chunks] == [
        len(chunk) for chunk in _chunks_of(reference_pcm, TEST_PCM_FORMAT)
    ]
    # byte-identical up to the anchor ...
    assert output[: anchor * frame_size] == reference_pcm[: anchor * frame_size]
    # ... and from the end of the program's ramp back to the music
    assert output[end_frame * frame_size :] == reference_pcm[end_frame * frame_size :]
    # while the program really is heard inside its window
    assert (
        output[anchor * frame_size : end_frame * frame_size]
        != (reference_pcm[anchor * frame_size : end_frame * frame_size])
    )


async def test_the_window_is_exactly_the_ducked_music_plus_the_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every frame in the window is the music at its ducked level with the voice added."""
    reference_pcm = await _reference_session(monkeypatch)
    audio, queue, mass, _music_pcm, first, _second = _two_item_flow(monkeypatch, program_bus=True)
    published = _publish(mass)
    voice = _voice(2.0)

    chunks, anchor = await _drain(
        _flow(audio, queue, first),
        published,
        place=_place_at(2.0, voice),
        at_seconds=3.0,
    )
    output = b"".join(chunks)

    assert anchor is not None
    # frame-exact: the whole session, not just the window, matches the reference sum
    assert output == _reference_sum(reference_pcm, voice, TEST_PCM_FORMAT, start_frame=anchor)
    # one frame under the voice is the music at full duck with the voice added
    frame = anchor + int(VOICE_OVER_RAMP * SECOND) + 100
    music_level = _channel(_frame_pcm(reference_pcm, frame, TEST_PCM_FORMAT), TEST_PCM_FORMAT)
    voice_level = _channel(_frame_pcm(voice, frame - anchor, TEST_PCM_FORMAT), TEST_PCM_FORMAT)
    assert _channel(_frame_pcm(output, frame, TEST_PCM_FORMAT), TEST_PCM_FORMAT) == (
        int(music_level * (1 - VOICE_OVER_DUCK_DEPTH)) + voice_level
    )


async def test_a_float_flow_never_leaves_full_scale_under_the_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session that mixes programs runs in float, and its sum is held at the ceiling."""
    audio, queue, mass, music, first, _second = _two_item_flow(
        monkeypatch, program_bus=True, pcm_format=F32_PCM_FORMAT
    )
    published = _publish(mass)
    # a program as loud as the music: the sum would pass full scale without a limiter
    voice = struct.pack("<ff", 0.9, 0.9) * 2 * SECOND
    reference_pcm = music["item-1"] + music["item-2"]
    frame_size = _frame_size(F32_PCM_FORMAT)

    chunks, anchor = await _drain(
        _flow(audio, queue, first, F32_PCM_FORMAT),
        published,
        place=_place_at(2.0, voice),
        at_seconds=3.0,
        pcm_format=F32_PCM_FORMAT,
    )
    output = b"".join(chunks)

    assert anchor is not None
    assert published[0].pcm_format is F32_PCM_FORMAT
    assert published[0].can_mix
    # no sample leaves 0 dBFS, and the sum is held at the limiter's ceiling
    samples = np.frombuffer(output, dtype=np.float32)
    assert np.isfinite(samples).all()
    assert float(np.max(np.abs(samples))) <= 1.0
    window = np.frombuffer(
        output[anchor * frame_size : (anchor + 2 * SECOND) * frame_size], dtype=np.float32
    )
    assert float(np.max(np.abs(window))) == pytest.approx(10 ** (MIX_CEILING_DB / 20), abs=1e-6)
    # the length and the chunking stay the session's own
    assert len(output) == len(reference_pcm)
    assert [len(chunk) for chunk in chunks] == [
        len(chunk) for chunk in _chunks_of(reference_pcm, F32_PCM_FORMAT)
    ]
    # and the exact sum is the reference, frame for frame
    assert output == _reference_sum(reference_pcm, voice, F32_PCM_FORMAT, start_frame=anchor)


async def test_a_program_ends_with_the_item_it_was_placed_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A program that outlasts its item stops on the item's last frame, not in the next one."""
    reference_pcm = await _reference_session(monkeypatch)
    audio, queue, mass, _music_pcm, first, _second = _two_item_flow(monkeypatch, program_bus=True)
    published = _publish(mass)
    # longer than what is left of the item it starts in
    voice = _voice(6.0)

    chunks, anchor = await _drain(
        _flow(audio, queue, first),
        published,
        place=_place_at(2.0, voice, stop_frame=FIRST_SECONDS * SECOND),
        at_seconds=8.0,
    )
    output = b"".join(chunks)
    frame_size = _frame_size(TEST_PCM_FORMAT)
    item_end = FIRST_SECONDS * SECOND

    assert anchor is not None
    # the item keeps its full length and the total is unchanged
    assert len(output) == len(reference_pcm)
    assert [len(chunk) for chunk in chunks] == [
        len(chunk) for chunk in _chunks_of(reference_pcm, TEST_PCM_FORMAT)
    ]
    # the program is heard to the item's last frame ...
    assert (
        output[(item_end - 1) * frame_size : item_end * frame_size]
        != (reference_pcm[(item_end - 1) * frame_size : item_end * frame_size])
    )
    # ... and not one byte of it sounds over the item after it
    assert output[item_end * frame_size :] == reference_pcm[item_end * frame_size :]
    # the whole session is the reference sum bounded by the item's end
    assert output == _reference_sum(
        reference_pcm, voice, TEST_PCM_FORMAT, start_frame=anchor, stop_frame=item_end
    )


async def test_a_program_without_a_stop_frame_rides_over_the_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A program that was not planned against an item (an announcement) is heard in full."""
    reference_pcm = await _reference_session(monkeypatch)
    audio, queue, mass, _music_pcm, first, _second = _two_item_flow(monkeypatch, program_bus=True)
    published = _publish(mass)
    voice = _voice(6.0)

    chunks, anchor = await _drain(
        _flow(audio, queue, first),
        published,
        place=_place_at(2.0, voice),
        at_seconds=8.0,
    )
    output = b"".join(chunks)
    frame_size = _frame_size(TEST_PCM_FORMAT)
    item_end = FIRST_SECONDS * SECOND

    assert anchor is not None
    # the item keeps its full length and the total is unchanged
    assert len(output) == len(reference_pcm)
    assert [len(chunk) for chunk in chunks] == [
        len(chunk) for chunk in _chunks_of(reference_pcm, TEST_PCM_FORMAT)
    ]
    # the session's frame count carries on across the item boundary
    assert output[item_end * frame_size :] != reference_pcm[item_end * frame_size :]
    assert output == _reference_sum(reference_pcm, voice, TEST_PCM_FORMAT, start_frame=anchor)


async def test_an_anchor_in_the_past_is_clamped_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frame that already went by starts at the flow's current frame and is logged late."""
    reference_pcm = await _reference_session(monkeypatch)
    audio, queue, mass, _music_pcm, first, _second = _two_item_flow(monkeypatch, program_bus=True)
    logger = MagicMock()
    audio.logger = logger
    published = _publish(mass)
    voice = _voice(2.0)
    frame_size = _frame_size(TEST_PCM_FORMAT)
    requested: list[int] = []

    def _place(bus: _FlowProgramBus) -> int | None:
        requested.append(bus.produced_frames - SECOND)
        return bus.schedule(voice, requested[-1], label="announcement")

    chunks, anchor = await _drain(
        _flow(audio, queue, first),
        published,
        place=_place,
        at_seconds=3.0,
    )
    output = b"".join(chunks)

    assert anchor is not None
    # the frame it asked for had already been produced; the program starts where the flow is
    assert requested[0] < anchor
    assert anchor == requested[0] + SECOND
    # the first second of the voice was dropped, and it is reported
    assert any(
        call.args[0] == "PROGRAM-BUS schedule label=%s started late by %.3fs"
        and call.args[2] == pytest.approx(1.0)
        for call in logger.debug.call_args_list
    )
    # the audio before the clamp is untouched; from there the trimmed program sums in
    assert output[: anchor * frame_size] == reference_pcm[: anchor * frame_size]
    assert output == _reference_sum(
        reference_pcm,
        voice[SECOND * frame_size :],
        TEST_PCM_FORMAT,
        start_frame=anchor,
    )


async def test_an_overlay_is_placed_through_the_session_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The controller's overlay accessor finds the bus the running session published."""
    controller: Any = StreamsController.__new__(StreamsController)
    controller._program_buses = {}
    controller.logger = MagicMock()
    reference_pcm = await _reference_session(monkeypatch)
    audio, queue, mass, _music_pcm, first, _second = _two_item_flow(monkeypatch, program_bus=True)
    voice = _voice(2.0)
    frame_size = _frame_size(TEST_PCM_FORMAT)
    published: list[_FlowProgramBus] = []

    def _register(queue_id: str, program_bus: _FlowProgramBus) -> None:
        published.append(program_bus)
        controller.register_program_bus(queue_id, program_bus)

    mass.streams.register_program_bus.side_effect = _register
    mass.streams.release_program_bus.side_effect = controller.release_program_bus
    scheduled: list[bool] = []
    expected_anchor: list[int] = []

    def _place(bus: _FlowProgramBus) -> int | None:
        assert controller.program_bus_active("queue-1")
        assert controller.program_bus_format("queue-1") is TEST_PCM_FORMAT
        expected_anchor.append(bus.produced_frames + int(LEAD_MIN_LEAD_SECONDS * SECOND))
        scheduled.append(controller.schedule_overlay("queue-1", voice, label="announcement"))
        return expected_anchor[-1]

    chunks, anchor = await _drain(
        _flow(audio, queue, first),
        published,
        place=_place,
        at_seconds=3.0,
    )
    output = b"".join(chunks)

    assert scheduled == [True]
    assert anchor == expected_anchor[0]
    # the session withdrew its bus when it ended
    assert controller.program_bus_active("queue-1") is False
    # the overlay is heard from the lead the controller placed it at
    first_changed = next(
        index
        for index in range(len(reference_pcm) // frame_size)
        if _frame_pcm(output, index, TEST_PCM_FORMAT)
        != _frame_pcm(reference_pcm, index, TEST_PCM_FORMAT)
    )
    assert first_changed == anchor
    assert output == _reference_sum(reference_pcm, voice, TEST_PCM_FORMAT, start_frame=anchor)

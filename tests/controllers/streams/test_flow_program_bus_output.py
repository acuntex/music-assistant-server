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
from music_assistant_models.enums import ContentType, CrossfadeMode, PlaybackState
from music_assistant_models.errors import QueueEmpty
from music_assistant_models.media_items import AudioFormat

from music_assistant.controllers.streams.audio import FlowProgramBus, StreamsAudio
from music_assistant.controllers.streams.constants import (
    CONF_STREAM_PROGRAM_BUS,
    LEAD_MIN_LEAD_SECONDS,
    VOICE_OVER_RAMP,
)
from music_assistant.controllers.streams.program_schedule import ProgramKind
from music_assistant.controllers.streams.smart_fades.filters import (
    MIX_CEILING_DB,
    VOICE_OVER_DUCK_DEPTH,
)
from tests.controllers.streams.program_bus_helpers import bare_streams_controller
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
# a chunk size that cuts into the flow's frames: 1002 bytes is 250 frames and a half
ODD_CHUNK_BYTES = 1002
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


def _chunks_by_size(pcm: bytes, size: int) -> list[bytes]:
    """Split PCM into chunks of a fixed size, whatever it does to the frames."""
    return [pcm[start : start + size] for start in range(0, len(pcm), size)]


def _two_item_flow(
    monkeypatch: pytest.MonkeyPatch,
    *,
    program_bus: bool,
    pcm_format: AudioFormat = TEST_PCM_FORMAT,
    crossfade_mode: CrossfadeMode = CrossfadeMode.DISABLED,
    second_seconds: int = SECOND_SECONDS,
    chunk_bytes: int | None = None,
) -> tuple[
    StreamsAudio,
    SimpleNamespace,
    MagicMock,
    dict[str, bytes],
    SimpleNamespace,
    SimpleNamespace,
]:
    """
    Build a flow of two tracks, marking their audio.

    :param program_bus: Whether the experimental program bus is switched on.
    :param pcm_format: PCM format the items and the flow use.
    :param crossfade_mode: The queue's crossfade; DISABLED cuts at the boundary.
    :param second_seconds: Length of the item after the boundary.
    :param chunk_bytes: Size the item streams deliver their audio in, or None for whole
        frames as a source would deliver them.
    """
    first = _queue_item("item-1", "First", duration=FIRST_SECONDS)
    second = _queue_item("item-2", "Second", duration=second_seconds)
    audio, queue, mass = _flow_audio(
        monkeypatch,
        next_item=second,
        load_next=[second, QueueEmpty],
        crossfade_mode=crossfade_mode,
        program_bus=program_bus,
    )
    audio.logger = MagicMock()
    music = {
        "item-1": _music(FIRST_SECONDS, pcm_format),
        "item-2": _music(second_seconds, pcm_format),
    }
    chunks = {
        item_id: (
            _chunks_of(pcm, pcm_format)
            if chunk_bytes is None
            else _chunks_by_size(pcm, chunk_bytes)
        )
        for item_id, pcm in music.items()
    }

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
) -> Callable[[FlowProgramBus], int]:
    """Place the program a given time ahead of the produced frame, returning that frame."""
    offset = int(offset_seconds * SECOND)

    def _place(bus: FlowProgramBus) -> int:
        anchor = bus.produced_frames + offset
        assert bus.schedule(voice, anchor, label="announcement", stop_frame=stop_frame) == anchor
        return anchor

    return _place


async def _drain(
    stream: AsyncGenerator[bytes],
    published: list[FlowProgramBus],
    *,
    place: Callable[[FlowProgramBus], int | None] | None = None,
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


def _publish(mass: MagicMock) -> list[FlowProgramBus]:
    """Record the bus every flow session publishes for the queue."""
    published: list[FlowProgramBus] = []
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


async def test_chunks_that_end_mid_frame_emit_the_same_bytes_through_the_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A chunk that ends inside a frame is carried over, not cut off, at the bus.

    The item streams deliver 1002 bytes at a time, which is 250 frames and a half in the
    flow's format. A bus that dropped those two bytes would shorten the stream and shift
    every frame behind the first drop, so the player would read sample halves as frames for
    the rest of the session: with the switch on the flow has to emit the bytes the switch
    off emits, whole and in the same order.
    """
    off_audio, off_queue, _off_mass, _off_music, off_first, _second = _two_item_flow(
        monkeypatch, program_bus=False, chunk_bytes=ODD_CHUNK_BYTES
    )
    reference_chunks, _anchor = await _drain(_flow(off_audio, off_queue, off_first), [])
    audio, queue, mass, music, first, _second = _two_item_flow(
        monkeypatch, program_bus=True, chunk_bytes=ODD_CHUNK_BYTES
    )
    published = _publish(mass)

    chunks, _anchor = await _drain(_flow(audio, queue, first), published)

    output = b"".join(chunks)
    # the item streams really did cut into the frames
    assert len(_chunks_by_size(music["item-1"], ODD_CHUNK_BYTES)[0]) % _frame_size(TEST_PCM_FORMAT)
    # nothing lost, nothing added, nothing moved
    assert output == b"".join(reference_chunks)
    assert output == music["item-1"] + music["item-2"]
    # and the bus counted exactly the frames it let out of the session
    assert len(published) == 1
    assert published[0].produced_frames == len(output) // _frame_size(TEST_PCM_FORMAT)


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


async def test_a_session_without_a_boundary_keeps_its_bytes_and_its_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a blend the bus counts and places exactly as it did before the change."""
    reference_pcm = await _reference_session(monkeypatch)
    audio, queue, mass, _music_pcm, first, _second = _two_item_flow(monkeypatch, program_bus=True)
    published = _publish(mass)
    voice = _voice(2.0)
    frame_size = _frame_size(TEST_PCM_FORMAT)

    chunks, anchor = await _drain(
        _flow(audio, queue, first),
        published,
        place=_place_at(2.0, voice),
        at_seconds=3.0,
    )
    output = b"".join(chunks)

    # the placement is the emitted frame it always was on a session that cuts at its boundary
    assert anchor == 5 * SECOND
    # the bus counted every frame it emitted, and none of them changed the session's length
    assert published[0].produced_frames == len(output) // frame_size == 20 * SECOND
    # the output is the untouched session with only the program summed over its window
    assert output == _reference_sum(reference_pcm, voice, TEST_PCM_FORMAT, start_frame=anchor)


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

    def _place(bus: FlowProgramBus) -> int | None:
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
    """The accessor finds the published bus and places the program where the player is."""
    controller: Any = bare_streams_controller()
    reference_pcm = await _reference_session(monkeypatch)
    audio, queue, mass, _music_pcm, first, _second = _two_item_flow(monkeypatch, program_bus=True)
    controller.mass = mass
    # the player has played the first 1.5 s of the stream when the program is requested
    heard = 1.5
    player = mass.players.get_player.return_value
    player.state.playback_state = PlaybackState.PLAYING
    player.state.corrected_elapsed_time = heard
    voice = _voice(2.0)
    frame_size = _frame_size(TEST_PCM_FORMAT)
    published: list[FlowProgramBus] = []

    def _register(queue_id: str, program_bus: FlowProgramBus) -> None:
        published.append(program_bus)
        controller.register_program_bus(queue_id, program_bus)

    mass.streams.register_program_bus.side_effect = _register
    mass.streams.release_program_bus.side_effect = controller.release_program_bus
    scheduled: list[bool] = []
    expected_anchor: list[int] = []

    def _place(bus: FlowProgramBus) -> int | None:
        assert controller.program_bus_active("queue-1")
        assert controller.program_bus_format("queue-1") is TEST_PCM_FORMAT
        # a minimum lead after what the player heard, which is behind what the session
        # already produced
        expected_anchor.append(int(heard * SECOND) + int(LEAD_MIN_LEAD_SECONDS * SECOND))
        assert expected_anchor[-1] < bus.produced_frames + int(LEAD_MIN_LEAD_SECONDS * SECOND)
        scheduled.append(
            controller.schedule_overlay(
                "queue-1", voice, label="announcement", kind=ProgramKind.ANNOUNCEMENT
            ).scheduled
        )
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


CROSSFADE_VOICE_SECONDS = 8.0


async def test_an_overlay_riding_over_a_crossfade_boundary_keeps_the_boundary_blend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A program mixed across a boundary leaves the crossfade under it untouched.

    The bus tests run with the queue's crossfade off, where every boundary is a cut. With
    a crossfade on, the boundary's own blend happens while the announcement is heard over
    it: the session keeps its length and chunking, the blend is still built, and the whole
    output is the crossfaded session with the program summed over it, frame for frame.
    """
    reference_audio, reference_queue, _reference_mass, _music_pcm, reference_first, _ = (
        _two_item_flow(
            monkeypatch, program_bus=False, crossfade_mode=CrossfadeMode.STANDARD_CROSSFADE
        )
    )
    reference_chunks, _reference_anchor = await _drain(
        _flow(reference_audio, reference_queue, reference_first), []
    )
    reference_pcm = b"".join(reference_chunks)
    audio, queue, mass, _music_pcm, first, _second = _two_item_flow(
        monkeypatch, program_bus=True, crossfade_mode=CrossfadeMode.STANDARD_CROSSFADE
    )
    published = _publish(mass)
    voice = _voice(CROSSFADE_VOICE_SECONDS)
    frame_size = _frame_size(TEST_PCM_FORMAT)
    ramp = int(VOICE_OVER_RAMP * SECOND)

    # placed while the first track still streams, so the voice is heard on both sides of
    # the boundary at FIRST_SECONDS
    chunks, anchor = await _drain(
        _flow(audio, queue, first),
        published,
        place=_place_at(0.0, voice),
        at_seconds=FIRST_SECONDS - 7.0,
    )
    output = b"".join(chunks)

    assert anchor is not None
    boundary = FIRST_SECONDS * SECOND
    voice_end = anchor + len(voice) // frame_size
    end_frame = voice_end + ramp
    assert anchor < boundary < voice_end
    # the boundary really blended: the crossfade was built for it
    build = cast("Any", audio.smart_fades_mixer.build).await_args
    assert build is not None
    assert build.kwargs["mode"] == CrossfadeMode.STANDARD_CROSSFADE
    assert len(build.kwargs["fade_out_data"]) > 0
    # the session keeps its exact length and chunking
    assert len(output) == len(reference_pcm)
    assert [len(chunk) for chunk in chunks] == [len(chunk) for chunk in reference_chunks]
    # and it is the crossfaded session with the program summed over it, frame for frame
    assert output == _reference_sum(reference_pcm, voice, TEST_PCM_FORMAT, start_frame=anchor)
    # the voice is heard on the track before the boundary and on the one after it ...
    assert (
        output[anchor * frame_size : boundary * frame_size]
        != reference_pcm[anchor * frame_size : boundary * frame_size]
    )
    assert (
        output[boundary * frame_size : voice_end * frame_size]
        != reference_pcm[boundary * frame_size : voice_end * frame_size]
    )
    # ... and it ends with its ramp back to the music, leaving the rest of the fade alone
    assert output[end_frame * frame_size :] == reference_pcm[end_frame * frame_size :]
    assert not published[0].active


async def _overlap_mix(
    _smart_fade: object,
    *,
    fade_in_part: bytes | AsyncGenerator[bytes],
    fade_out_part: bytes,
    **_kwargs: object,
) -> AsyncGenerator[bytes]:
    """Blend the two sides the way a real crossfade does: the overlap is merged, not added."""
    if isinstance(fade_in_part, bytes):
        incoming = fade_in_part
    else:
        incoming = b"".join([chunk async for chunk in fade_in_part])
    # the incoming side is what the crossfade overlaps; the output keeps the outgoing tail's
    # length, so the blended incoming bytes never become media time of their own
    assert incoming
    yield fade_out_part


async def test_a_program_after_a_crossfade_boundary_is_anchored_in_emitted_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A program scheduled past a boundary lands on the frame it was placed at in the output.

    The blend overlaps the incoming item's head with the outgoing item's tail, so the session
    emits fewer frames than it reads from the two item streams. The bus has to count the
    frames it emits, not the frames it reads, or the program is anchored short by the merged
    overlap and sounds before the frame it was placed at.
    """
    reference_audio, reference_queue, _reference_mass, _music_pcm, reference_first, _ = (
        _two_item_flow(
            monkeypatch,
            program_bus=False,
            crossfade_mode=CrossfadeMode.STANDARD_CROSSFADE,
            second_seconds=SECOND_SECONDS * 4,
        )
    )
    monkeypatch.setattr(reference_audio.smart_fades_mixer, "mix", _overlap_mix)
    reference_chunks, _reference_anchor = await _drain(
        _flow(reference_audio, reference_queue, reference_first), []
    )
    reference_pcm = b"".join(reference_chunks)
    audio, queue, mass, _music_pcm, first, _second = _two_item_flow(
        monkeypatch,
        program_bus=True,
        crossfade_mode=CrossfadeMode.STANDARD_CROSSFADE,
        second_seconds=SECOND_SECONDS * 4,
    )
    monkeypatch.setattr(audio.smart_fades_mixer, "mix", _overlap_mix)
    published = _publish(mass)
    voice = _voice(2.0)
    frame_size = _frame_size(TEST_PCM_FORMAT)
    boundary = FIRST_SECONDS * SECOND
    place_at = boundary + 2 * SECOND
    offset = SECOND

    chunks: list[bytes] = []
    emitted = 0
    anchor: int | None = None
    counted_at_place: int | None = None
    emitted_at_place: int | None = None
    async for chunk in _flow(audio, queue, first):
        chunks.append(chunk)
        emitted += len(chunk)
        if anchor is None and emitted >= place_at * frame_size:
            # the bus has just carried every emitted byte, so its count is the output's
            counted_at_place = published[0].produced_frames
            emitted_at_place = emitted // frame_size
            anchor = published[0].produced_frames + offset
            assert published[0].schedule(voice, anchor, label="announcement") == anchor
        await asyncio.sleep(0)
    output = b"".join(chunks)

    assert anchor is not None
    assert counted_at_place == emitted_at_place
    # the boundary really merged the incoming overlap: the session is shorter than its items
    assert len(output) < len(_music_pcm["item-1"]) + len(_music_pcm["item-2"])
    assert anchor > boundary
    # the program is heard from the emitted frame it was anchored at, not the fed one
    assert output == _reference_sum(reference_pcm, voice, TEST_PCM_FORMAT, start_frame=anchor)
    first_changed = next(
        index
        for index in range(len(reference_pcm) // frame_size)
        if _frame_pcm(output, index, TEST_PCM_FORMAT)
        != _frame_pcm(reference_pcm, index, TEST_PCM_FORMAT)
    )
    assert first_changed == anchor
    assert not published[0].active

"""
Identity, credit and cancellation of the program bus, through the real flow.

The flow runs for real (``get_queue_flow_stream`` with its own item streams and the bus
switched on); only the item source is served by the test. A covered clip is held against
a frame-by-frame reference of the two tracks, so a hole at the boundary, a clip that is
served as an item, or a position that lands on somebody else's seconds shows up in the
comparison. Cancelling a program goes through the same queue commands the player sends,
and the rest of the session is held against a run without the program, byte for byte.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from music_assistant_models.enums import (
    CrossfadeMode,
    MediaType,
    PlaybackState,
)
from music_assistant_models.errors import QueueEmpty
from music_assistant_models.player_queue import PlayerQueue
from music_assistant_models.queue_item import QueueItem
from music_assistant_models.streamdetails import TailOverlap

from music_assistant.controllers.player_queues.controller import PlayerQueuesController
from music_assistant.controllers.player_queues.state import PlayerQueueData
from music_assistant.controllers.streams import StreamsController
from music_assistant.controllers.streams.audio import StreamsAudio, _FlowProgramBus
from music_assistant.controllers.streams.constants import (
    PRE_OVERLAP_ATTR,
    PRE_OVERLAP_FROM_ATTR,
)
from tests.controllers.streams.test_crossfade_transition import (
    TEST_PCM_FORMAT,
    _flow_audio,
    _queue_item,
)
from tests.controllers.streams.test_flow_program_bus_output import (
    SECOND,
    _chunks_of,
    _drain,
    _flow,
    _frame_size,
    _music,
    _place_at,
    _publish,
    _reference_session,
    _reference_sum,
    _two_item_flow,
    _voice,
)

FRAME_SIZE = _frame_size(TEST_PCM_FORMAT)
TRACK_SECONDS = 40
CLIP_SECONDS = 4
HEAD_SECONDS = 2.0
POST_SECONDS = 2.0
AFTER_SECONDS = 8
# how long the program has been heard when a cancel command is sent
CANCEL_AFTER_SECONDS = 0.5


def _covered_clip() -> SimpleNamespace:
    """Build a clip whose head and post overlap cover all of it."""
    clip = _queue_item("item-2", "Transition", duration=CLIP_SECONDS)
    clip.media_type = MediaType.SOUND_EFFECT
    clip.extra_attributes[PRE_OVERLAP_ATTR] = HEAD_SECONDS
    clip.extra_attributes[PRE_OVERLAP_FROM_ATTR] = "item-1"
    clip.streamdetails.tail_overlap = TailOverlap(
        duration=POST_SECONDS, next_queue_item_id="item-3"
    )
    return clip


def _summing_mix(monkeypatch: pytest.MonkeyPatch, audio: StreamsAudio) -> list[tuple[bytes, bytes]]:
    """Replace the mixer with the shape a voice over has: the music side keeps its length."""
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
        # the carried clip is the shorter side, so the track after it owns the output
        yield fade_out_part if len(fade_out_part) >= len(fade_in_bytes) else fade_in_bytes

    monkeypatch.setattr(audio.smart_fades_mixer, "mix", _mix)
    return mixes


def _covered_clip_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    StreamsAudio,
    SimpleNamespace,
    MagicMock,
    SimpleNamespace,
    SimpleNamespace,
    SimpleNamespace,
    dict[str, bytes],
    list[tuple[bytes, bytes]],
    list[tuple[str, int]],
]:
    """Build a flow of track -> covered clip -> track with real music and the clip's voice."""
    track = _queue_item("item-1", "Track", duration=TRACK_SECONDS)
    clip = _covered_clip()
    after = _queue_item("item-3", "After", duration=AFTER_SECONDS)
    audio, queue, mass = _flow_audio(
        monkeypatch,
        next_item=clip,
        load_next=[clip, after, QueueEmpty],
        crossfade_mode=CrossfadeMode.DISABLED,
        program_bus=True,
        build_result=SimpleNamespace(
            timing_info=SimpleNamespace(
                fadein_trimmed_duration=0.0,
                crossfade_duration=POST_SECONDS,
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
    pcm = {
        "item-1": _music(TRACK_SECONDS),
        "item-2": _voice(CLIP_SECONDS),
        "item-3": _music(AFTER_SECONDS),
    }
    opened: list[tuple[str, int]] = []

    async def _item_stream(
        queue_item: SimpleNamespace, *_args: object, **kwargs: object
    ) -> AsyncGenerator[bytes]:
        item_id = queue_item.queue_item_id
        seek = int(cast("float", kwargs.get("seek_position", 0)))
        opened.append((item_id, seek))
        body = pcm[item_id][seek * SECOND * FRAME_SIZE :]
        for chunk in _chunks_of(body, TEST_PCM_FORMAT):
            yield chunk
            await asyncio.sleep(0)

    monkeypatch.setattr(audio, "get_queue_item_stream", _item_stream)
    return audio, queue, mass, track, clip, after, pcm, mixes, opened


def _clip_between_tracks_flow(
    monkeypatch: pytest.MonkeyPatch, *, delivered: dict[str, int]
) -> tuple[
    StreamsAudio,
    SimpleNamespace,
    MagicMock,
    SimpleNamespace,
    SimpleNamespace,
    SimpleNamespace,
    dict[str, bytes],
]:
    """Build a flow of track -> clip -> track, serving each item up to the given second."""
    track = _queue_item("item-1", "Track", duration=TRACK_SECONDS)
    clip = _queue_item("item-2", "Transition", duration=CLIP_SECONDS)
    clip.media_type = MediaType.SOUND_EFFECT
    after = _queue_item("item-3", "After", duration=AFTER_SECONDS)
    audio, queue, mass = _flow_audio(
        monkeypatch,
        next_item=clip,
        load_next=[clip, after, QueueEmpty],
        crossfade_mode=CrossfadeMode.DISABLED,
        program_bus=True,
    )
    next_items = {"item-1": clip, "item-2": after}
    mass.player_queues.get_next_item.side_effect = lambda _queue_id, item_id: next_items.get(
        item_id
    )
    pcm = {
        "item-1": _music(TRACK_SECONDS),
        "item-2": _music(CLIP_SECONDS),
        "item-3": _music(AFTER_SECONDS),
    }

    async def _item_stream(
        queue_item: SimpleNamespace, *_args: object, **kwargs: object
    ) -> AsyncGenerator[bytes]:
        item_id = queue_item.queue_item_id
        seek = int(cast("float", kwargs.get("seek_position", 0)))
        end = delivered[item_id] * SECOND * FRAME_SIZE
        body = pcm[item_id][seek * SECOND * FRAME_SIZE : end]
        for chunk in _chunks_of(body, TEST_PCM_FORMAT):
            yield chunk
            await asyncio.sleep(0)

    monkeypatch.setattr(audio, "get_queue_item_stream", _item_stream)
    return audio, queue, mass, track, clip, after, pcm


def _wire_streams(mass: MagicMock) -> StreamsController:
    """Plug a real streams controller into the mocked mass, so overlays go through it."""
    controller: Any = StreamsController.__new__(StreamsController)
    controller._program_buses = {}
    controller.logger = MagicMock()
    mass.streams.register_program_bus = MagicMock(side_effect=controller.register_program_bus)
    mass.streams.release_program_bus = MagicMock(side_effect=controller.release_program_bus)
    mass.streams.cancel_overlays = MagicMock(side_effect=controller.cancel_overlays)
    mass.streams.schedule_overlay = MagicMock(side_effect=controller.schedule_overlay)
    mass.streams.program_bus_active = MagicMock(side_effect=controller.program_bus_active)
    return cast("StreamsController", controller)


def _publish_on(controller: StreamsController, mass: MagicMock) -> list[_FlowProgramBus]:
    """Record the bus every flow session publishes and register it on the controller."""
    published: list[_FlowProgramBus] = []
    register = controller.register_program_bus

    def _register(queue_id: str, program_bus: _FlowProgramBus) -> None:
        published.append(program_bus)
        register(queue_id, program_bus)

    cast("Any", mass.streams.register_program_bus).side_effect = _register
    return published


def _queue_controller(mass: MagicMock) -> PlayerQueuesController:
    """Build a bare queue controller playing the first of two items."""
    controller = PlayerQueuesController.__new__(PlayerQueuesController)
    queue = PlayerQueue(
        queue_id="queue-1", active=True, display_name="Queue", available=True, items=2
    )
    items = [
        QueueItem(queue_id="queue-1", queue_item_id="item-1", name="one", duration=TRACK_SECONDS),
        QueueItem(queue_id="queue-1", queue_item_id="item-2", name="two", duration=AFTER_SECONDS),
    ]
    queue.current_index = 0
    queue.current_item = items[0]
    queue.state = PlaybackState.PLAYING
    queue.elapsed_time = 1
    queue.elapsed_time_last_updated = time.time()
    queue_data = PlayerQueueData(queue=queue)
    queue_data.items = items
    controller._queue_data = {"queue-1": queue_data}
    controller.mass = mass
    mass.players.get_player.return_value.offline_by_power_control = False
    mass.players.get_player.return_value.extra_data = {}
    controller.signal_update = Mock()  # type: ignore[method-assign]
    controller.on_player_update = Mock()  # type: ignore[method-assign]
    return controller


def _current_index(
    mass: MagicMock, queue: SimpleNamespace, items: list[SimpleNamespace]
) -> Callable[[float], int | None]:
    """Return the queue index the flow log reports at a given stream time."""
    data = mass.player_queues.queue_data.return_value
    data.items = items
    queue.current_index = 0
    controller = PlayerQueuesController.__new__(PlayerQueuesController)
    controller._queue_data = {queue.queue_id: data}

    def _position(elapsed: float) -> int | None:
        player = SimpleNamespace(
            state=SimpleNamespace(
                corrected_elapsed_time=elapsed, playback_state=PlaybackState.PLAYING
            )
        )
        index, _track_time = controller._get_flow_queue_stream_index(
            cast("Any", queue), cast("Any", player)
        )
        return index

    return _position


def _cancel_reasons(audio: StreamsAudio) -> list[str]:
    """Return the reasons the bus logged for dropped programs."""
    return [
        cast("str", call.args[1])
        for call in cast("Any", audio.logger).debug.call_args_list
        if call.args[0] == "PROGRAM-BUS cancel reason=%s"
    ]


async def test_a_covered_clip_leaves_the_flow_the_two_tracks_long(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clip is heard over the boundary only: no hole, no item of its own."""
    audio, queue, mass, track, _clip, _after, pcm, mixes, _opened = _covered_clip_flow(monkeypatch)
    published = _publish(mass)

    chunks, _anchor = await _drain(_flow(audio, queue, track), published)

    output = b"".join(chunks)
    reference = pcm["item-1"] + pcm["item-3"]
    anchor = int((TRACK_SECONDS - HEAD_SECONDS) * SECOND)
    head = _voice(CLIP_SECONDS)[: int(HEAD_SECONDS * SECOND) * FRAME_SIZE]
    # the flow is exactly the two tracks: the clip's own seconds are not in it
    assert len(output) == (TRACK_SECONDS + AFTER_SECONDS) * SECOND * FRAME_SIZE
    # the track after the boundary starts on its very first frame, without a hole
    assert output[TRACK_SECONDS * SECOND * FRAME_SIZE :] == pcm["item-3"]
    # the only thing on the tracks is the clip's head, mixed over the outgoing tail
    assert output == _reference_sum(
        reference,
        head,
        TEST_PCM_FORMAT,
        start_frame=anchor,
        stop_frame=TRACK_SECONDS * SECOND,
    )
    # its remaining audio is the mixed side of the boundary, not an item served in between
    carried = len(_voice(CLIP_SECONDS)) - int(HEAD_SECONDS * SECOND) * FRAME_SIZE
    assert len(mixes) == 1
    assert mixes[0][0] == _voice(CLIP_SECONDS)[-carried:]
    assert len(mixes[0][1]) > len(mixes[0][0])
    flow_log = mass.player_queues.queue_data.return_value.flow_mode_stream_log
    # the two tracks are credited with exactly the audio the flow delivered for them
    assert [(entry.queue_item_id, entry.seconds_streamed) for entry in flow_log] == [
        ("item-1", TRACK_SECONDS),
        ("item-3", AFTER_SECONDS),
    ]


async def test_a_covered_clip_is_never_the_current_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The queue names the track before and after the boundary, never the clip."""
    audio, queue, mass, track, clip, after, _pcm, _mixes, opened = _covered_clip_flow(monkeypatch)
    published = _publish(mass)

    await _drain(_flow(audio, queue, track), published)

    flow_log = mass.player_queues.queue_data.return_value.flow_mode_stream_log
    # the clip is no entry of the play log at all
    assert [entry.queue_item_id for entry in flow_log] == ["item-1", "item-3"]
    # it was read once, as the clip whose head is armed, and never opened as an item
    assert [entry for entry in opened if entry[0] == "item-2"] == [("item-2", 0)]
    # the position is on the track before the boundary, then on the one after it
    position = _current_index(mass, queue, [track, clip, after])
    assert position(TRACK_SECONDS - 1.0) == 0
    assert position(TRACK_SECONDS + 0.5) == 2
    assert [position(elapsed) for elapsed in (39.0, 39.5, 40.5, 41.0)] == [0, 0, 2, 2]


async def test_the_track_after_a_clip_keeps_its_own_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clip's seconds belong to the clip; the track behind it keeps its length."""
    delivered = {
        "item-1": TRACK_SECONDS,
        "item-2": CLIP_SECONDS,
        "item-3": AFTER_SECONDS,
    }
    audio, queue, mass, track, clip, after, pcm = _clip_between_tracks_flow(
        monkeypatch, delivered=delivered
    )
    published = _publish(mass)

    chunks, _anchor = await _drain(_flow(audio, queue, track), published)

    # the output is what each item really delivered, in order and without a seam
    output = b"".join(chunks)
    assert output == pcm["item-1"] + pcm["item-2"] + pcm["item-3"]
    # the clip carries its own length, not the length of a track around it
    assert clip.duration == CLIP_SECONDS
    assert clip.streamdetails.duration == CLIP_SECONDS
    # and the track after it keeps its declared length
    assert after.duration == AFTER_SECONDS
    assert after.streamdetails.duration == AFTER_SECONDS
    assert after.streamdetails.seconds_streamed == AFTER_SECONDS
    flow_log = mass.player_queues.queue_data.return_value.flow_mode_stream_log
    assert [(entry.queue_item_id, entry.seconds_streamed) for entry in flow_log] == [
        ("item-1", TRACK_SECONDS),
        ("item-2", CLIP_SECONDS),
        ("item-3", AFTER_SECONDS),
    ]
    assert [entry.duration for entry in flow_log] == [TRACK_SECONDS, CLIP_SECONDS, AFTER_SECONDS]
    # what the play log credits is exactly the PCM that went out per item
    assert [
        int(cast("float", entry.seconds_streamed) * SECOND * FRAME_SIZE) for entry in flow_log
    ] == [len(pcm["item-1"]), len(pcm["item-2"]), len(pcm["item-3"])]


async def test_a_track_keeps_its_declared_length_when_its_source_stops_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source that ends mid-item must not hand the item its short streamed length."""
    delivered = {"item-1": TRACK_SECONDS, "item-2": CLIP_SECONDS, "item-3": 5}
    audio, queue, mass, track, _clip, after, pcm = _clip_between_tracks_flow(
        monkeypatch, delivered=delivered
    )
    published = _publish(mass)

    chunks, _anchor = await _drain(_flow(audio, queue, track), published)

    # exactly what reached the flow, with no padding and no skipped hole
    output = b"".join(chunks)
    assert output == pcm["item-1"] + pcm["item-2"] + pcm["item-3"][: 5 * SECOND * FRAME_SIZE]
    # the item keeps the length it declared, not the five seconds its source delivered
    assert after.duration == AFTER_SECONDS
    assert after.streamdetails.duration == AFTER_SECONDS
    # while the queue can still advance from where the audio really ended
    assert after.streamdetails.seconds_streamed == 5
    flow_log = mass.player_queues.queue_data.return_value.flow_mode_stream_log
    assert flow_log[-1].seconds_streamed == 5
    assert after.duration > after.streamdetails.seconds_streamed


@pytest.mark.parametrize("command", ["next", "play_index"])
async def test_a_queue_command_drops_the_program_while_it_is_heard(
    monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    """Moving on mid-program leaves nothing scheduled and the rest of the flow untouched."""
    reference_pcm = await _reference_session(monkeypatch)
    audio, queue, mass, _music_pcm, first, _second = _two_item_flow(monkeypatch, program_bus=True)
    controller = _wire_streams(mass)
    published = _publish_on(controller, mass)
    queue_controller = _queue_controller(mass)
    queue_controller._get_next_index = Mock(return_value=1)  # type: ignore[method-assign]
    queue_controller._play_index = AsyncMock()  # type: ignore[method-assign]
    voice = _voice(2.0)
    frame_size = _frame_size(TEST_PCM_FORMAT)
    cancelled: list[int] = []
    output = bytearray()
    tasks_before = asyncio.all_tasks()

    async def _cancel() -> None:
        if command == "next":
            await queue_controller.next("queue-1")
        else:
            await queue_controller.play_index("queue-1", 1)

    stream = _flow(audio, queue, first)
    stream_anchor: int | None = None
    async for chunk in stream:
        output.extend(chunk)
        if stream_anchor is None and published and len(output) >= 3 * SECOND * frame_size:
            stream_anchor = _place_at(2.0, voice)(published[0])
        if (
            stream_anchor is not None
            and not cancelled
            and len(output) >= (stream_anchor + CANCEL_AFTER_SECONDS * SECOND) * frame_size
        ):
            during = asyncio.all_tasks()
            await _cancel()
            # dropping a program is synchronous: nothing of it keeps running
            assert asyncio.all_tasks() == during
            assert published[0].active is False
            cancelled.append(published[0].produced_frames)
            assert cancelled[0] == len(output) // frame_size
        await asyncio.sleep(0)

    assert stream_anchor is not None
    assert cancelled
    # the session withdrew its bus at its end
    assert controller.program_bus_active("queue-1") is False
    # the flow before the program and after the cancel is the run without it, byte for byte
    assert (
        bytes(output[: stream_anchor * frame_size]) == reference_pcm[: stream_anchor * frame_size]
    )
    assert bytes(output[cancelled[0] * frame_size :]) == reference_pcm[cancelled[0] * frame_size :]
    # while the program really was heard before the command arrived
    heard = slice(stream_anchor * frame_size, cancelled[0] * frame_size)
    assert bytes(output[heard]) != reference_pcm[heard]
    # and no task of an overlay or its mixer outlived the session
    assert asyncio.all_tasks() == tasks_before
    assert command in _cancel_reasons(audio)


async def test_a_session_that_ends_drops_what_it_still_had_to_play(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A closed session leaves no program behind, and the next session plays untouched."""
    reference_pcm = await _reference_session(monkeypatch)
    audio, queue, mass, _music_pcm, first, _second = _two_item_flow(monkeypatch, program_bus=True)
    controller = _wire_streams(mass)
    published = _publish_on(controller, mass)
    voice = _voice(2.0)
    frame_size = _frame_size(TEST_PCM_FORMAT)
    output = bytearray()
    tasks_before = asyncio.all_tasks()
    anchor: int | None = None

    stream = _flow(audio, queue, first)
    async for chunk in stream:
        output.extend(chunk)
        if anchor is None and published and len(output) >= 3 * SECOND * frame_size:
            anchor = _place_at(2.0, voice)(published[0])
        if (
            anchor is not None
            and len(output) >= (anchor + CANCEL_AFTER_SECONDS * SECOND) * frame_size
        ):
            # the player closes the stream while the program is being heard
            await stream.aclose()
            break
        await asyncio.sleep(0)

    bus = published[0]
    assert anchor is not None
    # the ended session dropped its program and withdrew its bus
    assert bus.active is False
    assert controller.program_bus_active("queue-1") is False
    assert "flow stream ended" in _cancel_reasons(audio)
    # nothing of the ended session can be mixed into anything now
    assert controller.schedule_overlay("queue-1", voice, label="announcement") is False
    assert asyncio.all_tasks() == tasks_before
    # a session that follows is exactly the run without a program
    fresh_audio, fresh_queue, _fresh_mass, _fresh_music, fresh_first, _fresh_second = (
        _two_item_flow(monkeypatch, program_bus=True)
    )
    fresh_chunks, _fresh_anchor = await _drain(_flow(fresh_audio, fresh_queue, fresh_first), [])
    assert b"".join(fresh_chunks) == reference_pcm


async def test_the_switch_off_keeps_a_scheduled_program_out_of_the_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the bus there is nothing to schedule into, and the items play untouched."""
    reference_pcm = await _reference_session(monkeypatch)
    audio, queue, mass, _music_pcm, first, _second = _two_item_flow(monkeypatch, program_bus=False)
    controller = _wire_streams(mass)
    frame_size = _frame_size(TEST_PCM_FORMAT)
    scheduled: list[bool] = []
    chunks: list[bytes] = []
    produced = 0

    async for chunk in _flow(audio, queue, first):
        chunks.append(chunk)
        produced += len(chunk)
        if not scheduled and produced >= 3 * SECOND * frame_size:
            # the queue has no live session to mix an announcement into
            assert controller.program_bus_active("queue-1") is False
            scheduled.append(
                controller.schedule_overlay("queue-1", _voice(2.0), label="announcement")
            )
        await asyncio.sleep(0)

    assert scheduled == [False]
    assert controller.program_bus_active("queue-1") is False
    cast("Any", mass.streams.register_program_bus).assert_not_called()
    # the flow is exactly the reference run, chunk for chunk
    assert b"".join(chunks) == reference_pcm
    assert [len(chunk) for chunk in chunks] == [
        len(chunk) for chunk in _chunks_of(reference_pcm, TEST_PCM_FORMAT)
    ]

"""
The AI DJ's term engine against the real program bus, ledger and streams controller.

Every other test of the term engine answers with a scripted ``ScheduleResult`` on a stand-in
for the streams controller, so the seam that produced the accounting bugs — what the engine
is told about a placement and what the bus and the ledger really did with it — was never
crossed. Here the engine decides, a render is stood in for while the producer keeps running,
and the term is placed through the real ``StreamsController.schedule_overlay`` onto a real
``FlowProgramBus``: the frames the bus emits are asserted to be the ones the ledger books,
and the term ends as aired when the listener reached the end of what was really mixed.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from music_assistant.controllers.streams.program_ledger import OverlayProgramState
from music_assistant.controllers.streams.program_schedule import ScheduleReason
from music_assistant.controllers.streams.smart_fades.filters import VOICE_OVER_DUCK_DEPTH
from music_assistant.providers.ai_radio.constants import OVERLAY_LABEL_TRANSITION
from music_assistant.providers.ai_radio.terms import ProgramTerm, TermState
from tests.controllers.streams.program_bus_helpers import (
    bare_streams_controller,
    hear,
    register_bus,
)

from .test_queue_dj import PLAYBACK_PCM_FORMAT, TRACK_SECONDS, _overlay_dj

if TYPE_CHECKING:
    from music_assistant_models.media_items import AudioFormat

    from music_assistant.controllers.streams.program_ledger import LedgerEntry

QUEUE_ID = "queue-1"
SECOND = PLAYBACK_PCM_FORMAT.sample_rate
FRAME_SIZE = (PLAYBACK_PCM_FORMAT.bit_depth // 8) * PLAYBACK_PCM_FORMAT.channels
MUSIC_LEVEL = 100
VOICE_LEVEL = 1000
# seconds of music the session produced before the term's pass, and how far the listener is
# behind that: the lead the bus reports to the engine
PRODUCED_SECONDS = 4.0
LISTENER_SECONDS = 3.5
# seconds of the running item that are left when the term is decided, and what the render
# adds to the producer while the engine waits for it
REMAINING_SECONDS = 8.0
RENDER_SECONDS = 3.0
PRODUCED_DURING_RENDER_SECONDS = 7.5


def _music(seconds: float, level: int = MUSIC_LEVEL) -> bytes:
    """Build music at the given level for the given number of seconds."""
    return struct.pack("<hh", level, level) * int(seconds * SECOND)


def _voice(seconds: float) -> bytes:
    """Build a program at its own level for the given number of seconds."""
    return struct.pack("<hh", VOICE_LEVEL, VOICE_LEVEL) * int(seconds * SECOND)


def _levels(pcm: bytes) -> list[int]:
    """Return the first channel's samples of a PCM chunk."""
    return list(struct.unpack(f"<{len(pcm) // 2}h", pcm))[::2]


class _Harness:
    """A queue DJ on the real streams controller, with its bus and the produced audio."""

    def __init__(self, tmp_path: Path) -> None:
        """Arm the DJ, publish a real bus for its queue and produce the audio up to now."""
        self.dj, self.tracks = _overlay_dj(tmp_path)
        # a break that carries over the boundary has no head: it starts on the boundary
        self.dj._sections["Song_Transition"]["allow_post"] = True
        self.streams = bare_streams_controller(audible=LISTENER_SECONDS, enabled=True)
        self.dj.mass.streams = self.streams
        self.queue = self.dj.player_queues._queue
        # the queue plays without a crossfade, so its boundary is the item's end
        cast("Any", self.queue).crossfade_enabled = False
        self.bus = register_bus(self.streams, pcm_format=PLAYBACK_PCM_FORMAT)
        self.emitted = bytearray()

    async def produce(self, seconds: float) -> None:
        """Push music through the bus, as the flow session does while the queue plays."""
        self.emitted.extend(await self.bus.feed(_music(seconds)))

    async def run_from_the_boundary(self) -> tuple[ProgramTerm, ...]:
        """Decide, render and place the term of the boundary eight seconds away."""
        await self.produce(PRODUCED_SECONDS)
        # the queue is eight seconds from its boundary, and the listener 0.5 s behind the
        # producer: the engine sees a lead of half a second
        self.queue.current_item = self.tracks[1]
        self.queue.corrected_elapsed_time = TRACK_SECONDS - REMAINING_SECONDS
        await self.dj._replan_queue(QUEUE_ID)
        await self.dj._pump_terms(QUEUE_ID)
        return self.dj._term_store(QUEUE_ID).terms()

    def heard(self, seconds: float) -> None:
        """Move the queue's player to the given position of the stream."""
        hear(self.streams, seconds)

    async def feed_to_end(self) -> None:
        """Emit music until the program the bus holds has been heard to its end."""
        while self.bus.active:
            await self.produce(1.0)
            # the listener keeps up with the producer, half a second behind it
            self.heard(self.bus.produced_frames / SECOND - 0.5)


def _assert_the_heard_audio(entry: LedgerEntry, emitted: bytearray) -> None:
    """
    Hold what the listener heard against the frames the ledger booked.

    :param entry: The ledger's entry for the program.
    :param emitted: Everything the bus let out of the session.
    """
    levels = _levels(bytes(emitted))
    mixed = int(MUSIC_LEVEL * (1 - VOICE_OVER_DUCK_DEPTH)) + VOICE_LEVEL
    # the program is mixed in from its anchor to the deadline the ledger used, and not a
    # frame beyond it
    assert levels[entry.start_frame - 1] == MUSIC_LEVEL
    assert levels[entry.start_frame + SECOND // 2] == mixed
    assert levels[entry.deadline_frame - 1] == mixed
    assert levels[entry.deadline_frame + SECOND] == MUSIC_LEVEL
    # it lasted the audio the render delivered, no more and no less
    heard_seconds = (entry.deadline_frame - entry.start_frame) / SECOND
    assert heard_seconds == pytest.approx(RENDER_SECONDS, abs=1 / SECOND)
    assert len(emitted) // FRAME_SIZE >= entry.deadline_frame


async def test_a_term_airs_through_the_real_bus_ledger_and_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A decided term is rendered, placed on the real bus and ends as aired when it is heard.

    The render stands in for the engine, but it takes as long as the producer needs to run
    seven and a half seconds ahead, which is what makes the placement late: the boundary the
    term was written for is already produced. The bus takes the term two hundred and fifty
    milliseconds behind that boundary, moves it to the producer's edge and says so, and the
    ledger books exactly the frames the listener then hears — which is what the term ends on.
    """
    harness = _Harness(tmp_path)
    rendered = _voice(RENDER_SECONDS)
    rendered_at: list[float] = []

    async def _render(
        _queue_id: str, _program: dict[str, Any], _section: Any, _pcm_format: AudioFormat
    ) -> bytes:
        # the producer keeps running while the engine is busy: nothing about the placement
        # is read from the state the decision was made on
        await harness.produce(PRODUCED_DURING_RENDER_SECONDS)
        rendered_at.append(harness.streams.program_bus_lead_seconds(QUEUE_ID) or 0.0)
        return rendered

    monkeypatch.setattr(harness.dj, "_render_section_pcm", _render)
    placements: list[Any] = []
    schedule_overlay = harness.streams.schedule_overlay

    def _schedule_overlay(*args: Any, **kwargs: Any) -> Any:
        result = schedule_overlay(*args, **kwargs)
        placements.append(result)
        return result

    monkeypatch.setattr(harness.streams, "schedule_overlay", _schedule_overlay)

    terms = await harness.run_from_the_boundary()

    # the engine placed on a reading the render moved: eight seconds of lead, not the half
    # a second the decision was made on
    assert rendered_at == [pytest.approx(8.0)]
    # one term was placed, through the controller's own api
    assert len(placements) == 1
    result = placements[0]
    assert result.scheduled is True
    # it missed the boundary it was written for by what the producer ran ahead during the
    # render, and the api reports that instead of pretending it landed on the boundary
    assert result.late_by_seconds == pytest.approx(0.25, abs=1 / SECOND)
    # the vocabulary of a placement that had to be moved: a late boundary is not a taken one
    assert result.reason is None
    # the frames the api says it took are the frames the ledger books for it
    frames = len(rendered) // FRAME_SIZE
    assert result.frames == frames
    entry = harness.streams._program_ledger.entries(QUEUE_ID)[-1]
    assert entry.label == OVERLAY_LABEL_TRANSITION
    assert entry.state is OverlayProgramState.SCHEDULED
    assert entry.frames == frames
    assert entry.late_by_seconds == pytest.approx(result.late_by_seconds)
    assert entry.reason is None
    # the bus had nothing of the program to cut: it starts at the producer's edge, whole
    assert entry.start_frame == result.anchor_frame
    assert harness.bus.scheduled_voice_frames(OVERLAY_LABEL_TRANSITION) == frames
    # the deadline the ledger airs it on is the end of what the bus really holds
    assert entry.deadline_frame == entry.start_frame + frames

    term = next(term for term in terms if term.state is TermState.SCHEDULED)
    assert term.scheduled_frame == entry.start_frame

    await harness.feed_to_end()

    # the listener reached the end of the program: the ledger and the term both say aired
    state = harness.streams.overlay_state(QUEUE_ID)[-1]
    assert state.label == OVERLAY_LABEL_TRANSITION
    assert state.state is OverlayProgramState.AIRED
    assert state.start_frame == entry.start_frame
    assert state.frames == frames
    await harness.dj._pump_terms(QUEUE_ID)
    assert term.state is TermState.AIRED
    assert term.dropped_reason is None

    # and the audio behind that bookkeeping is the one the listener heard
    _assert_the_heard_audio(entry, harness.emitted)


async def test_a_boundary_the_producer_already_passed_is_dropped_as_too_late(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A term of a boundary the producer ran past is dropped by the engine's own vocabulary.

    The engine reads the lead the bus reports, so a session that produced far ahead of what
    the listener heard can not reach a boundary that close: the term ends as ``too_late``,
    which is the reason the panel and the log show, and nothing of it reaches the ledger.
    """
    harness = _Harness(tmp_path)

    async def _render(*_args: Any) -> bytes:
        return _voice(RENDER_SECONDS)

    monkeypatch.setattr(harness.dj, "_render_section_pcm", _render)

    terms = await harness.run_from_the_boundary()
    dropped = next(term for term in terms if term.state is TermState.DECIDED)
    # the producer runs twenty-eight seconds ahead of what the listener heard, and the next
    # boundary is eight seconds away: the engine gives it up instead of paying for it
    await harness.produce(20.0)
    harness.queue.current_item = harness.tracks[2]
    harness.queue.corrected_elapsed_time = TRACK_SECONDS - REMAINING_SECONDS

    await harness.dj._pump_terms(QUEUE_ID)

    assert dropped.state is TermState.DROPPED
    assert dropped.dropped_reason == ScheduleReason.TOO_LATE
    # nothing was offered to the bus for it, so the ledger knows nothing of it either
    assert [entry.label for entry in harness.streams._program_ledger.entries(QUEUE_ID)] == [
        OVERLAY_LABEL_TRANSITION
    ]

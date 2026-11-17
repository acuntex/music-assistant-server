"""
Place a clip's head over the tail of the item it was planned against.

A clip that declares a pre is heard over the item before it: its head is taken out of the clip's
own stream, mixed over that item's tail on the program bus and the rest of the stream then
continues from past the head, so the clip is still decoded once. The head is placed where the
declared durations put the item's end, never bought with a hold on the item's audio, and the
residue between that placement and the end the item really produced is reported as drift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from music_assistant_models.enums import MediaType
from music_assistant_models.errors import MusicAssistantError

from music_assistant.controllers.streams.constants import (
    LEAD_MIN_LEAD_SECONDS,
    MAX_LEAD_OVERLAP,
    PRE_OVERLAP_ATTR,
    PRE_OVERLAP_FROM_ATTR,
)
from music_assistant.controllers.streams.program_mixer import FlowProgramBus
from music_assistant.controllers.streams.program_schedule import ProgramPriority, ScheduleReason

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from music_assistant_models.media_items import AudioFormat
    from music_assistant_models.queue_item import QueueItem

    from music_assistant.controllers.streams.audio import (
        StreamsAudio,
        _IncomingFadePrefetcher,
    )


async def schedule_program_lead(
    audio: StreamsAudio,
    program_bus: FlowProgramBus,
    queue_id: str,
    queue_track: QueueItem,
    pcm_format: AudioFormat,
    seek_position: float,
    playback_speed: float,
    incoming_prefetcher: _IncomingFadePrefetcher,
    *,
    held_tail_frames: int = 0,
    incoming_crossfade_frames: int = 0,
) -> _ProgramLead | None:
    """
    Place the next clip's head on the program bus at the frame this item ends on.

    The head is taken out of the stream the clip continues on, so the clip is decoded
    once: the bus plays it over this item's tail and the clip's own stream carries on
    from past it, reported from the media time it is really heard at. The item's own
    audio is not held back for the placement; the frame comes from the declared
    durations and the residue is reported as the anchor drift.

    :param audio: The audio sub-controller the item and its queue are read through.
    :param program_bus: The session's program bus.
    :param queue_id: The queue the item plays on.
    :param queue_track: The item that is playing.
    :param pcm_format: The flow's PCM format.
    :param seek_position: Where this item's stream starts, in media seconds.
    :param playback_speed: Multiplier between stream time and media time.
    :param incoming_prefetcher: Prefetcher that keeps the clip's head and its open stream.
    :param held_tail_frames: Frames of the item before this one that the boundary still
        holds back and will emit before this item's own audio.
    :param incoming_crossfade_frames: Frames of this item's head the boundary's mix takes
        out of its stream, so its audio ends that much earlier than its length says.
    """
    streamdetails = queue_track.streamdetails
    if (
        queue_track.media_type != MediaType.TRACK
        or streamdetails is None
        or not streamdetails.duration
    ):
        return None
    next_item = audio.mass.player_queues.get_next_item(queue_id, queue_track.queue_item_id)
    lead_seconds = planned_lead(next_item, queue_track.queue_item_id)
    if lead_seconds is None or next_item is None:
        return None
    if next_item.streamdetails is None:
        # the clip is due only after this item, so its details may not exist yet
        try:
            await audio.get_stream_details(queue_item=next_item)
        except MusicAssistantError as err:
            audio.logger.debug("Not starting %s over %s: %s", next_item.name, queue_track.name, err)
            return None
    remaining_media = streamdetails.duration - seek_position
    speed = max(playback_speed, 1e-6)
    if remaining_media <= 0:
        return None
    lead_seconds = min(lead_seconds, remaining_media / speed)
    if lead_seconds < LEAD_MIN_LEAD_SECONDS:
        # the head has to travel through the buffer and the encoder before the item
        # ends; a shorter lead can not be mixed in time and plays as a plain cut
        audio.logger.debug(
            "Not starting %s over %s: only %.2fs of lead are left, %.1fs are needed",
            next_item.name,
            queue_track.name,
            lead_seconds,
            LEAD_MIN_LEAD_SECONDS,
        )
        return None
    if not program_bus.can_mix:
        # taking the head out of the stream is only worth it when the bus can sum it
        # back in: a session that passes a source format through can not
        audio.logger.debug(
            "Not starting %s over %s: %s PCM can not be mixed",
            next_item.name,
            queue_track.name,
            pcm_format.content_type,
        )
        return None
    voice_pcm = await incoming_prefetcher.collect_head(next_item, lead_seconds)
    if not voice_pcm:
        audio.logger.debug(
            "Not starting %s over %s: its head could not be read",
            next_item.name,
            queue_track.name,
        )
        return None
    item_end_frame = (
        program_bus.produced_frames
        + held_tail_frames
        + int(remaining_media / speed * pcm_format.sample_rate)
        - incoming_crossfade_frames
    )
    anchor = item_end_frame - int(lead_seconds * pcm_format.sample_rate)
    audio.logger.info(
        "Starting %s over the last %.1fs of %s", next_item.name, lead_seconds, queue_track.name
    )
    # the anchor the declared durations give is the placement: nothing of this item is
    # held back for it, and the residue is reported when the item really ended
    lead = _ProgramLead(
        program_bus,
        audio.logger,
        voice_pcm,
        label=f"lead:{next_item.queue_item_id}",
        declared_anchor=anchor,
        # a head the bus drops before it was heard has to go back to the clip: its stream
        # was opened past the head, so it would start with the words after it
        on_dropped=lambda: incoming_prefetcher.discard_head(next_item.queue_item_id),
    )
    if lead.arm() is None:
        # the head was taken out of the stream but is not on the bus: close the
        # prefetch so the clip's own stream plays it from the start
        audio.logger.debug(
            "Not starting %s over %s: its head could not be placed",
            next_item.name,
            queue_track.name,
        )
        await incoming_prefetcher.close()
        return None
    return lead


def planned_lead(queue_item: QueueItem | None, outgoing_item_id: str) -> float | None:
    """
    Return the lead a clip declares over the given outgoing item, if any.

    :param queue_item: The incoming queue item, which may declare a lead.
    :param outgoing_item_id: Queue item the lead was planned against.
    """
    if queue_item is None:
        return None
    declared = queue_item.extra_attributes.get(PRE_OVERLAP_ATTR)
    if not isinstance(declared, (int, float)) or declared <= 0:
        return None
    if queue_item.extra_attributes.get(PRE_OVERLAP_FROM_ATTR) != outgoing_item_id:
        return None
    return min(float(declared), MAX_LEAD_OVERLAP)


@dataclass(frozen=True)
class _LeadAnchorReport:
    """Where a lead's head was placed, relative to the end the item really produced."""

    drift_ms: float
    anchor: int
    produced_end: int
    declared_end: int


class _ProgramLead:
    """
    A clip's head, placed on the bus over the tail of the item it was planned against.

    The head has to end on the item's last frame: the clip's own stream continues from
    there, so an end short of it leaves a hole in its speech. A track's duration is a
    whole-second estimate and a seek lands a few hundred ms away from it, so the placement
    can not be exact. It is also not bought with a hold: withholding the item's last
    seconds starves a player whose buffer is shorter than the lead, which is heard as the
    stream standing still and the reported position stepping back. The head is placed where
    the declared durations put the item's end, the bus moves it forward when that frame is
    already produced, and the residue is reported as the anchor drift once the item ended.
    """

    def __init__(
        self,
        program_bus: FlowProgramBus,
        logger: logging.Logger,
        voice_pcm: bytes,
        *,
        label: str,
        declared_anchor: int,
        on_dropped: Callable[[], None] | None = None,
    ) -> None:
        """
        Prepare the head and its placement.

        :param program_bus: The session's program bus.
        :param logger: Logger of the audio sub-controller, for the drift report.
        :param voice_pcm: The clip's head, in the flow's PCM format.
        :param label: Name of the program, for the log.
        :param declared_anchor: Frame the declared durations put the head at.
        :param on_dropped: Called when the bus drops the head before it was heard, so the
            audio that was taken out of the clip's stream can go back to the clip.
        """
        self._bus = program_bus
        self._logger = logger
        self._pcm = voice_pcm
        self._on_dropped = on_dropped
        self.label = label
        self._declared_anchor = declared_anchor
        self._frame_size = max(program_bus.frame_size, 1)
        self._sample_rate = max(program_bus.pcm_format.sample_rate, 1)
        self.voice_frames = len(voice_pcm) // self._frame_size
        # nothing is trimmed off the head until the bus has to move its anchor forward
        self._played_frames = self.voice_frames
        self._anchor: int | None = None
        self._finished = False
        self._report: _LeadAnchorReport | None = None

    @property
    def anchor_report(self) -> _LeadAnchorReport | None:
        """Return where the head was placed relative to the item's end, once known."""
        return self._report

    def arm(self) -> int | None:
        """
        Place the head where the declared durations expect the item to end.

        :return: The frame it was anchored at, or None when it can not be mixed.
        """
        # the window the declared durations give, so the head can not outlast the item
        placement = self._bus.place(
            self._pcm,
            self._declared_anchor,
            label=self.label,
            stop_frame=self._declared_anchor + self.voice_frames,
            # a head is the front of a clip whose stream continues outside the bus, so it
            # outranks a term (which gives way) but not an announcement (which is heard
            # over everything and may take the window)
            priority=ProgramPriority.LEAD,
            on_dropped=self._on_dropped,
        )
        if placement is not None:
            # a late anchor drops the front of the head, so only what is left of it plays
            self._anchor = placement.start_frame
            self._played_frames = placement.frames
        return self._anchor

    def finish(self, pending_frames: int = 0) -> None:
        """
        Report where the head landed once the item's own stream ended.

        A head whose window is not over yet is dropped here: what is left of it would sound
        over the item after this one while that item's body starts its own speech.

        :param pending_frames: Frames of the item that are still held back for a boundary,
            so the end is where the item's audio really ends, not where it was fed to.
        """
        if self._finished:
            return
        self._finished = True
        end_frame = self._bus.produced_frames + pending_frames
        self._log_drift(end_frame)
        if self._anchor is not None and self._anchor + self._played_frames > end_frame:
            self._bus.cancel(self.label, ScheduleReason.HEAD_OUTLASTED)

    def abandon(self) -> None:
        """Drop the head when the item's stream ends before it reached its anchor."""
        if self._finished:
            return
        self._finished = True
        self._bus.cancel(self.label, ScheduleReason.ITEM_ABANDONED)

    # --- Private methods ---

    def _log_drift(self, end_frame: int) -> None:
        """
        Report how far the head's anchor is from the item's produced end.

        :param end_frame: Frame the item's audio really ended on.
        """
        if self._anchor is None:
            return
        # only the part of the head the clamp left is heard, and its end, and so the residue
        # against the item's end, is measured from that part
        report = _LeadAnchorReport(
            drift_ms=(self._anchor - (end_frame - self._played_frames)) / self._sample_rate * 1000,
            anchor=self._anchor,
            produced_end=end_frame,
            declared_end=self._declared_anchor + self.voice_frames,
        )
        self._report = report
        self._logger.debug(
            "PROGRAM-BUS anchor drift=%.0fms label=%s anchor=%d produced_end=%d declared_end=%d",
            report.drift_ms,
            self.label,
            report.anchor,
            report.produced_end,
            report.declared_end,
        )


async def through_program_lead(
    music: AsyncGenerator[bytes],
    lead: _ProgramLead | None = None,
    *,
    pending_buffer: bytearray | None = None,
    frame_size: int = 0,
) -> AsyncGenerator[bytes]:
    """
    Run an item's audio out, finishing the head placed over its tail when it ends.

    The program bus itself sits on the emitted stream, so this wrapper only carries the
    item through unchanged and reports the head against the item's real end.

    :param music: The item's own stream.
    :param lead: The clip's head placed over this item's tail, if one was armed for it.
    :param pending_buffer: Frames of this item still held back for a boundary, if any.
    :param frame_size: Size in bytes of one frame of the flow's PCM format.
    """
    finished = False
    try:
        # a lead never holds the item back: every chunk goes through as it comes, so the
        # player is fed at the rate the source delivers
        async for chunk in music:
            yield chunk
        finished = True
    finally:
        if lead is not None:
            # the item's end is known now: report where the head landed
            if finished:
                pending_frames = (
                    len(pending_buffer) // frame_size
                    if pending_buffer is not None and frame_size
                    else 0
                )
                lead.finish(pending_frames)
            else:
                lead.abandon()
        # closing this wrapper has to release the decoder underneath it as well
        await music.aclose()

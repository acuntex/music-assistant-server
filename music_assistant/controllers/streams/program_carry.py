"""
Read what is left of a clip that is heard entirely across the boundary around it.

A clip whose head was mixed over the item before it and whose tail overlap plays into the item
after it has no body of its own: serving it as an item would hand the player an item that
produces no audio and report it as what is playing. This module decides when that is the case and
returns the audio the boundary still owes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from music_assistant_models.enums import MediaType

from music_assistant.controllers.streams.constants import (
    BUS_CLIP_CARRY_SLACK,
    BUS_CLIP_HEAD_SLACK,
)

if TYPE_CHECKING:
    from music_assistant_models.queue_item import QueueItem
    from music_assistant_models.streamdetails import TailOverlap

    from music_assistant.controllers.streams.audio import (
        StreamsAudio,
        _IncomingFadePrefetcher,
    )


def bus_only_clip(audio: StreamsAudio, queue_item: QueueItem) -> TailOverlap | None:
    """
    Return the tail overlap that leaves an item no body of its own, if it has one.

    Such an item is heard in full across the boundary around it: its head plays over the
    item before it and this overlap plays over the item after it, so there is nothing of
    it left to stream as an item. The overlap only ever plays into the item it was planned
    against, so anything else keeps the item an item.

    :param audio: The audio sub-controller the item's queue is read through.
    :param queue_item: The item about to be streamed.
    """
    streamdetails = queue_item.streamdetails
    if streamdetails is None or not streamdetails.duration:
        return None
    if (tail_overlap := streamdetails.tail_overlap) is None:
        return None
    next_item = audio.mass.player_queues.get_next_item(
        queue_item.queue_id, queue_item.queue_item_id
    )
    if next_item is None or next_item.queue_item_id != tail_overlap.next_queue_item_id:
        return None
    if next_item.media_type != MediaType.TRACK:
        return None
    return tail_overlap


async def carry_bus_only_clip(
    audio: StreamsAudio,
    queue_item: QueueItem,
    pcm_sample_size: int,
    incoming_prefetcher: _IncomingFadePrefetcher,
) -> bytes | None:
    """
    Read what is left of a clip that is heard entirely across the boundary around it.

    The head was taken out of the clip's stream and mixed over the item before it, so the
    rest of that stream is everything the clip still owes: it is summed into the boundary
    into the item after it instead of being served as an item of its own. A clip its head
    covered whole owes nothing, and is reported as such with an empty result. A head that
    only covers the declared length of a clip whose stream still delivers is no cover: that
    audio is speech, and the clip is served as the item it is, from past the head.

    :param audio: The audio sub-controller the item's queue is read through.
    :param queue_item: The clip about to be streamed.
    :param pcm_sample_size: Bytes of one second of the flow's PCM format.
    :param incoming_prefetcher: Prefetcher still holding the clip's stream past its head.
    :return: The clip's remaining audio (empty when there is none), or None when it is
        to be served itself.
    """
    head_seconds = incoming_prefetcher.head_mixed(queue_item)
    if head_seconds is None:
        return None
    streamdetails = queue_item.streamdetails
    duration = float(streamdetails.duration or 0) if streamdetails is not None else 0.0
    tail_overlap = bus_only_clip(audio, queue_item)
    if tail_overlap is None and duration:
        # no post carries the rest past the boundary, so the head has to cover the
        # clip on its own; the queue reports media time, the head was taken in stream time
        speed = cast("float", queue_item.extra_attributes.get("playback_speed", 1.0)) or 1.0
        if (
            head_seconds + BUS_CLIP_HEAD_SLACK >= duration / speed
            # the declared duration is a whole-second estimate, so covering it says
            # nothing about the stream: only one that really ends at the head leaves
            # nothing to serve, and anything past it is the clip's own speech
            and not await incoming_prefetcher.rest_pending(queue_item)
        ):
            audio.logger.debug(
                "LEAD-OVERLAY cover clip=%s(%s) head=%.2fs dur=%.2fs",
                queue_item.queue_item_id,
                queue_item.media_type,
                head_seconds,
                duration,
            )
            return b""
    if tail_overlap is None:
        return None
    if head_seconds + tail_overlap.duration < duration:
        # a body of its own is left: it is heard as an item, as any item before it was
        return None
    carried = await incoming_prefetcher.carry_rest(
        queue_item, int(pcm_sample_size * (tail_overlap.duration + BUS_CLIP_CARRY_SLACK))
    )
    if not carried:
        return None
    audio.logger.debug(
        "LEAD-OVERLAY carry clip=%s(%s) head=%.2fs post=%.2fs carried=%.2fs",
        queue_item.queue_item_id,
        queue_item.media_type,
        head_seconds,
        tail_overlap.duration,
        len(carried) / pcm_sample_size,
    )
    return carried

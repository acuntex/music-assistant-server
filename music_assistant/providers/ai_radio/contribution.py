"""
The ceiling on one spoken contribution.

A section's character limit decides how long the host talks, and a long script can turn a
break into a monologue. The host sets a ceiling in seconds, and a contribution rendered for
the queue's stream is cut to it - after the render, frame-exact, so the spoken delivery is
never changed by shortening the text. The cut lands behind a short fade, so the ceiling is
heard as the end of the break instead of as a step in the middle of a word. A clip that
plays as an item keeps its own length.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import ContentType

from .constants import CONTRIBUTION_FADE_SECONDS, MAX_CONTRIBUTION_DEFAULT_SECONDS
from .helpers import coerce_float

if TYPE_CHECKING:
    from music_assistant_models.media_items import AudioFormat


def contribution_seconds(program: dict[str, Any]) -> float:
    """
    Return the host's ceiling for one spoken contribution, in seconds.

    :param program: The station and host a contribution belongs to.
    """
    return coerce_float(program.get("max_contribution_seconds"), MAX_CONTRIBUTION_DEFAULT_SECONDS)


def cap_contribution_pcm(pcm: bytes, pcm_format: AudioFormat, max_seconds: float) -> bytes:
    """
    Cut a rendered contribution to the given ceiling, on a whole frame, with a short fade.

    :param pcm: The rendered contribution, in the queue stream's format.
    :param pcm_format: The PCM format the queue's stream produces.
    :param max_seconds: The longest the contribution may last.
    """
    frame_bytes = _frame_bytes(pcm_format)
    if frame_bytes is None:
        return pcm
    limit = int(max_seconds * pcm_format.sample_rate) * frame_bytes
    if len(pcm) <= limit:
        return pcm
    cut = bytearray(pcm[:limit])
    _fade_out(cut, pcm_format, frame_bytes)
    return bytes(cut)


def _frame_bytes(pcm_format: AudioFormat) -> int | None:
    """
    Return the bytes of one PCM frame, or None when the format does not say.

    :param pcm_format: The PCM format to measure.
    """
    bit_depth = pcm_format.bit_depth or 0
    channels = pcm_format.channels or 0
    if bit_depth < 8 or channels < 1 or not pcm_format.sample_rate:
        return None
    return (bit_depth // 8) * channels


def _fade_out(pcm: bytearray, pcm_format: AudioFormat, frame_bytes: int) -> None:
    """
    Fade the last frames of a cut linearly to silence, so the cut is not a step.

    :param pcm: The cut contribution, faded in place.
    :param pcm_format: The PCM format of the audio.
    :param frame_bytes: Bytes of one PCM frame.
    """
    codec = _sample_codec(pcm_format)
    if codec is None:
        return
    sample = struct.Struct(f"<{pcm_format.channels}{codec}")
    frames = len(pcm) // frame_bytes
    fade_frames = min(frames, int(CONTRIBUTION_FADE_SECONDS * pcm_format.sample_rate))
    if fade_frames < 1:
        return
    start = (frames - fade_frames) * frame_bytes
    for index in range(fade_frames):
        gain = 1.0 - (index + 1) / fade_frames
        offset = start + index * frame_bytes
        values = sample.unpack_from(pcm, offset)
        faded = (
            [value * gain for value in values]
            if codec == "f"
            else [int(value * gain) for value in values]
        )
        sample.pack_into(pcm, offset, *faded)


def _sample_codec(pcm_format: AudioFormat) -> str | None:
    """
    Return the struct codec one sample is written in, if it can be faded.

    :param pcm_format: The PCM format to read.
    """
    if pcm_format.content_type == ContentType.PCM_F32LE and pcm_format.bit_depth == 32:
        return "f"
    if pcm_format.content_type == ContentType.PCM_S16LE and pcm_format.bit_depth == 16:
        return "h"
    return None

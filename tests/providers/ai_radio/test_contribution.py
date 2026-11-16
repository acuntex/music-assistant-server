"""Tests for the ceiling on one spoken contribution."""

from __future__ import annotations

import struct
from itertools import pairwise
from typing import Any

import pytest
from music_assistant_models.enums import ContentType
from music_assistant_models.media_items import AudioFormat

from music_assistant.providers.ai_radio.constants import (
    CONTRIBUTION_FADE_SECONDS,
    MAX_CONTRIBUTION_DEFAULT_SECONDS,
    MAX_CONTRIBUTION_MAX_SECONDS,
    MAX_CONTRIBUTION_MIN_SECONDS,
)
from music_assistant.providers.ai_radio.contribution import (
    cap_contribution_pcm,
    contribution_seconds,
)

# the format a live flow session mixes a program into, and the one the item path serves
FLOW_FORMAT = AudioFormat(
    content_type=ContentType.PCM_F32LE, sample_rate=48000, bit_depth=32, channels=2
)
S16_FORMAT = AudioFormat(
    content_type=ContentType.PCM_S16LE, sample_rate=48000, bit_depth=16, channels=2
)
# the level the tone carries, in each format's own scale
TONE_LEVEL = 0.5


def _pcm(seconds: float, pcm_format: AudioFormat = FLOW_FORMAT) -> bytes:
    """Return silent PCM of the given length in the given format."""
    frame_bytes = (pcm_format.bit_depth // 8) * pcm_format.channels
    return b"\x00" * (int(seconds * pcm_format.sample_rate) * frame_bytes)


def _tone(
    seconds: float, pcm_format: AudioFormat = FLOW_FORMAT, level: float = TONE_LEVEL
) -> bytes:
    """Return speech-like PCM of the given length at a constant level."""
    samples = int(seconds * pcm_format.sample_rate) * pcm_format.channels
    codec = "f" if pcm_format.bit_depth == 32 else "h"
    return struct.pack(f"<{samples}{codec}", *([level] * samples))


def _samples(pcm: bytes, pcm_format: AudioFormat = FLOW_FORMAT) -> list[float]:
    """Return the first channel's samples of the given PCM."""
    frame_bytes = (pcm_format.bit_depth // 8) * pcm_format.channels
    codec = "f" if pcm_format.bit_depth == 32 else "h"
    frames = len(pcm) // frame_bytes
    return list(struct.unpack(f"<{frames * pcm_format.channels}{codec}", pcm))[
        :: pcm_format.channels
    ]


def _seconds(pcm: bytes, pcm_format: AudioFormat = FLOW_FORMAT) -> float:
    """Return the length of the given PCM in seconds."""
    frame_bytes = (pcm_format.bit_depth // 8) * pcm_format.channels
    return len(pcm) / frame_bytes / pcm_format.sample_rate


def test_the_default_ceiling_is_the_documented_default() -> None:
    """A program that says nothing about the ceiling speaks up to the default."""
    assert MAX_CONTRIBUTION_DEFAULT_SECONDS == 45.0
    assert contribution_seconds({}) == MAX_CONTRIBUTION_DEFAULT_SECONDS


@pytest.mark.parametrize("value", [MAX_CONTRIBUTION_MIN_SECONDS, 20, MAX_CONTRIBUTION_MAX_SECONDS])
def test_the_program_carries_the_hosts_ceiling(value: float) -> None:
    """The ceiling a program carries is the one the cap uses."""
    assert contribution_seconds({"max_contribution_seconds": value}) == pytest.approx(float(value))


@pytest.mark.parametrize("value", [None, "twenty", [20]])
def test_a_ceiling_the_program_cannot_read_falls_back_to_the_default(value: Any) -> None:
    """A stored program is never validated again at render time, so a broken value is no cap."""
    assert contribution_seconds({"max_contribution_seconds": value}) == (
        MAX_CONTRIBUTION_DEFAULT_SECONDS
    )


def test_a_contribution_over_the_ceiling_is_cut_frame_exact() -> None:
    """A seven-in-ten-seconds clip is cut to the ceiling, on a whole frame."""
    pcm = _pcm(70)

    cut = cap_contribution_pcm(pcm, FLOW_FORMAT, 45)

    assert _seconds(cut) == pytest.approx(45.0)
    # frame exact: four bytes per frame, no half frame at the cut
    assert len(cut) % 4 == 0


def test_a_cut_contribution_fades_out_at_the_cut() -> None:
    """A cut ends with a short fade instead of a step, and keeps exactly the ceiling."""
    pcm = _tone(70)

    cut = cap_contribution_pcm(pcm, FLOW_FORMAT, 45)

    fade_frames = int(CONTRIBUTION_FADE_SECONDS * FLOW_FORMAT.sample_rate)
    frame_bytes = (FLOW_FORMAT.bit_depth // 8) * FLOW_FORMAT.channels
    assert len(cut) == 45 * FLOW_FORMAT.sample_rate * frame_bytes
    samples = _samples(cut)
    assert samples[-1] == pytest.approx(0.0)
    # linear and monotone: the fade only ever gets quieter, and the frame before it is untouched
    tail = samples[-fade_frames:]
    assert all(later <= earlier for earlier, later in pairwise(tail))
    assert samples[-fade_frames - 1] == pytest.approx(TONE_LEVEL)
    # everything before the fade is the rendered speech, unaltered
    plain = (45 * FLOW_FORMAT.sample_rate - fade_frames) * frame_bytes
    assert cut[:plain] == pcm[:plain]


@pytest.mark.parametrize(("pcm_format", "level"), [(FLOW_FORMAT, TONE_LEVEL), (S16_FORMAT, 8000)])
def test_the_fade_follows_the_flow_format(pcm_format: Any, level: float) -> None:
    """The fade is written in the format the contribution came in, frame exact."""
    pcm = _tone(70, pcm_format, level)

    cut = cap_contribution_pcm(pcm, pcm_format, 20)

    frame_bytes = (pcm_format.bit_depth // 8) * pcm_format.channels
    assert len(cut) == 20 * pcm_format.sample_rate * frame_bytes
    assert _samples(cut, pcm_format)[-1] == pytest.approx(0.0)
    assert _samples(cut, pcm_format)[0] == pytest.approx(level)


def test_the_cut_lands_on_the_last_whole_frame() -> None:
    """A pcm with a partial frame at its end is cut on the frame the ceiling ends at."""
    pcm = _pcm(70) + b"\x01\x02"
    frame_bytes = (FLOW_FORMAT.bit_depth // 8) * FLOW_FORMAT.channels

    cut = cap_contribution_pcm(pcm, FLOW_FORMAT, 45)

    assert len(cut) == int(45 * FLOW_FORMAT.sample_rate) * frame_bytes


def test_a_shorter_ceiling_cuts_further() -> None:
    """The host decides where the cut lands: twenty seconds leave twenty seconds of speech."""
    cut = cap_contribution_pcm(_pcm(70, S16_FORMAT), S16_FORMAT, 20)

    assert _seconds(cut, S16_FORMAT) == pytest.approx(20.0)


@pytest.mark.parametrize("seconds", [12, 45])
def test_a_contribution_inside_the_ceiling_is_left_alone(seconds: float) -> None:
    """Speech up to the ceiling is served whole, never padded or trimmed."""
    pcm = _pcm(seconds)

    assert cap_contribution_pcm(pcm, FLOW_FORMAT, 45) == pcm


def test_a_format_the_bus_cannot_frame_carries_no_cut() -> None:
    """Without a sample rate or a bit depth there is no frame to cut on, so nothing is cut."""
    pcm = _pcm(70)
    unknown = AudioFormat(content_type=ContentType.UNKNOWN, sample_rate=0, bit_depth=0, channels=0)

    assert cap_contribution_pcm(pcm, unknown, 1) == pcm

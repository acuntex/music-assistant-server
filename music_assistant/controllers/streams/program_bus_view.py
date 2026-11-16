"""The read-only view of a queue's program bus, as the audio path's details show it."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field

from mashumaro import DataClassDictMixin

from music_assistant.controllers.streams.program_ledger import (
    LedgerEntry,
    OverlayProgramState,
    ProgramLedger,
)
from music_assistant.controllers.streams.program_mixer import FlowProgramBus

# how much of a program label the ui is given; the full label stays in the log
MAX_LABEL_LEN = 64

# the states a program can still be heard in; everything else already ended
PENDING_STATES = (OverlayProgramState.SCHEDULED, OverlayProgramState.ORPHANED)

# code points that belong to the character before them, so a cut never splits them off
_JOINING_CATEGORIES = frozenset({"Mn", "Mc", "Me"})
_VARIATION_SELECTORS = frozenset({*range(0xFE00, 0xFE10), *range(0xE0100, 0xE01F0)})
_SKIN_TONES = frozenset(range(0x1F3FB, 0x1F400))
_REGIONAL_INDICATORS = frozenset(range(0x1F1E6, 0x1F200))
_ZERO_WIDTH_JOINER = "\u200d"


@dataclass(frozen=True)
class ProgramBusFormat(DataClassDictMixin):
    """The PCM format a program has to be delivered in to be mixed."""

    sample_rate: int
    bit_depth: int
    channels: int


@dataclass(frozen=True)
class ProgramBusClaims(DataClassDictMixin):
    """What the owners of a queue currently have the stream do for their programs."""

    flow: bool
    boundary: bool


@dataclass(frozen=True)
class ProgramBusOverlay(DataClassDictMixin):
    """A program the queue's stream still has to play."""

    label: str
    state: OverlayProgramState
    start_in_seconds: float | None
    frames: int
    # seconds before its boundary the program was planned to start
    head_seconds: float
    # seconds the bus had to move it past that boundary
    late_by_seconds: float


@dataclass(frozen=True)
class ProgramBusEnded(DataClassDictMixin):
    """A program that was heard to its end, or ended without being heard."""

    label: str
    state: OverlayProgramState
    reason: str | None
    frames: int


@dataclass(frozen=True)
class ProgramBusView(DataClassDictMixin):
    """What the queue's program bus is doing, without a single byte of program audio."""

    enabled: bool
    session_active: bool
    no_audible_position: bool
    format: ProgramBusFormat | None
    claims: ProgramBusClaims
    overlays: list[ProgramBusOverlay] = field(default_factory=list)
    last: list[ProgramBusEnded] = field(default_factory=list)


def build_program_bus_view(
    *,
    enabled: bool,
    queue_id: str,
    ledger: ProgramLedger,
    program_bus: FlowProgramBus | None,
    claims: ProgramBusClaims,
    position: int | None,
) -> ProgramBusView:
    """
    Build the reading of one queue's program bus.

    With the switch off nothing is mixed, so the reading is empty instead of showing what a
    session that started before the switch still holds.

    :param enabled: Whether the program bus is switched on.
    :param queue_id: Queue whose bus to describe.
    :param ledger: The ledger the programs of the queue are read from.
    :param program_bus: The queue's live program bus, or None without a flow session.
    :param claims: The claims the queue's owners currently hold.
    :param position: The frame of the live bus the listener has reached, or None when that
        can not be told. Read once, so the deadlines and the countdowns of one answer agree.
    """
    if not enabled:
        return ProgramBusView(
            enabled=False,
            session_active=False,
            no_audible_position=False,
            format=None,
            claims=ProgramBusClaims(flow=False, boundary=False),
        )
    ledger.refresh(queue_id, lambda bus: position if bus is program_bus else None)
    overlays: list[ProgramBusOverlay] = []
    ended: list[ProgramBusEnded] = []
    for entry in ledger.entries(queue_id):
        if entry.state in PENDING_STATES:
            overlays.append(
                ProgramBusOverlay(
                    label=_short_label(entry.label),
                    state=entry.state,
                    start_in_seconds=_seconds_until(entry, program_bus, position),
                    frames=entry.frames,
                    head_seconds=entry.head_seconds,
                    late_by_seconds=entry.late_by_seconds,
                )
            )
            continue
        ended.append(
            ProgramBusEnded(
                label=_short_label(entry.label),
                state=entry.state,
                reason=entry.reason,
                frames=entry.frames,
            )
        )
    return ProgramBusView(
        enabled=True,
        session_active=program_bus is not None,
        no_audible_position=program_bus is not None and position is None,
        format=_format_of(program_bus),
        claims=claims,
        overlays=overlays,
        last=ended,
    )


def _format_of(program_bus: FlowProgramBus | None) -> ProgramBusFormat | None:
    """Return the PCM format a program for the bus has to come in."""
    if program_bus is None:
        return None
    pcm_format = program_bus.pcm_format
    return ProgramBusFormat(
        sample_rate=pcm_format.sample_rate,
        bit_depth=pcm_format.bit_depth,
        channels=pcm_format.channels,
    )


def _short_label(label: str) -> str:
    """
    Cut a program label to the length the panel shows.

    The cut counts characters that stand on their own, so it never leaves half an emoji, a
    lone accent or half a flag behind. It approximates the grapheme clusters of the unicode
    standard rather than importing a table for them: combining marks, variation selectors,
    zero-width-joiner sequences, skin tones and flag pairs stay with their neighbour.
    """
    if len(label) <= MAX_LABEL_LEN:
        return label
    kept: list[str] = []
    for cluster in _clusters(label):
        if len(kept) == MAX_LABEL_LEN:
            break
        kept.append(cluster)
    return "".join(kept)


def _clusters(text: str) -> Iterator[str]:
    """Yield the text in clusters no wider than one visible character."""
    cluster = ""
    regionals = 0
    for char in text:
        code = ord(char)
        regional = code in _REGIONAL_INDICATORS
        joins = bool(cluster) and (
            unicodedata.category(char) in _JOINING_CATEGORIES
            or code in _VARIATION_SELECTORS
            or code in _SKIN_TONES
            or char == _ZERO_WIDTH_JOINER
            or cluster.endswith(_ZERO_WIDTH_JOINER)
            # a flag is a pair of regional indicators
            or (regional and regionals % 2 == 1)
        )
        if joins:
            cluster += char
        else:
            if cluster:
                yield cluster
            cluster = char
        regionals = regionals + 1 if regional else 0
    if cluster:
        yield cluster


def _seconds_until(
    entry: LedgerEntry, program_bus: FlowProgramBus | None, position: int | None
) -> float | None:
    """
    Return how long it takes until the program starts, if that can be told.

    A program of a bus that was replaced has no countdown: its frames were counted in a
    session that no longer produces them, so they say nothing about what is heard now.

    :param entry: The program to measure.
    :param program_bus: The queue's live program bus, if it has one.
    :param position: Frame of the live bus the listener has reached, if it is known.
    """
    if position is None or entry.bus is None or entry.bus is not program_bus:
        return None
    rate = max(entry.bus.pcm_format.sample_rate, 1)
    # a clock that moved past the anchor shows no countdown, never a negative one
    return max(0.0, (entry.start_frame - position) / rate)

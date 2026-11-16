"""Data models for AI Radio."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from music_assistant_models.errors import MusicAssistantError

from .constants import (
    AI_COOLDOWN_SECONDS,
    RENDER_FAILURES_BEFORE_COOLDOWN,
    TTS_COOLDOWN_SECONDS,
)


class RenderPart(StrEnum):
    """The engine a render step talks to; each keeps its own failures and its own pause."""

    AI = "ai"
    TTS = "tts"

    @property
    def label(self) -> str:
        """Return how the part is named in a log line."""
        return "AI engine" if self is RenderPart.AI else "TTS engine"


class EngineRenderError(MusicAssistantError):
    """A render step that failed on one engine, so only that engine's pause is owed."""

    def __init__(self, part: RenderPart, engine: str, message: str) -> None:
        """
        Initialize the error with the engine that failed.

        :param part: The engine the failed step talked to.
        :param engine: The engine's own name, as the log line says it.
        :param message: What went wrong.
        """
        super().__init__(message)
        self.part = part
        self.engine = engine


@dataclass(slots=True)
class EngineHealth:
    """Consecutive failures of one engine and the cool-down they trigger."""

    cooldown_seconds: float
    failures: int = 0
    # loop time the cool-down ends; None while this engine renders freely
    cooldown_until: float | None = None

    def cooling_down(self, now: float) -> bool:
        """Return whether this engine must not be called at the given loop time."""
        return self.cooldown_until is not None and now < self.cooldown_until

    def record_failure(self, now: float) -> bool:
        """Count a failed call and return whether that starts the cool-down."""
        self.failures += 1
        # a render that was already in flight when the pause started fails late: that
        # neither extends the pause nor announces it again
        if self.failures < RENDER_FAILURES_BEFORE_COOLDOWN or self.cooling_down(now):
            return False
        self.cooldown_until = now + self.cooldown_seconds
        return True

    def record_success(self, now: float) -> None:
        """Clear the failure count after a call that came back outside the cool-down."""
        # a call that was already in flight when the pause started proves nothing about the
        # engine now, so the pause runs its course and the first call after it decides
        if self.cooling_down(now):
            return
        self.failures = 0
        self.cooldown_until = None


@dataclass(slots=True)
class RenderHealth:
    """What one queue DJ remembers about the engines its renders talk to."""

    # a query and a voice fail for different reasons and come back on their own, so each
    # bucket counts its own failures and pauses for its own time
    ai: EngineHealth = field(default_factory=lambda: EngineHealth(AI_COOLDOWN_SECONDS))
    tts: EngineHealth = field(default_factory=lambda: EngineHealth(TTS_COOLDOWN_SECONDS))

    def bucket(self, part: RenderPart) -> EngineHealth:
        """Return the failure bookkeeping of the given engine."""
        return self.ai if part is RenderPart.AI else self.tts

    def cooling_down(self, now: float) -> bool:
        """Return whether a contribution can not be rendered at all."""
        return self.ai.cooling_down(now) or self.tts.cooling_down(now)

    def ai_cooling_down(self, now: float) -> bool:
        """Return whether the AI engine is paused, so no new script can be written."""
        return self.ai.cooling_down(now)

    def tts_cooling_down(self, now: float) -> bool:
        """Return whether the voice is paused, so no clip can be spoken at all."""
        return self.tts.cooling_down(now)

    def record_failure(self, part: RenderPart, now: float) -> bool:
        """Count a failed call of the given engine, returning whether its pause starts."""
        return self.bucket(part).record_failure(now)

    def record_success(self, part: RenderPart, now: float) -> None:
        """Clear one engine's count after a call that used it came back."""
        self.bucket(part).record_success(now)

    def record_render_success(self, now: float) -> None:
        """Clear both counts after a whole contribution came back."""
        for part in RenderPart:
            self.bucket(part).record_success(now)

    def pause_seconds(self, part: RenderPart) -> float:
        """Return how long the given engine pauses when its failures add up."""
        return self.bucket(part).cooldown_seconds

    def resume_in(self, now: float) -> float:
        """Return the seconds until no engine is paused anymore."""
        ends = [
            health.cooldown_until
            for health in (self.ai, self.tts)
            if health.cooldown_until is not None and health.cooldown_until > now
        ]
        return max(ends) - now if ends else 0.0


def engine_pause_message(queue_id: str, health: RenderHealth, part: RenderPart, engine: str) -> str:
    """
    Return the one line that announces a paused engine, naming that engine and its time.

    :param queue_id: The queue whose breaks pause.
    :param health: The failure bookkeeping that just tripped.
    :param part: The engine that tripped its pause.
    :param engine: The engine's own name, as the failing call named it.
    """
    return (
        f"AI Radio pauses the breaks on queue {queue_id} for "
        f"{round(health.pause_seconds(part) / 60)} minutes: the {part.label} failed "
        f"{health.bucket(part).failures} times in a row, the last one on {engine}. Check that "
        "engine's own log for the reason (for a Home Assistant engine that is the Home "
        "Assistant core log); a cloud engine may be out of quota."
    )


@dataclass(slots=True)
class Slot:
    """Insertion slot between source tracks."""

    when: str
    at_index: int
    prev_index: int | None
    next_index: int | None
    very_next_index: int | None
    minute_mark: float


@dataclass(slots=True)
class PlannedSection:
    """A section that should be generated for a run."""

    order: int
    clip_id: str
    section_id: str
    section_name: str
    when: str
    insert_at_index: int
    prompt: str
    max_chars: int
    web_search_mode: str
    # when true, a failed weather fetch skips the clip instead of airing it without a forecast
    weather_required: bool = False
    # when true, the section's break may be split so its tail plays over the next record's intro
    allow_post: bool = False
    # when true, the break may start over the running record's tail (a "pre"); with both on
    # the break straddles the boundary instead of sitting entirely on one side of it
    allow_pre: bool = False
    # the guard history events this plan claimed, as (section_id, (song, minute)). a caller
    # that drops the plan can drop these too, so a clip that never aired carries no weight
    history_events: list[tuple[str, tuple[int, float]]] = field(default_factory=list)


@dataclass(slots=True)
class DJQueueState:
    """State container for one sticky queue DJ."""

    queue_id: str
    host_id: str
    dj_session_id: str
    # non-empty when this DJ was auto-armed by playing a show's radio item, empty for a
    # manual pick (which is never auto-detached)
    station_id: str = ""
    # how loud this queue's DJ speaks, and how far it ducks the music under itself; the
    # duck of None keeps the depth the streams bus mixes every queue with
    voice_level_db: float = 0.0
    duck_db: float | None = None
    clip_counter: int = 0
    songs_before_window: int = 0
    minutes_before_window: float = 0.0
    # queue_item_ids of the tracks whose preceding gap the item path already settled, by
    # injecting a clip, by leaving it empty on purpose or because it became unusable. only
    # the item path (bus off) keeps this ledger: with the bus on, the term store is the one
    # that answers whether a boundary is settled
    decided_gap_ids: set[str] = field(default_factory=set)
    history: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    # a freshly armed state may only plan once the previous host's clips are cleared
    ready: bool = False
    # the program bus was on for the last pass, so the item path's clips have already been
    # dropped; only a fresh switch to on has to purge again
    bus_took_over: bool = False
    replan_pending: bool = False
    # a term pump is running or on its way; a request that lands meanwhile is served by the
    # pass that follows it instead of starting a second one
    pump_pending: bool = False
    # loop time until which the intro of this session may still be rendering; breaks are
    # held back meanwhile, and past it they are planned regardless
    intro_until: float | None = None
    # the session's intro is still owed: nothing could air it yet, so it is spoken once the
    # queue plays and its stream carries programs
    intro_pending: bool = False
    # the queue item the intro belongs to, so a skip or another album loaded before the
    # playback starts does not get the intro written for the previous content
    intro_item_id: str | None = None
    # the session's intro reached the stream, so it is never spoken twice
    intro_placed: bool = False
    render_health: RenderHealth = field(default_factory=RenderHealth, repr=False, compare=False)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    task: asyncio.Task[Any] | None = field(default=None, repr=False, compare=False)

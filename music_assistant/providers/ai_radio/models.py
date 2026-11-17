"""Data models for AI Radio."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .constants import RENDER_COOLDOWN_SECONDS, RENDER_FAILURES_BEFORE_COOLDOWN


@dataclass(slots=True)
class RenderHealth:
    """Consecutive render failures of one queue DJ and the cool-down they trigger."""

    failures: int = 0
    # loop time the cool-down ends; None while the DJ renders freely
    cooldown_until: float | None = None

    def cooling_down(self, now: float) -> bool:
        """Return whether the DJ must not render at the given loop time."""
        return self.cooldown_until is not None and now < self.cooldown_until

    def record_failure(self, now: float) -> bool:
        """Count a failed render and return whether that starts the cool-down."""
        self.failures += 1
        # a render that was already in flight when the pause started fails late: that
        # neither extends the pause nor announces it again
        if self.failures < RENDER_FAILURES_BEFORE_COOLDOWN or self.cooling_down(now):
            return False
        self.cooldown_until = now + RENDER_COOLDOWN_SECONDS
        return True

    def record_success(self, now: float) -> None:
        """Clear the failure count after a render that succeeded outside the cool-down."""
        # a render that was already in flight when the pause started proves nothing about
        # the engine now, so the pause runs its course and the first render after it decides
        if self.cooling_down(now):
            return
        self.failures = 0
        self.cooldown_until = None


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

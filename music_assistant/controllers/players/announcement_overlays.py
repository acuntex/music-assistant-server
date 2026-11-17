"""Retention and re-anchoring of announcements mixed into a queue's stream."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from music_assistant.controllers.streams.program_bus_registry import (
    ProgramBusChange,
    ProgramBusEvent,
)
from music_assistant.controllers.streams.program_schedule import (
    PRIORITY_ANNOUNCEMENT,
    ProgramKind,
    ScheduleReason,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from music_assistant.controllers.streams.controller import StreamsController

# prefix of an announcement's overlay label; every placement appends its own id
ANNOUNCEMENT_OVERLAY_LABEL = "announcement"

# how long a rendered announcement is kept so it can be anchored again after a bus change
RETENTION_SECONDS = 300.0

# how often a queue's retained clips are checked against that deadline. A queue that renders a
# clip and never hears it (the player stands, the bus is gone) gets nothing else to look at it,
# so the retention runs its own tick instead of pruning only when the next clip is rendered
SWEEP_SECONDS = 60.0

# above this many pending clips of one queue the retention says so once, but keeps them:
# a rendered announcement is not silently dropped
SOFT_RETAINED_PER_QUEUE = 16

# the last resort, so one runaway queue can not pile up PCM without bound
HARD_RETAINED_PER_QUEUE = 64

# the bytes one queue may keep retained. The clip ceiling above counts clips, and a clip is
# seconds of PCM: 64 of them are hundreds of megabytes whatever the stream's format, so the
# budget is what really bounds a queue
RETAINED_BYTES_PER_QUEUE = 32 * 1024 * 1024


# the ledger rows of one queue, keyed by label: (state, reason, playing right now)
LedgerRows = dict[str, list[tuple[str, str | None, bool]]]


def new_announcement_label() -> str:
    """Return the label of one announcement placement, unique per placement."""
    return f"{ANNOUNCEMENT_OVERLAY_LABEL}:{uuid4().hex}"


@dataclass
class _RetainedAnnouncement:
    """A rendered announcement that has not been heard to its end yet."""

    queue_id: str
    label: str
    pcm: bytes
    placed_at: float


class OverlayAnnouncementRetention:
    """
    Keep rendered announcements until they aired, and anchor them again after a bus change.

    An announcement that was already mixed in is only heard once: a placement the ledger
    reports as ``aired`` is forgotten, one the ledger reports as ``orphaned`` (the bus that
    took it was replaced or ended) is placed on the new bus, and everything else is left
    alone. A placement older than ``RETENTION_SECONDS`` is dropped instead of re-anchored.
    """

    def __init__(
        self,
        streams: StreamsController,
        logger: logging.Logger,
        clock: Callable[[], float] = time.monotonic,
        sweep_seconds: float = SWEEP_SECONDS,
        bytes_per_queue: int = RETAINED_BYTES_PER_QUEUE,
    ) -> None:
        """
        Initialize the retention and subscribe to the queue bus changes.

        :param streams: The streams controller that mixes the announcements in.
        :param logger: Logger of the announcement controller.
        :param clock: Monotonic clock, for the retention deadline.
        :param sweep_seconds: Seconds between two of the retention's own prune ticks.
        :param bytes_per_queue: How many bytes one queue may keep retained at most.
        """
        self._streams = streams
        self._logger = logger
        self._clock = clock
        self._sweep_seconds = max(0.0, sweep_seconds)
        self._bytes_per_queue = max(0, bytes_per_queue)
        self._retained: dict[tuple[str, str], _RetainedAnnouncement] = {}
        self._soft_limit_reported: set[str] = set()
        # the pending tick that prunes the retention on its own, None while nothing is held
        self._sweep_handle: asyncio.TimerHandle | None = None
        self._remove_listener = streams.add_program_bus_listener(self.on_bus_change)
        self._remove_aired_listener = streams.add_program_aired_listener(self._on_aired)

    def remember(self, queue_id: str, label: str, pcm: bytes) -> None:
        """
        Keep a rendered announcement until the ledger says it aired.

        :param queue_id: Queue the announcement was mixed into.
        :param label: Label the announcement was placed with.
        :param pcm: The rendered announcement audio.
        """
        states = self._prune()
        self._retained[(queue_id, label)] = _RetainedAnnouncement(
            queue_id=queue_id,
            label=label,
            pcm=pcm,
            placed_at=self._clock(),
        )
        self._enforce_limit(states)
        self._arm_sweep()

    def forget_queue(self, queue_id: str) -> None:
        """
        Release every clip retained for a queue that is gone.

        :param queue_id: Queue whose clips are released.
        """
        for key in [key for key in self._retained if key[0] == queue_id]:
            del self._retained[key]
        self._soft_limit_reported.discard(queue_id)

    def forget(self, queue_id: str, label: str) -> None:
        """
        Drop a retained announcement.

        :param queue_id: Queue the announcement was mixed into.
        :param label: Label the announcement was placed with.
        """
        self._retained.pop((queue_id, label), None)

    def retained_labels(self) -> tuple[str, ...]:
        """Return the labels of the announcements still being retained."""
        self._prune()
        return tuple(retained.label for retained in self._retained.values())

    def on_bus_change(self, event: ProgramBusEvent) -> None:
        """
        Re-anchor orphaned announcements of the queue that got a bus again.

        :param event: The bus change that was announced.
        """
        if event.change not in (ProgramBusChange.REGISTERED, ProgramBusChange.REPLACED):
            return
        self._reanchor(event.queue_id)

    def close(self) -> None:
        """Release every retained clip and stop listening to the queue buses."""
        self._retained.clear()
        if self._sweep_handle is not None:
            self._sweep_handle.cancel()
            self._sweep_handle = None
        for remove in (self._remove_listener, self._remove_aired_listener):
            with suppress(ValueError):
                remove()

    def _on_aired(self, queue_id: str, label: str) -> None:
        """Release the clip the ledger just heard to its end."""
        self.forget(queue_id, label)

    def _arm_sweep(self) -> None:
        """Ask the loop for the next prune tick, while any clip is still being retained."""
        if self._sweep_handle is not None or not self._retained:
            # a tick is on its way already, or there is nothing left for one to look at
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # the retention is only filled from the running server; without a loop there is
            # nothing to tick, and what is held is pruned on the next access
            return
        self._sweep_handle = loop.call_later(self._sweep_seconds, self._sweep)

    def _sweep(self) -> None:
        """Prune the retained clips on their deadline, and tick again while one is left."""
        self._sweep_handle = None
        self._enforce_limit(self._prune())
        self._arm_sweep()

    def _prune(self) -> dict[str, LedgerRows]:
        """
        Release what the ledger reports as aired, and what outlived its retention.

        :return: The ledger rows that were read, grouped by queue and label.
        """
        now = self._clock()
        states = self._refresh_states()
        for key, retained in list(self._retained.items()):
            if now - retained.placed_at > RETENTION_SECONDS:
                del self._retained[key]
                continue
            rows = states[retained.queue_id].get(retained.label, [])
            if any(state == "aired" for state, _reason, _playing in rows):
                del self._retained[key]
        return states

    def _refresh_states(self) -> dict[str, LedgerRows]:
        """Read the ledger once per retained queue; the read also refreshes its deadlines."""
        return {
            queue_id: self._states(queue_id)
            for queue_id in {retained.queue_id for retained in self._retained.values()}
        }

    def _states(self, queue_id: str) -> LedgerRows:
        """Return the ledger rows of the queue, grouped by label."""
        states: LedgerRows = {}
        for state in self._streams.overlay_state(queue_id):
            states.setdefault(state.label, []).append(
                (str(state.state), state.reason, state.playing)
            )
        return states

    def _enforce_limit(self, states: dict[str, LedgerRows]) -> None:
        """
        Report a queue above the soft limit, and bound only a runaway one.

        :param states: The ledger rows the caller just read, per queue and label.
        """
        per_queue: dict[str, list[_RetainedAnnouncement]] = {}
        for retained in self._retained.values():
            per_queue.setdefault(retained.queue_id, []).append(retained)
        for queue_id, retained_clips in per_queue.items():
            playing = {
                label
                for label, rows in states.get(queue_id, {}).items()
                if any(row_playing for _state, _reason, row_playing in rows)
            }
            if len(retained_clips) > SOFT_RETAINED_PER_QUEUE:
                if queue_id not in self._soft_limit_reported:
                    self._soft_limit_reported.add(queue_id)
                    self._logger.warning(
                        "Queue %s has %d announcements waiting to air - keeping them all",
                        queue_id,
                        len(retained_clips),
                    )
            # the hard ceiling is the last resort, so it keeps its bound even when nothing
            # but a playing clip is left
            self._give_up(queue_id, retained_clips, playing, keep_playing=False)
            self._give_up(
                queue_id,
                retained_clips,
                playing,
                keep_playing=True,
                budget=self._bytes_per_queue,
            )

    def _give_up(
        self,
        queue_id: str,
        retained_clips: list[_RetainedAnnouncement],
        playing: set[str],
        *,
        keep_playing: bool,
        budget: int | None = None,
    ) -> None:
        """
        Give up the oldest clips until the queue fits the clip ceiling or the byte budget.

        :param queue_id: Queue the clips belong to.
        :param retained_clips: The queue's retained clips, refreshed in place.
        :param playing: Labels the ledger reports as being heard right now.
        :param keep_playing: Whether a clip that is being heard is never given up.
        :param budget: Byte budget to stay inside, or None for one more clip over the ceiling.
        """
        while self._over_limit(retained_clips, budget):
            oldest = next(
                (
                    retained
                    for retained in sorted(retained_clips, key=lambda clip: clip.placed_at)
                    if not keep_playing or retained.label not in playing
                ),
                None,
            )
            if oldest is None:
                # every clip left is being heard right now: dropping one would end an
                # announcement mid-word, which is worse than being over the budget
                self._logger.info(
                    "Queue %s holds %d announcements being heard - keeping them over its budget",
                    queue_id,
                    len(retained_clips),
                )
                return
            if budget is None:
                self._logger.warning(
                    "Announcement %s is not kept: queue %s holds more than %d clips",
                    oldest.label,
                    queue_id,
                    HARD_RETAINED_PER_QUEUE,
                )
            else:
                self._logger.warning(
                    "Announcement %s is not kept: queue %s holds more than %d bytes",
                    oldest.label,
                    queue_id,
                    budget,
                )
            del self._retained[(oldest.queue_id, oldest.label)]
            retained_clips.remove(oldest)

    @staticmethod
    def _over_limit(retained_clips: list[_RetainedAnnouncement], budget: int | None) -> bool:
        """
        Return whether a queue's clips are still over its clip ceiling or its byte budget.

        :param retained_clips: The queue's retained clips.
        :param budget: Byte budget to measure against, or None to measure the clip ceiling.
        """
        if budget is None:
            return len(retained_clips) > HARD_RETAINED_PER_QUEUE
        return sum(len(retained.pcm) for retained in retained_clips) > budget

    def _reanchor(self, queue_id: str) -> None:
        """Place every orphaned announcement of the queue on the bus it now has."""
        states = self._states(queue_id)
        for retained in list(self._retained.values()):
            if retained.queue_id != queue_id:
                continue
            rows = states.get(retained.label, [])
            label_states = {state for state, _reason, _playing in rows}
            if "aired" in label_states:
                self.forget(queue_id, retained.label)
                continue
            if any(
                state == "orphaned" and reason == ScheduleReason.LEDGER_EVICTED
                for state, reason, _playing in rows
            ):
                # the ledger gave this placement up, so anchoring it again would play it twice
                self._logger.info(
                    "Announcement %s is not anchored again: the ledger gave it up",
                    retained.label,
                )
                self.forget(queue_id, retained.label)
                continue
            if self._clock() - retained.placed_at > RETENTION_SECONDS:
                # a clip that stays scheduled (the player sat idle before its deadline) must
                # not live forever just because the bus changed
                self._logger.info(
                    "Announcement %s is not anchored again: its retention time is over",
                    retained.label,
                )
                self.forget(queue_id, retained.label)
                continue
            if "scheduled" in label_states or "orphaned" not in label_states:
                # still to be heard, or never lost its bus: nothing to anchor
                continue
            self._place(retained)

    def _place(self, retained: _RetainedAnnouncement) -> None:
        """Place a retained announcement on the queue's current bus."""
        result = self._streams.schedule_overlay(
            retained.queue_id,
            retained.pcm,
            label=retained.label,
            priority=PRIORITY_ANNOUNCEMENT,
            kind=ProgramKind.ANNOUNCEMENT,
        )
        if result.scheduled:
            self._logger.info(
                "Announcement %s was anchored again on the queue's new bus", retained.label
            )
            return
        self._logger.info(
            "Announcement %s could not be anchored again: %s",
            retained.label,
            result.reason,
        )
        if result.reason in (ScheduleReason.NO_BUS, ScheduleReason.BUS_REPLACED):
            # the next bus change is the one that can take it
            return
        self.forget(retained.queue_id, retained.label)

"""Sticky per-queue AI DJ for AI Radio."""
# mypy: disable-error-code=attr-defined

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

import aiofiles
from music_assistant_models.enums import EventType, PlaybackState
from music_assistant_models.errors import InvalidDataError

from music_assistant.controllers.player_queues.helpers import committed_index
from music_assistant.helpers.json import async_json_loads

from .constants import (
    ATTR_GAP_NEXT_ID,
    ATTR_QUEUE_DJ,
    ATTR_SESSION_ID,
    EVENT_QUEUE_DJ_UPDATED,
    FALLBACK_TRACK_SECONDS,
    PROGRAM_BUS_OFF_HINT,
    PROGRAM_BUS_OFF_LINK,
)
from .helpers import check_player_access, has_player_access
from .hosts import host_plays_outro
from .models import DJQueueState, EngineRenderError, PlannedSection, engine_pause_message
from .queue_audio import (
    QueueAudioLevels,
    levels_from_stored,
    normalize_duck_db,
    normalize_voice_level_db,
)
from .term_engine import (
    gap_can_air as _gap_can_air,
)
from .term_engine import (
    running_boundary,
    term_boundary_plans,
    term_plans_in_horizon,
)
from .term_engine import (
    section_boundary as _section_boundary,
)
from .terms import BoundaryKey
from .upgrade import remove_legacy_dj_items

if TYPE_CHECKING:
    from pathlib import Path

    from music_assistant_models.event import MassEvent
    from music_assistant_models.media_items import AudioFormat
    from music_assistant_models.player_queue import PlayerQueue
    from music_assistant_models.queue_item import QueueItem

    from music_assistant.mass import MusicAssistant

QUEUE_PAGE_SIZE = 500

# per section history cap, generous for the widest guard window (60 minutes)
HISTORY_EVENTS_PER_SECTION = 50

# result of one splice attempt. only "gap_gone" leaves its gap unserved: a gap that already
# holds a clip is served, and one that slipped behind the player can never be served again
DJSpliceOutcome = Literal["injected", "gap_gone", "too_close", "occupied"]


def _can_play(item: QueueItem | None) -> bool:
    """
    Return whether a queue item can still deliver audio.

    :param item: The queue item to check, or None.
    """
    return item is not None and bool(getattr(item, "available", True))


class AIRadioQueueDJMixin:
    """Mixin managing sticky queue DJ state and clip injection."""

    if TYPE_CHECKING:
        mass: MusicAssistant
        logger: logging.Logger
        _hosts: dict[str, dict[str, Any]]
        _stations: dict[str, dict[str, Any]]
        _dj_queues: dict[str, DJQueueState]
        _queue_levels: dict[str, QueueAudioLevels]
        _armed_show_queues: dict[str, str]
        _sections: dict[str, dict[str, Any]]
        _dj_file: Path
        _dj_lock: asyncio.Lock
        _unloading: bool

        def _station_id_from_source_uri(self, uri: str | None) -> str | None: ...

        def _render_ahead(self, queue_id: str) -> None: ...

        def get_config_value(
            self, key: str, default: Any = None, *, return_type: Any = None
        ) -> Any:
            """Return the provider's configured value for the given key."""

        def _build_program(
            self, station: dict[str, Any], host: dict[str, Any]
        ) -> dict[str, Any]: ...

        def _plan_sections(
            self,
            session_id: str,
            tracks: list[dict[str, Any]],
            program: dict[str, Any],
            track_index_offset: int,
            minute_offset: float,
            history_state: dict[str, list[tuple[int, float]]],
            allowed_slot_when: list[str] | None,
            runtime_tokens: dict[str, str],
            decided_next_item_ids: set[str] | None = None,
        ) -> tuple[list[PlannedSection], dict[str, list[tuple[int, float]]]]: ...

        async def _prepare_runtime_tokens(self, program: dict[str, Any]) -> dict[str, str]: ...

        async def _render_section_pcm(
            self,
            queue_id: str,
            program: dict[str, Any],
            section: PlannedSection,
            pcm_format: AudioFormat,
        ) -> bytes: ...

        def _decide_terms(
            self,
            queue_id: str,
            program: dict[str, Any],
            plans: list[Any],
            *,
            history: dict[str, list[tuple[int, float]]] | None = None,
            horizon: set[tuple[str, str | None]] | None = None,
        ) -> None: ...

        def _term_gap_horizon(
            self,
            queue_id: str,
            window_tracks: list[dict[str, Any]],
            *,
            reserve: int = 0,
            index_base: int = 0,
        ) -> set[tuple[str, str | None]]:
            """Return the boundaries a pass may decide now."""

        def _term_outro_reserve(self, queue_id: str, outro_target: dict[str, Any] | None) -> int:
            """Return the store room a show's outro has to keep free."""

        def _term_holds_boundary(self, queue_id: str, boundary: Any) -> bool:
            """Return whether the queue already holds a term for the given boundary."""

        async def _pump_terms(self, queue_id: str) -> None: ...

        def _reset_terms(self, queue_id: str) -> None: ...

    async def set_queue_dj(self, queue_id: str, host_id: str | None) -> dict[str, dict[str, Any]]:
        """
        Enable, switch or disable the sticky AI DJ on a queue.

        :param queue_id: The queue to change.
        :param host_id: The host to enable, or None to disable.
        :return: The full per-queue DJ status after the change.
        """
        queue_id = str(queue_id).strip()
        if not queue_id:
            raise InvalidDataError("queue_id is required")
        check_player_access(queue_id)
        armed: DJQueueState | None = None
        try:
            async with self._dj_lock:
                if host_id is None:
                    self._drop_queue_state(queue_id)
                else:
                    host_id = str(host_id).strip()
                    if host_id not in self._hosts:
                        raise InvalidDataError(f"Unknown host id: {host_id}")  # noqa: TRY301 - re-raised after cleanup
                    previous = self._dj_queues.get(queue_id)
                    armed = self._arm_dj_state(queue_id, host_id)
                    if previous is not None:
                        # a mid-show host switch keeps the show binding and the level the
                        # queue is mixed at: both belong to the queue, not to one host
                        armed.station_id = previous.station_id
                        armed.voice_level_db = previous.voice_level_db
                        armed.duck_db = previous.duck_db
                        if armed.station_id:
                            self._apply_stream_claims(queue_id, host_id)
                await self._write_queue_dj()
            # stale clips carry the old host's persona, so they must go before a replan
            # can reuse the gaps they occupy
            self._remove_pending_dj_clips(queue_id)
        except Exception:
            # an armed state that never reached its cleanup stays unready forever, which
            # reads as an armed DJ that never speaks. dropping it lets a retry arm cleanly
            if armed is not None:
                async with self._dj_lock:
                    if self._dj_queues.get(queue_id) is armed:
                        self._drop_queue_state(queue_id)
                        self.signal_provider_event({"event": EVENT_QUEUE_DJ_UPDATED})
            raise
        if armed is not None:
            if self._dj_queues.get(queue_id) is armed:
                # only if we're still the live state: a newer switch may have replaced us.
                # flipped after cleanup so a racing pass doesn't mark old clips' gaps served
                armed.ready = True
                # the level is set before the intro renders, so it is spoken at it too
                self._apply_queue_levels(queue_id)
                self._start_intro_overlay(queue_id, armed)
            # the DJ is set up for its player, which is kept also while a browser or app is away
            self.mass.players.keep_player(queue_id)
            self._schedule_replan(queue_id)
        return await self.get_queue_dj_status()

    async def get_queue_dj_status(self) -> dict[str, dict[str, Any]]:
        """
        Return, for the queues the calling user may see, the armed host and show station.

        A DJ armed while the experimental program bus is off is reported with a hint and the
        path of the settings that switch it on, so a client can say why nothing is spoken.
        """
        # the switch is not this plugin's, so an armed DJ that can not be heard says where
        # to switch it on instead of refusing to arm
        bus_enabled = self.mass.streams.program_bus_enabled()
        hint = None if bus_enabled else PROGRAM_BUS_OFF_HINT
        hint_link = None if bus_enabled else PROGRAM_BUS_OFF_LINK
        return {
            queue_id: {
                "host_id": state.host_id,
                "station_id": state.station_id,
                "voice_level_db": state.voice_level_db,
                "duck_db": state.duck_db,
                "bus_enabled": bus_enabled,
                "hint": hint,
                "hint_link": hint_link,
            }
            for queue_id, state in self._dj_queues.items()
            if has_player_access(queue_id)
        }

    async def set_queue_dj_voice_level(
        self,
        queue_id: str,
        voice_level_db: float | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Set how loud a queue's DJ speaks, leaving the ducking depth as it is.

        The two levels are independent settings, and each has its own command: a client that
        saves one slider must not reset the other one to its default by not sending it.

        :param queue_id: The queue to change.
        :param voice_level_db: Gain of the spoken program in decibels, 0 to -9 dB. None is
            0 dB, the level a DJ that never set one speaks at.
        :return: The full per-queue DJ status after the change.
        """
        queue_id = self._checked_queue_id(queue_id)
        level = normalize_voice_level_db(voice_level_db)
        async with self._dj_lock:
            state = self._online_queue_dj(queue_id)
            state.voice_level_db = level
            await self._write_queue_dj()
        self._apply_queue_levels(queue_id)
        return await self.get_queue_dj_status()

    async def set_queue_dj_duck(
        self,
        queue_id: str,
        duck_db: float | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Set how far a queue's music ducks under its DJ, leaving the voice level as it is.

        :param queue_id: The queue to change.
        :param duck_db: Level the music goes down to under a program, 0 to -12 dB, or None
            to keep the ducking depth the streams bus mixes every queue with.
        :return: The full per-queue DJ status after the change.
        """
        queue_id = self._checked_queue_id(queue_id)
        depth = normalize_duck_db(duck_db)
        async with self._dj_lock:
            state = self._online_queue_dj(queue_id)
            state.duck_db = depth
            await self._write_queue_dj()
        self._apply_queue_levels(queue_id)
        return await self.get_queue_dj_status()

    async def _load_queue_dj(self) -> None:
        """Load persisted queue DJ assignments and arm their states."""
        file_exists = await asyncio.to_thread(self._dj_file.exists)
        if not file_exists:
            self._dj_queues = {}
            self._queue_levels = {}
            return
        async with aiofiles.open(self._dj_file) as file_handle:
            content = await file_handle.read()
        try:
            payload = await async_json_loads(content)
        except ValueError as err:
            self.logger.error("Queue DJ file is corrupt, starting without queue DJs: %s", err)
            payload = {}
        queues = payload.get("queues", {}) if isinstance(payload, dict) else {}
        self._dj_queues = {}
        self._queue_levels = {}
        if isinstance(queues, dict):
            for queue_id, entry in queues.items():
                host_id = str(entry.get("host_id", "")).strip() if isinstance(entry, dict) else ""
                if host_id not in self._hosts:
                    self.logger.warning(
                        "Dropping queue DJ for %s: host %s no longer exists", queue_id, host_id
                    )
                    continue
                # no clip cleanup precedes a boot arm, so this state may plan right away
                state = self._arm_dj_state(str(queue_id), host_id)
                state.station_id = str(entry.get("station_id", "")).strip()
                levels = levels_from_stored(entry)
                state.voice_level_db = levels.voice_level_db
                state.duck_db = levels.duck_db
                # the level belongs to the queue, so an arm after this DJ is gone keeps it
                self._remember_queue_levels(str(queue_id))
                state.ready = True
                # the queue's level belongs to the queue, so the stream it opens next is
                # mixed at it without waiting for the DJ to set it again
                self._apply_queue_levels(str(queue_id))
                if self.mass.streams.program_bus_enabled():
                    # the bus speaks no clip items, so what the item path settled about the
                    # gaps and the guards of a restored session is not its to keep
                    self._forget_item_path_decisions(state)
                # a restored DJ has no terms yet, but the stream it will need is claimed
                # now: the claim takes effect at the next play start, and it is the same
                # claim the arming path places, whichever side of the switch is on
                self._apply_stream_claims(str(queue_id), host_id)

    async def _write_queue_dj(self) -> None:
        """Persist queue DJ assignments to disk."""
        payload = {
            "version": 1,
            "queues": {
                queue_id: {
                    "host_id": state.host_id,
                    "station_id": state.station_id,
                    "voice_level_db": state.voice_level_db,
                    "duck_db": state.duck_db,
                }
                for queue_id, state in sorted(self._dj_queues.items())
            },
        }
        await self._write_json_file(self._dj_file, payload)
        self.signal_provider_event({"event": EVENT_QUEUE_DJ_UPDATED})

    def _arm_dj_state(self, queue_id: str, host_id: str) -> DJQueueState:
        """Create fresh in-memory DJ state for a queue."""
        # a fresh session id per arm keeps clip ids from colliding with clips
        # persisted in the queue by a previous run of this provider
        state = DJQueueState(
            queue_id=queue_id,
            host_id=host_id,
            dj_session_id=f"dj{uuid4().hex[:12]}",
        )
        levels = getattr(self, "_queue_levels", {}).get(queue_id)
        if levels is not None:
            # the level belongs to the queue, so a later arm speaks at it again
            state.voice_level_db = levels.voice_level_db
            state.duck_db = levels.duck_db
        self._dj_queues[queue_id] = state
        return state

    def _drop_queue_state(self, queue_id: str, *, keep_levels: bool = True) -> None:
        """
        Drop the queue's DJ state, its terms and the stream claims it held.

        :param queue_id: The queue whose DJ is dropped.
        :param keep_levels: Whether the queue keeps the level it was set to. A queue that is
            gone has no level left to speak at; a DJ that was only switched off, detached or
            replaced keeps it, because it belongs to the queue and not to one arm.
        """
        self._remember_queue_levels(queue_id, keep=keep_levels)
        self._dj_queues.pop(queue_id, None)
        self._reset_terms(queue_id)
        self._release_stream_claims(queue_id)
        # a DJ that is gone leaves no gain behind on what else the queue's stream mixes
        self._apply_queue_levels(queue_id)

    def _remember_queue_levels(self, queue_id: str, *, keep: bool = True) -> None:
        """
        Hold the level the queue was set to, or forget it with the queue.

        :param queue_id: The queue whose level is held.
        :param keep: Whether to hold the level instead of forgetting it.
        """
        levels_by_queue: dict[str, QueueAudioLevels] = getattr(self, "_queue_levels", {})
        if not hasattr(self, "_queue_levels"):
            self._queue_levels = levels_by_queue
        if not keep:
            levels_by_queue.pop(queue_id, None)
            return
        state = self._dj_queues.get(queue_id)
        if state is None:
            return
        levels_by_queue[queue_id] = QueueAudioLevels(
            voice_level_db=state.voice_level_db, duck_db=state.duck_db
        )

    def _apply_queue_levels(self, queue_id: str) -> None:
        """Hand the queue's level to the bus that mixes its programs."""
        state = self._dj_queues.get(queue_id)
        levels = (
            QueueAudioLevels()
            if state is None
            else QueueAudioLevels(voice_level_db=state.voice_level_db, duck_db=state.duck_db)
        )
        self.mass.streams.set_program_levels(
            queue_id, voice_db=levels.voice_level_db, duck_db=levels.duck_db
        )

    @staticmethod
    def _checked_queue_id(queue_id: str) -> str:
        """Return the queue id of a DJ setting, refusing an empty one and one out of reach."""
        checked = str(queue_id).strip()
        if not checked:
            raise InvalidDataError("queue_id is required")
        check_player_access(checked)
        return checked

    def _online_queue_dj(self, queue_id: str) -> DJQueueState:
        """Return the queue's armed DJ, refusing a queue that runs none."""
        state = self._dj_queues.get(queue_id)
        if state is None:
            raise InvalidDataError(f"Queue {queue_id} has no AI DJ")
        return state

    async def _ensure_show_dj(self, queue_id: str) -> None:
        """Arm, once per play, or re-bind after a restart the show's host on a show queue."""
        station_id = self._queue_show_station(queue_id)
        if station_id is None:
            return
        state = self._dj_queues.get(queue_id)
        if state is not None:
            # an empty station_id is a manual pick, which is never bound to the show
            if state.station_id == station_id:
                self._armed_show_queues[queue_id] = station_id
            # a different station means the show was replaced: the stale state detaches on
            # this same event, and the next one arms the new show's host afresh
            return
        if self._armed_show_queues.get(queue_id) == station_id:
            # already armed once for this play (a detach or a manual disable must not be
            # re-armed by a later event)
            return
        queue = self.mass.player_queues.get(queue_id)
        if queue is None or queue.ended:
            # an ended queue keeps its sources; a replay un-ends it and the next event arms
            return
        station = self._stations.get(station_id)
        if station is None:
            return
        host_id = str(station["host_id"])
        if host_id not in self._hosts:
            # set_queue_dj would raise, and this runs inside the queue event handler
            self.logger.debug(
                "Not arming the DJ for show %s: host %s no longer exists", station_id, host_id
            )
            return
        # the show's sections decide what the queue's stream must do: a break that carries
        # over a boundary needs mixed boundaries, one that starts over the outgoing tail
        # needs that tail held. Both are decided at the play start this arm precedes.
        self._apply_stream_claims(queue_id, host_id)
        try:
            await self.set_queue_dj(queue_id, host_id)
        except Exception:
            self._release_stream_claims(queue_id)
            raise
        if (state := self._dj_queues.get(queue_id)) is None:
            return
        state.station_id = station_id
        self._armed_show_queues[queue_id] = station_id
        # set_queue_dj persisted the state before it carried the binding
        async with self._dj_lock:
            await self._write_queue_dj()

    async def _maybe_detach_show_dj(self, queue_id: str) -> None:
        """Detach an auto-armed show DJ once the queue left the show."""
        state = self._dj_queues.get(queue_id)
        if state is None or not state.station_id:
            return
        queue = self.mass.player_queues.get(queue_id)
        if queue is None:
            # a dropped player comes back with its queue, only PLAYER_REMOVED drops the DJ
            return
        if self._queue_show_station(queue_id) == state.station_id and not queue.ended:
            return
        await self.set_queue_dj(queue_id, None)
        self._armed_show_queues.pop(queue_id, None)

    def _forget_armed_show(self, queue_id: str, queue_removed: bool = False) -> None:
        """Drop the queue's arm-once record once it stops sourcing the show it names."""
        armed_station_id = self._armed_show_queues.get(queue_id)
        if armed_station_id is None:
            return
        queue = None if queue_removed else self.mass.player_queues.get(queue_id)
        if not queue_removed and queue is None:
            # a dropped player keeps its record until it is back or removed
            return
        if (
            queue is not None
            and not queue.ended
            and self._queue_show_station(queue_id) == armed_station_id
        ):
            return
        self._armed_show_queues.pop(queue_id, None)

    async def _on_dj_queue_event(self, event: MassEvent) -> None:
        """Handle queue and player events for the queues that run a DJ."""
        queue_id = str(event.object_id or "")
        await self._ensure_show_dj(queue_id)
        # a queue that registered after the start still needs its old DJ clips cleared, and a
        # switch that came on since the last pass takes the item path's clips out here
        if event.event == EventType.QUEUE_ADDED:
            await self._cleanup_legacy_dj_items()
        self._sync_item_path(queue_id)
        if queue_id in self._dj_queues:
            if event.event == EventType.PLAYER_REMOVED:
                async with self._dj_lock:
                    # the queue is gone with its player, so its level has nothing left to arm
                    self._drop_queue_state(queue_id, keep_levels=False)
                    await self._write_queue_dj()
                self.logger.debug("Dropped queue DJ for removed player %s", queue_id)
            else:
                self._ensure_stream_claims(queue_id)
                self._schedule_replan(queue_id)
                if self.mass.streams.program_bus_enabled():
                    self._pump_terms_soon(queue_id)
                await self._maybe_detach_show_dj(queue_id)
                # an unready state is still clearing the previous host's clips
                state = self._dj_queues.get(queue_id)
                if state is not None and state.ready and not self._unloading:
                    self._render_ahead(queue_id)
        # an arm-once record can outlive its DJ state (manual disable, deleted host, a failed
        # arm), so it is reconciled against the queue's current sources independently here
        self._forget_armed_show(queue_id, queue_removed=event.event == EventType.PLAYER_REMOVED)

    def _schedule_replan(self, queue_id: str) -> None:
        """Request a replan pass for the given queue."""
        if self._unloading:
            return
        state = self._dj_queues.get(queue_id)
        if state is None or state.replan_pending:
            return
        state.replan_pending = True
        state.task = self.mass.create_task(
            self._drain_replans(queue_id), task_id=f"ai_radio_dj_replan_{queue_id}"
        )

    async def _drain_replans(self, queue_id: str) -> None:
        """Run replan passes until no further request landed during the last one."""
        # the inserts of a pass re-fire QUEUE_ITEMS_UPDATED while this task still holds the
        # replan task id, so its follow-up request is served here instead of by a new task
        while (state := self._dj_queues.get(queue_id)) is not None and state.replan_pending:
            if self._unloading:
                return
            try:
                await self._replan_queue(queue_id)
            except Exception:
                # cleared (not left pending) so a later event can retry without hot-looping
                # here; re-fetched since a re-arm mid-pass may have swapped in a new state
                self.logger.exception("Queue DJ replan failed for %s", queue_id)
                if (live_state := self._dj_queues.get(queue_id)) is not None:
                    live_state.replan_pending = False
                return

    async def _replan_queue(self, queue_id: str) -> None:  # noqa: PLR0915
        """Run one planning, injection and repair pass over a queue."""
        state = self._dj_queues.get(queue_id)
        if state is None:
            return
        async with state.lock:
            # cleared up front so an event landing mid-pass requests a fresh pass
            state.replan_pending = False
            if not state.ready:
                # a switch armed this state but hasn't finished clearing the old clips yet;
                # planning now would mark their gaps served. the switch replans once ready
                self.logger.debug("Queue %s is waiting for its DJ switch cleanup", queue_id)
                return
            self._sync_item_path(queue_id)
            if state.intro_pending:
                # the intro was blocked by a queue that was still loading when this DJ was
                # switched on: it is spoken now that playback and its stream are there
                self._start_intro_overlay(queue_id, state)
            if self._intro_holds_breaks(queue_id, state):
                return
            if state.render_health.cooling_down(asyncio.get_running_loop().time()):
                # the engines keep failing, so no break is planned until the cool-down ends;
                # whichever render tripped it booked the pass that runs then
                self.logger.debug(
                    "Queue %s DJ is paused after repeated render failures, skipping replan",
                    queue_id,
                )
                return
            queue = self.mass.player_queues.get(queue_id)
            if queue is None:
                # usually the queue just hasn't registered yet (players appear seconds after
                # load); state is kept, QUEUE_ADDED resumes it, PLAYER_REMOVED is what drops it
                self.logger.debug("Queue %s is not registered (yet), skipping replan", queue_id)
                return
            items = self._dj_queue_items(queue_id)
            guard_index = self._dj_guard_index(queue)
            if self._repair_dj_clips(queue_id, state, items, guard_index):
                items = self._dj_queue_items(queue_id)
            on_the_bus = self.mass.streams.program_bus_enabled()
            if not on_the_bus:
                # the term path books its boundaries in the term store instead
                self._prune_item_path_decisions(state, items, guard_index)

            window = self._dj_window(
                items, self._dj_term_window_guard(queue_id, queue, items, guard_index)
            )
            if self._nothing_to_plan(window, state):
                self.logger.debug(
                    "Queue %s has no plannable gap ahead of the player, skipping replan", queue_id
                )
                return
            # measured from the same point the planner counts from, or OPTIONAL guard
            # positions drift between passes. recomputed every pass so state self-corrects
            offsets = self._dj_window_offsets(items, window[0].queue_item_id)
            # a lower song count means the queue rewound under the history, e.g. a clear or a
            # jump back to the top; minutes dip on their own when a probed duration lands
            if offsets[0] < state.songs_before_window:
                self._rebase_dj_history(state, *offsets)
            state.songs_before_window, state.minutes_before_window = offsets
            host = self._hosts.get(state.host_id)
            if host is None:
                self.logger.warning(
                    "Disabling queue DJ on %s: host %s no longer exists", queue_id, state.host_id
                )
                async with self._dj_lock:
                    self._drop_queue_state(queue_id)
                    await self._write_queue_dj()
                return

            window_tracks = [
                self._queue_item_to_track(index, item) for index, item in enumerate(window)
            ]
            program = self._build_program({"id": "", "name": f"AI DJ {host['name']}"}, host)
            runtime_tokens = await self._prepare_runtime_tokens(program)
            if self._dj_queues.get(queue_id) is not state or state.render_health.cooling_down(
                asyncio.get_running_loop().time()
            ):
                # a switch or disable replaced this queue's state while the fetch above
                # was in flight, so the session this pass planned for is gone; or the
                # engines failed once too often meanwhile and _pause_queue_dj took the
                # pending clips out, so this pass must not put new ones in
                return
            allowed_slot_when = ["between_songs"]
            outro_target: dict[str, Any] | None = None
            if state.station_id and host_plays_outro(host):
                # the whole show is enqueued up front, so the outro belongs after the
                # queue's last song; it is only offered once that item is in the window
                outro_target = self._show_outro_target(items, window, window_tracks)
                if outro_target is not None:
                    allowed_slot_when.append("end_of_playlist")
            planned, history, horizon, evaluated_gap_ids = self._plan_and_decide(
                queue_id=queue_id,
                state=state,
                program=program,
                window_tracks=window_tracks,
                allowed_slot_when=allowed_slot_when,
                runtime_tokens=runtime_tokens,
                outro_target=outro_target,
                on_the_bus=on_the_bus,
            )
            # spliced into a working copy and applied as one update; a call per clip would
            # flood every client with queue events
            working = self._dj_queue_items(queue_id)
            # re-read: the planning awaits above can take seconds, letting the player buffer
            # further ahead
            guard_index = self._dj_guard_index(queue)
            # insert order is not load bearing: every clip resolves its own target position
            # in the working copy. descending walks back from the queue tail
            injected = 0
            skipped: dict[str, int] = {}
            rejected: list[PlannedSection] = []
            for section in sorted(planned, key=lambda item: item.insert_at_index, reverse=True):
                target, guard_at, after_target = _section_boundary(
                    section, window_tracks, outro_target
                )
                # a break written for a boundary that never airs is written, paid for and
                # dropped: the tracks on both sides of it have to be able to play
                if not _gap_can_air(window_tracks, guard_at):
                    rejected.append(section)
                    skipped["unplayable"] = skipped.get("unplayable", 0) + 1
                    continue
                outcome = self._splice_dj_clip(
                    queue_id=queue_id,
                    items=working,
                    guard_index=guard_index,
                    state=state,
                    program=program,
                    target=target,
                    section=section,
                    after_target=after_target,
                )
                if outcome == "injected":
                    injected += 1
                else:
                    rejected.append(section)
                    skipped[outcome] = skipped.get(outcome, 0) + 1
                    if outcome == "gap_gone":
                        # the target moved or left the queue, so nothing was decided about
                        # its gap and a later pass has to look at it again
                        evaluated_gap_ids.discard(str(target["item_id"]))
            if injected:
                self.mass.player_queues.update_items(queue_id, working)
            # a rejected clip never airs, so its guard event must be removed: left in, it
            # would block its own successor for a full guard window and double-count later
            for section in rejected:
                for section_id, event in section.history_events:
                    if (events := history.get(section_id)) and event in events:
                        events.remove(event)
            # only the newest events matter to the guards (the last one for min_gap_songs,
            # a 60 minute window for max_per_60min), so the tail is dropped
            state.history = {
                section_id: events[-HISTORY_EVENTS_PER_SECTION:]
                for section_id, events in history.items()
            }
            if on_the_bus:
                self._pump_terms_soon(queue_id)
            else:
                state.decided_gap_ids |= evaluated_gap_ids
            served, decided = self._replan_counts(
                queue_id, state, on_the_bus=on_the_bus, horizon=horizon, evaluated=evaluated_gap_ids
            )
            self._log_replan(
                queue_id,
                window_tracks=window_tracks,
                on_the_bus=on_the_bus,
                served=served,
                decided=decided,
                planned=planned,
                injected=injected,
                skipped=skipped,
            )

    def _plan_and_decide(
        self,
        *,
        queue_id: str,
        state: DJQueueState,
        program: dict[str, Any],
        window_tracks: list[dict[str, Any]],
        allowed_slot_when: list[str],
        runtime_tokens: dict[str, str],
        outro_target: dict[str, Any] | None,
        on_the_bus: bool,
    ) -> tuple[
        list[PlannedSection], dict[str, list[tuple[int, float]]], set[BoundaryKey], set[str]
    ]:
        """
        Plan one pass and settle what its boundaries mean on the path it runs.

        :param queue_id: The queue that is planned.
        :param state: The queue's DJ state.
        :param program: The station and host the sections are rendered with.
        :param window_tracks: The tracks of the planning window, in playing order.
        :param allowed_slot_when: The slot kinds this pass may plan.
        :param runtime_tokens: The tokens the sections are prompted with.
        :param outro_target: The window track the show's outro belongs after, if any.
        :param on_the_bus: Whether the pass plans terms instead of queue items.
        :return: The sections still to splice, the guard history, the boundaries the term
            path evaluated and the gaps the item path settled.
        """
        horizon: set[BoundaryKey] = set()
        evaluated_gap_ids: set[str] = set()
        # the term path books its boundaries in the term store, never in the item path's set,
        # and what the store decided is what keeps a settled boundary from being rolled again
        decided_next_item_ids = (
            self._term_decided_next_item_ids(queue_id) if on_the_bus else state.decided_gap_ids
        )
        planned, history = self._plan_sections(
            session_id=state.dj_session_id,
            tracks=window_tracks,
            program=program,
            track_index_offset=state.songs_before_window,
            minute_offset=state.minutes_before_window,
            history_state=state.history,
            allowed_slot_when=allowed_slot_when,
            runtime_tokens=runtime_tokens,
            decided_next_item_ids=decided_next_item_ids,
        )
        if on_the_bus:
            horizon = self._decide_boundaries(
                queue_id,
                program,
                planned,
                window_tracks,
                outro_target,
                history,
                index_base=state.songs_before_window,
            )
            return [], history, horizon, evaluated_gap_ids
        return planned, history, horizon, self._evaluated_gaps(window_tracks, state)

    def _decide_boundaries(
        self,
        queue_id: str,
        program: dict[str, Any],
        planned: list[PlannedSection],
        window_tracks: list[dict[str, Any]],
        outro_target: dict[str, Any] | None,
        history: dict[str, list[tuple[int, float]]],
        *,
        index_base: int = 0,
    ) -> set[BoundaryKey]:
        """
        Turn a pass's planned sections into terms on the queue's program bus.

        :param queue_id: The queue that is planned.
        :param program: The station and host the sections are rendered with.
        :param planned: The sections this pass planned.
        :param window_tracks: The tracks of the planning window, in playing order.
        :param outro_target: The window track the show's outro belongs after, if any.
        :param history: The guard history of the pass.
        :param index_base: Songs that play before the window, so a term knows where it plays.
        :return: The boundaries this pass evaluated, i.e. the horizon it decided in.
        """
        # the switch moves the show onto the queue's program bus: every boundary becomes a
        # term, never a queue item, so the user's queue stays untouched
        plans = term_boundary_plans(
            planned, window_tracks, outro_target, history, index_base=index_base
        )
        horizon = self._term_gap_horizon(
            queue_id,
            window_tracks,
            reserve=self._term_outro_reserve(queue_id, outro_target),
            index_base=index_base,
        )
        self._decide_terms(
            queue_id,
            program,
            term_plans_in_horizon(plans, horizon, history),
            history=history,
            horizon=horizon,
        )
        return horizon

    @staticmethod
    def _nothing_to_plan(window: list[QueueItem], state: DJQueueState) -> bool:
        """
        Return whether a pass has no boundary ahead of the player it could plan.

        A between-songs gap needs two window tracks, but a show's outro is still due once
        only its last song is left in the window.

        :param window: The planning window of the pass.
        :param state: The queue's DJ state.
        """
        return not window or (len(window) < 2 and not state.station_id)

    def _replan_counts(
        self,
        queue_id: str,
        state: DJQueueState,
        *,
        on_the_bus: bool,
        horizon: set[tuple[str, str | None]],
        evaluated: set[str],
    ) -> tuple[int, int]:
        """
        Return how many boundaries a pass looked at, and how many are decided afterwards.

        :param queue_id: The queue that was planned.
        :param state: The queue's DJ state.
        :param on_the_bus: Whether the pass planned terms instead of queue items.
        :param horizon: The boundaries the term path evaluated.
        :param evaluated: The gaps the item path evaluated.
        """
        if on_the_bus:
            return len(horizon), len(self._term_store(queue_id).active())
        return len(evaluated), len(state.decided_gap_ids)

    def _log_replan(
        self,
        queue_id: str,
        *,
        window_tracks: list[dict[str, Any]],
        on_the_bus: bool,
        served: int,
        decided: int,
        planned: list[PlannedSection],
        injected: int,
        skipped: dict[str, int],
    ) -> None:
        """
        Say what one planning pass looked at, and what it left the queue with.

        :param queue_id: The queue that was planned.
        :param window_tracks: The tracks of the planning window, in playing order.
        :param on_the_bus: Whether the pass planned terms instead of queue items.
        :param served: How many boundaries the pass looked at.
        :param decided: How many boundaries the path it runs holds afterwards.
        :param planned: The sections the item path has still to splice.
        :param injected: How many clips the item path spliced.
        :param skipped: Why the item path spliced no clip, per reason.
        """
        if on_the_bus:
            # a term is never a queue item, so what was spliced says nothing here: what the
            # pass looked at and what the store holds is what explains its silence
            self.logger.debug(
                "Replanned queue %s: window %s tracks, served %s boundaries, %s terms held",
                queue_id,
                len(window_tracks),
                served,
                decided,
            )
            return
        self.logger.debug(
            "Replanned queue %s: window %s tracks, %s open boundaries, planned %s, "
            "injected %s, skipped %s, decided %s",
            queue_id,
            len(window_tracks),
            served,
            len(planned),
            injected,
            skipped,
            decided,
        )

    def _prune_item_path_decisions(
        self, state: DJQueueState, items: list[QueueItem], guard_index: int
    ) -> None:
        """
        Forget the item path's settled gaps once the tracks they belong to left the queue.

        :param state: The queue's DJ state.
        :param items: All items of the queue, in playing order.
        :param guard_index: The highest queue index the player already owns.
        """
        decided_before = state.decided_gap_ids
        state.decided_gap_ids = decided_before & {item.queue_item_id for item in items}
        if decided_before and not state.decided_gap_ids:
            # nothing this history was recorded against is left, so its queue is gone
            self._drop_unaired_dj_history(state, items, guard_index)

    def _splice_dj_clip(
        self,
        queue_id: str,
        items: list[QueueItem],
        guard_index: int,
        state: DJQueueState,
        program: dict[str, Any],
        target: dict[str, Any],
        section: PlannedSection,
        after_target: bool = False,
    ) -> DJSpliceOutcome:
        """Insert one planned clip in front of (or, for an outro, after) its target track."""
        target_index = next(
            (index for index, item in enumerate(items) if item.queue_item_id == target["item_id"]),
            None,
        )
        if target_index is None:
            return "gap_gone"
        # the insertion point shifts one slot past the target for an after-target splice, so
        # its guard check is the before-target one shifted by that same one slot
        insert_index = target_index + 1 if after_target else target_index
        # the target keeps one full slot of margin from the guard, since the slot right after
        # the guard may still be handed to the player at any moment
        if insert_index <= guard_index + 1:
            return "too_close"
        # occupied checks the slot on the insertion side of the target: behind it (the item
        # that would follow the new clip) for an after-target splice, ahead of it otherwise.
        # an outro behind the queue tail has no such slot
        occupant_index = insert_index if after_target else insert_index - 1
        if occupant_index < len(items) and items[occupant_index].extra_attributes.get(
            ATTR_QUEUE_DJ
        ):
            return "occupied"
        # the planner numbers its clips from zero every pass, so the id comes from the
        # state counter instead to stay unique for the lifetime of the session
        section.clip_id = f"{state.dj_session_id}_{state.clip_counter:03d}"
        state.clip_counter += 1
        clip = self._section_to_clip_item(queue_id, state.dj_session_id, program, section)
        clip.extra_attributes[ATTR_QUEUE_DJ] = True
        if not after_target:
            clip.extra_attributes[ATTR_GAP_NEXT_ID] = target["item_id"]
        # else: an outro announces no successor track, so it carries no gap-next id;
        # _repair_dj_clips exempts clips without one from its stale-successor check
        # sharing the neighbouring item's sort index keeps the clip next to the track it
        # announces when the queue is un-shuffled, without renumbering everything behind it
        clip.sort_index = items[target_index].sort_index
        items.insert(insert_index, clip)
        return "injected"

    def _pump_terms_soon(self, queue_id: str) -> None:
        """Drive the queue's terms one step, without waiting for the pass that asked."""
        if self._unloading:
            return
        state = self._dj_queues.get(queue_id)
        if state is None:
            # a queue without a DJ holds no terms to drive
            return
        if state.pump_pending:
            # a pass is running or on its way, and it serves this request too: the queue
            # sends an event about every second, and a pass per event only queues up reads
            return
        state.pump_pending = True
        self.mass.create_task(
            self._drain_term_pumps(queue_id), task_id=f"ai_radio_dj_pump_{queue_id}"
        )

    async def _drain_term_pumps(self, queue_id: str) -> None:
        """Run term pumps until no further request landed during the last one."""
        # the inserts of a pass re-fire QUEUE_ITEMS_UPDATED while this task still holds the
        # pump task id, so its follow-up request is served here instead of by a new task
        while (state := self._dj_queues.get(queue_id)) is not None and state.pump_pending:
            # cleared up front so an event landing mid-pass requests a fresh pass
            state.pump_pending = False
            if self._unloading:
                return
            try:
                await self._pump_terms(queue_id)
            except Exception:
                # not retried from here, so a pump that keeps failing can not hot-loop; the
                # request is cleared instead of left pending, so the next event can try again
                self.logger.exception("Queue DJ term pump failed for %s", queue_id)
                if (live_state := self._dj_queues.get(queue_id)) is not None:
                    live_state.pump_pending = False
                return

    async def _render_sections_pcm(
        self,
        queue_id: str,
        program: dict[str, Any],
        sections: list[PlannedSection],
        pcm_format: AudioFormat,
    ) -> bytes:
        """
        Render the given sections, one after another, into the given PCM format.

        :param queue_id: The queue whose stream the sections are mixed into.
        :param program: The station and host the sections belong to.
        :param sections: The sections to speak, in order.
        :param pcm_format: The format the queue's stream produces.
        """
        pcm = bytearray()
        for section in sections:
            pcm.extend(await self._render_section_pcm(queue_id, program, section, pcm_format))
        return bytes(pcm)

    async def _render_mixed_program(
        self,
        queue_id: str,
        program: dict[str, Any],
        sections: list[PlannedSection],
        pcm_format: AudioFormat,
    ) -> bytes:
        """
        Render a program for the queue's stream, counting it against the DJ's render health.

        A program mixed over the stream is paid for the same way a clip is, so an engine
        that keeps failing pauses the queue's programs instead of being paid again for
        every gap and every intro.

        :param queue_id: The queue the program is mixed into.
        :param program: The station and host the sections belong to.
        :param sections: The sections to speak, in order.
        :param pcm_format: The format the queue's stream produces.
        :return: The rendered program, or empty while the DJ's programs are paused.
        """
        state = self._dj_queues.get(queue_id)
        if state is not None and state.render_health.cooling_down(
            asyncio.get_running_loop().time()
        ):
            self.logger.debug("Queue %s mixes no program in while its breaks are paused", queue_id)
            return b""
        try:
            pcm = await self._render_sections_pcm(queue_id, program, sections, pcm_format)
        except EngineRenderError as err:
            now = asyncio.get_running_loop().time()
            if state is not None and state.render_health.record_failure(err.part, now):
                # no queue event of its own would ask the DJ to render again once the pause
                # ends, so the pass that does is booked here for every program mixed in
                health = state.render_health
                self.logger.warning(engine_pause_message(queue_id, health, err.part, err.engine))
                self._book_replan_after_cooldown(queue_id, health.resume_in(now))
            raise
        if state is not None:
            state.render_health.record_render_success(asyncio.get_running_loop().time())
        return pcm

    def _overlay_window_tracks(self, queue_id: str) -> list[dict[str, Any]]:
        """
        Return the upcoming tracks a mixed-in program can name, in playing order.

        :param queue_id: The queue to look ahead in.
        """
        queue = self.mass.player_queues.get(queue_id)
        guard_index = self._dj_guard_index(queue) if queue is not None else -1
        window = self._dj_window(self._dj_queue_items(queue_id), guard_index)
        return [self._queue_item_to_track(index, item) for index, item in enumerate(window)]

    @staticmethod
    def _queue_plays(queue: PlayerQueue) -> bool:
        """
        Return whether the queue is playing, so a program mixed in would be heard.

        :param queue: The queue to check.
        """
        return not queue.ended and queue.state == PlaybackState.PLAYING

    def _remove_pending_dj_clips(
        self, queue_id: str, *, keep_live_session: bool = True
    ) -> list[QueueItem]:
        """Remove not-yet-played DJ clips from the queue, except the armed session's own."""
        # await-free on purpose: nothing can mutate the queue between the snapshot below and
        # the update that applies the filtered list
        queue = self.mass.player_queues.get(queue_id)
        if queue is None:
            return []
        # only the live session's clips survive, so a re-enable racing this cleanup keeps
        # its own; clips from a session nothing remembers (ids re-roll each load) are cleared
        live_state = self._dj_queues.get(queue_id)
        keep_session_id = (
            live_state.dj_session_id if keep_live_session and live_state is not None else None
        )
        guard_index = self._dj_guard_index(queue)
        items = self._dj_queue_items(queue_id)
        # one update for the whole cleanup: a delete per clip floods every connected client
        # with queue events. items up to the guard are what the player already owns
        kept: list[QueueItem] = []
        removed: list[QueueItem] = []
        for index, item in enumerate(items):
            if (
                index <= guard_index
                or not item.extra_attributes.get(ATTR_QUEUE_DJ)
                or (
                    keep_session_id is not None
                    and item.extra_attributes.get(ATTR_SESSION_ID) == keep_session_id
                )
            ):
                kept.append(item)
                continue
            removed.append(item)
        if not removed:
            return []
        self.mass.player_queues.update_items(queue_id, kept)
        self.logger.debug("Removed %s pending DJ clip(s) from queue %s", len(removed), queue_id)
        return removed

    async def _cleanup_legacy_dj_items(self) -> None:
        """Clear the DJ clips an older release spliced into the queues, once."""
        await remove_legacy_dj_items(
            self.mass, self.logger, self.instance_id, self._live_dj_session_ids()
        )

    def _sync_item_path(self, queue_id: str) -> None:
        """Drop the item path's clips once the program bus has taken the queue over."""
        state = self._dj_queues.get(queue_id)
        if state is None:
            return
        bus_enabled = self.mass.streams.program_bus_enabled()
        if bus_enabled and not state.bus_took_over:
            # the switch came on since the last pass: a clip item and a term at the same
            # boundary would both speak, so the clips go (what the player already owns stays)
            removed = self._remove_pending_dj_clips(queue_id, keep_live_session=False)
            # every gap the item path settled belongs to a clip, and every guard event to a
            # break it spliced; a term has to be free to decide both for itself
            self._forget_item_path_decisions(state)
            if removed:
                self.logger.debug(
                    "Queue %s switched to terms: removed %d pending item clip(s)",
                    queue_id,
                    len(removed),
                )
        state.bus_took_over = bus_enabled

    def _forget_item_path_decisions(self, state: DJQueueState) -> None:
        """Forget what the item path settled about a queue's gaps and its breaks."""
        # the set holds successor item ids of clips that will never air, and the history
        # counts breaks the listener never heard; both would hold the term path back
        state.decided_gap_ids.clear()
        state.history.clear()

    def _live_dj_session_ids(self) -> dict[str, str]:
        """Return the session id of every armed queue, so a cleanup can spare its clips."""
        return {queue_id: state.dj_session_id for queue_id, state in self._dj_queues.items()}

    def _pause_queue_dj(self, queue_id: str, failed_item_id: str, cooldown_seconds: float) -> None:
        """
        Take the queue's pending DJ clips out and plan again once the engine's pause ends.

        :param queue_id: The queue whose DJ pauses.
        :param failed_item_id: The queue item id of the clip whose render failed last.
        :param cooldown_seconds: How long the engine that failed stays paused.
        """
        if (state := self._dj_queues.get(queue_id)) is None:
            return
        if (queue := self.mass.player_queues.get(queue_id)) is not None:
            guard_index = self._dj_guard_index(queue)
            items = self._dj_queue_items(queue_id)
            # the failed clip may be the one the queue is loading while this runs, and the slot
            # right after the guard may be handed to the player at any moment: both stay (the
            # clip marked unplayable), since taking either out would shift the indexes the
            # queue's load walks. everything beyond them goes
            failed_index = next(
                (index for index, item in enumerate(items) if item.queue_item_id == failed_item_id),
                guard_index + 1,
            )
            removed = [
                item
                for item in items[max(guard_index + 1, failed_index) + 1 :]
                if item.extra_attributes.get(ATTR_QUEUE_DJ)
            ]
            removed_ids = {item.queue_item_id for item in removed}
            kept = [item for item in items if item.queue_item_id not in removed_ids]
            if removed:
                self.mass.player_queues.update_items(queue_id, kept)
                # the gaps of the breaks taken out are open again
                for item in removed:
                    if (gap_next_id := item.extra_attributes.get(ATTR_GAP_NEXT_ID)) is not None:
                        state.decided_gap_ids.discard(str(gap_next_id))
                self.logger.debug(
                    "Removed %s pending DJ clip(s) from paused queue %s", len(removed), queue_id
                )
            # neither the failed break nor the ones taken out ever aired, so their guard history
            # must not hold the planner back once it resumes. the window start it measures from
            # is the last full pass's, so a break that aired since may be forgotten with them,
            # which costs at most one early break after the pause
            self._drop_unaired_dj_history(state, kept, guard_index)
        self._book_replan_after_cooldown(queue_id, cooldown_seconds)

    def _book_replan_after_cooldown(self, queue_id: str, cooldown_seconds: float) -> None:
        """
        Book the pass that plans again once the engine's render cool-down has ended.

        :param queue_id: The queue whose DJ paused.
        :param cooldown_seconds: How long the paused engine stays paused.
        """
        # a queue that only plays music raises no items event, so the end of the cool-down has
        # to request the replan itself
        self.mass.call_later(
            cooldown_seconds,
            self._schedule_replan,
            queue_id,
            task_id=f"ai_radio_dj_cooldown_{queue_id}",
        )

    def _dj_guard_index(self, queue: PlayerQueue) -> int:
        """Return the highest queue index the player already owns."""
        # the player owns the current and the already buffered item, and the slot right
        # after the buffered one may be handed to the player at any moment
        boundary_index = committed_index(queue)
        return boundary_index if boundary_index is not None else -1

    def _dj_term_window_guard(
        self, queue_id: str, queue: PlayerQueue, items: list[QueueItem], guard_index: int
    ) -> int:
        """
        Return the queue index a planning pass starts its window at.

        The player owning the running item keeps the item path from splicing ahead of it, but
        a term writes nothing into the queue: the boundary the running item ends at is spoken
        over the stream. It is planned from the running item on, unless a term already speaks
        it.

        :param queue_id: The queue that is planned.
        :param queue: The queue that is playing.
        :param items: All items of the queue, in playing order.
        :param guard_index: The highest queue index the player already owns.
        """
        if not self.mass.streams.program_bus_enabled():
            return guard_index
        current = queue.current_item
        boundary = running_boundary(items, None if current is None else current.queue_item_id)
        if boundary is None or boundary.index > guard_index:
            # the running item is not in the queue, or the window already starts at it
            return guard_index
        if self._term_holds_boundary(queue_id, boundary):
            return guard_index
        return boundary.index - 1

    def _repair_dj_clips(
        self, queue_id: str, state: DJQueueState, items: list[QueueItem], guard_index: int
    ) -> bool:
        """Delete DJ clips that no longer sit in front of the track they announce."""
        stale_ids: set[str] = set()
        for index in range(guard_index + 2, len(items)):
            item = items[index]
            if not item.extra_attributes.get(ATTR_QUEUE_DJ):
                continue
            gap_next_id = item.extra_attributes.get(ATTR_GAP_NEXT_ID)
            if gap_next_id is None:
                # an outro announces no successor track, so there is nothing to repair against
                continue
            successor = items[index + 1] if index + 1 < len(items) else None
            if successor is not None and successor.queue_item_id == gap_next_id:
                preceding = items[index - 1] if index > 0 else None
                if _can_play(preceding) and _can_play(successor):
                    continue
                # a boundary that cannot play is nothing to speak over, and its gap stays
                # decided: re-deciding it would render one break per skipped track
                stale_ids.add(item.queue_item_id)
                state.decided_gap_ids.add(str(gap_next_id))
                continue
            stale_ids.add(item.queue_item_id)
            # the gap this clip was serving is open again, so let a later pass decide it anew
            state.decided_gap_ids.discard(str(gap_next_id))
        if not stale_ids:
            return False
        # one update for all of them, so the clients see a single queue change
        self.mass.player_queues.update_items(
            queue_id, [item for item in items if item.queue_item_id not in stale_ids]
        )
        self.logger.debug(
            "Repaired queue %s: deleted %s stale DJ clip(s)", queue_id, len(stale_ids)
        )
        return True

    def _show_outro_target(
        self,
        items: list[QueueItem],
        window: list[QueueItem],
        window_tracks: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Return the window track the show's outro belongs after, once it has landed."""
        # one outro per play: a clip without a gap target is the outro, wherever it sits
        # (songs appended behind an already-spliced outro must not earn a second one)
        for item in items:
            if item.extra_attributes.get(ATTR_QUEUE_DJ) and not item.extra_attributes.get(
                ATTR_GAP_NEXT_ID
            ):
                return None
        # the whole show is enqueued up front, so its last song is simply the queue's last
        # non-DJ item (searched from the tail, in case something got appended behind it)
        last_song_id = next(
            (
                item.queue_item_id
                for item in reversed(items)
                if not item.extra_attributes.get(ATTR_QUEUE_DJ)
            ),
            None,
        )
        if last_song_id is None:
            return None
        for index, window_item in enumerate(window):
            if window_item.queue_item_id == last_song_id:
                return window_tracks[index]
        return None

    def _evaluated_gaps(self, window_tracks: list[dict[str, Any]], state: DJQueueState) -> set[str]:
        """
        Return the gaps an item-path pass settles.

        :param window_tracks: The tracks of the planning window, in playing order.
        :param state: The queue's DJ state.
        """
        # every gap the planner is about to evaluate, so gaps where chance or a guard
        # picks nothing count as decided too instead of being rolled again next pass
        return {
            str(track["item_id"])
            for track in window_tracks[1:]
            if track["item_id"] not in state.decided_gap_ids
        }

    def _dj_window(self, items: list[QueueItem], guard_index: int) -> list[QueueItem]:
        """Return the upcoming music items that this pass may plan against."""
        # every upcoming track, decided or not: the planner counts songs and minutes over a
        # contiguous run, and per gap decisions are what keeps the work from being redone
        return [
            item
            for item in items[guard_index + 1 :]
            if not item.extra_attributes.get(ATTR_QUEUE_DJ)
        ]

    def _dj_window_offsets(self, items: list[QueueItem], window_start_id: str) -> tuple[int, float]:
        """Return the songs and minutes of music playing before the first window track."""
        behind = []
        for item in items:
            if item.queue_item_id == window_start_id:
                break
            if not item.extra_attributes.get(ATTR_QUEUE_DJ):
                behind.append(item)
        minutes = sum(item.duration or FALLBACK_TRACK_SECONDS for item in behind) / 60.0
        return len(behind), minutes

    def _drop_unaired_dj_history(
        self, state: DJQueueState, items: list[QueueItem], guard_index: int
    ) -> None:
        """Forget the guard history of breaks that were planned but never reached the player."""
        # events strictly behind the window start have aired; one exactly on it is ambiguous,
        # since its clip may be owned by the player (airing or buffered) or still one slot
        # beyond the guard. a clip surviving in the owned head is what tells the two apart
        owns_clip = any(
            item.extra_attributes.get(ATTR_QUEUE_DJ) for item in items[: guard_index + 1]
        )
        boundary = state.songs_before_window if owns_clip else state.songs_before_window - 1
        state.history = {
            section_id: [(song, minute) for song, minute in events if song <= boundary]
            for section_id, events in state.history.items()
        }

    def _rebase_dj_history(
        self, state: DJQueueState, songs_before_window: int, minutes_before_window: float
    ) -> None:
        """Re-anchor the guard history onto the given start of the planning window."""
        # events behind the window move with it so they keep their distance, while the ones
        # ahead are capped at the window start: their old position no longer means anything
        song_delta = min(songs_before_window - state.songs_before_window, 0)
        minute_delta = min(minutes_before_window - state.minutes_before_window, 0.0)
        state.history = {
            section_id: [
                (
                    min(song + song_delta, songs_before_window),
                    min(minute + minute_delta, minutes_before_window),
                )
                for song, minute in events
            ]
            for section_id, events in state.history.items()
        }

    def _dj_queue_items(self, queue_id: str) -> list[QueueItem]:
        """Return all items of a queue."""
        items: list[QueueItem] = []
        offset = 0
        while True:
            page = self.mass.player_queues.items(queue_id, limit=QUEUE_PAGE_SIZE, offset=offset)
            items.extend(page)
            if len(page) < QUEUE_PAGE_SIZE:
                return items
            offset += QUEUE_PAGE_SIZE

    def _queue_item_to_track(self, index: int, item: QueueItem) -> dict[str, Any]:
        """Convert a music queue item into the track dict the planner consumes."""
        name = ""
        artist = ""
        if (media_item := item.media_item) is not None:
            name = str(media_item.name or "")
            artists = getattr(media_item, "artists", None)
            if artists:
                artist = str(artists[0].name)
        if not name:
            raw_name = str(item.name or "")
            artist, separator, name = raw_name.partition(" - ")
            if not separator:
                artist, name = "", raw_name
        return {
            "index": index,
            "item_id": item.queue_item_id,
            "name": name,
            "artist": artist,
            "songinfo": f"{artist} - {name}".strip(" -"),
            "duration": item.duration,
            "available": bool(item.available),
            "media_item": None,
        }

    def _ensure_stream_claims(self, queue_id: str) -> None:
        """
        Place a queue's stream claims again, for a stream this event may have opened.

        A restart can read the program bus switch before the core config is ready, so the
        restore finds it off and claims the item path's boundaries instead. The queue's next
        event re-evaluates the claim, and the bus is claimed then. The call is idempotent,
        so a queue that already holds its claim sees no change.

        :param queue_id: The queue whose claims are checked.
        """
        state = self._dj_queues.get(queue_id)
        if state is None:
            return
        if queue_id in getattr(self, "_flow_claimed", set()):
            return
        self._apply_stream_claims(queue_id, state.host_id)

    def _refresh_stream_claims(self) -> None:
        """Re-evaluate which armed queues need their item boundaries mixed or held for a lead."""
        for queue_id, state in list(self._dj_queues.items()):
            self._apply_stream_claims(queue_id, state.host_id)

    def _apply_stream_claims(self, queue_id: str, host_id: str) -> None:
        """Record what the host's sections need from the queue's stream."""
        flow_claimed: set[str] = getattr(self, "_flow_claimed", set())
        if not hasattr(self, "_flow_claimed"):
            self._flow_claimed = flow_claimed
        if self.mass.streams.program_bus_enabled():
            # the bus exists only in a flow stream and carries the breaks itself, so the
            # flow claim replaces the item claims the overlay path needed
            self.mass.streams.set_flow_mode_required(queue_id, self.instance_id, True)
            flow_claimed.add(queue_id)
            self.mass.streams.set_boundary_mix_required(queue_id, self.instance_id, False)
            self.mass.streams.set_lead_hold_required(queue_id, self.instance_id, False)
            return
        if queue_id in flow_claimed:
            # only release a claim this owner really holds, so the flag-off path places
            # exactly the claims it placed before the switch existed
            self.mass.streams.set_flow_mode_required(queue_id, self.instance_id, False)
            flow_claimed.discard(queue_id)
        self.mass.streams.set_boundary_mix_required(
            queue_id, self.instance_id, self._host_mixes_boundary(host_id)
        )
        self.mass.streams.set_lead_hold_required(
            queue_id, self.instance_id, self._host_carries_lead(host_id)
        )

    def _release_stream_claims(self, queue_id: str) -> None:
        """Let go of every stream claim the DJ holds on a queue."""
        flow_claimed: set[str] = getattr(self, "_flow_claimed", set())
        if queue_id in flow_claimed:
            self.mass.streams.set_flow_mode_required(queue_id, self.instance_id, False)
            flow_claimed.discard(queue_id)
        self.mass.streams.set_boundary_mix_required(queue_id, self.instance_id, False)
        self.mass.streams.set_lead_hold_required(queue_id, self.instance_id, False)

    def _host_mixes_boundary(self, host_id: str) -> bool:
        """Return whether any section of the host lets its break straddle a track boundary."""
        return any(
            bool(section.get("allow_post", False)) or bool(section.get("allow_pre", False))
            for section in self._host_sections(host_id)
        )

    def _host_carries_lead(self, host_id: str) -> bool:
        """Return whether any section of the host lets a break start over the outgoing tail."""
        return any(
            bool(section.get("allow_pre", False)) for section in self._host_sections(host_id)
        )

    def _host_sections(self, host_id: str) -> list[dict[str, Any]]:
        """Return the sections the host may plan, as stored."""
        host = self._hosts.get(host_id) or {}
        return [
            self._sections.get(str(section_id), {}) for section_id in host.get("section_ids", [])
        ]

    def _queue_show_station(self, queue_id: str) -> str | None:
        """Return the station id of the show in the queue's sources, if any."""
        queue = self.mass.player_queues.get(queue_id)
        if queue is None:
            return None
        for source in queue.sources:
            if (station_id := self._station_id_from_source_uri(source.uri)) is not None:
                return station_id
        return None

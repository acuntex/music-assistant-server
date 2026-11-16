"""
The intro a queue DJ speaks when it is switched on.

An intro is not planned for a boundary: it airs right where the listener is, once per
activation, and waits for a stream that can mix it in. A queue that is still loading or
paused owes the intro rather than losing it, and a DJ restored after a restart owes none.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from music_assistant.controllers.streams.program_schedule import ProgramKind

from .constants import (
    CONF_PLAY_INTRO_ON_ACTIVATE,
    DEFAULT_PLAY_INTRO_ON_ACTIVATE,
    INTRO_PENDING_LOOKS,
    INTRO_RENDER_TIMEOUT_SECONDS,
    INTRO_RETRY_SECONDS,
    OVERLAY_LABEL_INTRO,
)
from .hosts import host_plays_intro
from .models import DJQueueState, PlannedSection

if TYPE_CHECKING:
    import logging

    from music_assistant_models.media_items import AudioFormat
    from music_assistant_models.player_queue import PlayerQueue

    from music_assistant.mass import MusicAssistant


class AIRadioIntroMixin:
    """Mixin speaking the intro of a queue DJ, once per activation."""

    if TYPE_CHECKING:
        mass: MusicAssistant
        logger: logging.Logger
        _hosts: dict[str, dict[str, Any]]
        _dj_queues: dict[str, DJQueueState]

        def get_config_value(
            self, key: str, default: Any = None, *, return_type: Any = None
        ) -> Any:
            """Return the provider's configured value for the given key."""

        def _build_program(self, station: dict[str, Any], host: dict[str, Any]) -> dict[str, Any]:
            """Merge a station and its host into the program the planner consumes."""

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
        ) -> tuple[list[PlannedSection], dict[str, list[tuple[int, float]]]]:
            """Return the sections the planner decided on."""

        async def _prepare_runtime_tokens(self, program: dict[str, Any]) -> dict[str, str]:
            """Return the deferred placeholder values of this render."""

        async def _render_mixed_program(
            self,
            queue_id: str,
            program: dict[str, Any],
            sections: list[PlannedSection],
            pcm_format: AudioFormat,
        ) -> bytes:
            """Render a program for the queue's stream."""

        def _overlay_window_tracks(self, queue_id: str) -> list[dict[str, Any]]:
            """Return the upcoming tracks a program can name."""

        def _schedule_replan(self, queue_id: str) -> None:
            """Request a replan pass for the given queue."""

        @staticmethod
        def _queue_plays(queue: PlayerQueue) -> bool:
            """Return whether the queue is playing, so a program mixed in would be heard."""

    def _start_intro_overlay(self, queue_id: str, state: DJQueueState) -> None:
        """Speak the intro the host opens with, over the queue's stream, once per activation."""
        if state.intro_placed or state.intro_until is not None:
            return
        if state.intro_pending and not self._intro_belongs_to_playback(queue_id, state):
            # the content the intro was written for is gone: a skip, or another album loaded
            # before the playback started. it is not re-written for what plays now
            self.logger.debug(
                "Queue %s left the playback its intro was owed for - the intro is dropped",
                queue_id,
            )
            state.intro_pending = False
            state.intro_item_id = None
            return
        if (reason := self._intro_skipped_reason(queue_id, state)) is not None:
            self.logger.debug("Queue %s speaks no intro: %s", queue_id, reason)
            # a show arms its DJ while its queue is still loading and before the flow
            # stream exists, so an intro blocked by that is owed, not dropped
            state.intro_pending = self._intro_owes_a_retry(state)
            if state.intro_item_id is None:
                state.intro_item_id = self._current_queue_item_id(queue_id)
            queue = self.mass.player_queues.get(queue_id)
            if state.intro_pending and queue is not None and self._queue_plays(queue):
                # the stream opens seconds after playback starts and does not always send
                # a queue event of its own, so the intro is looked for again a few times
                self._book_intro_retry(queue_id, state, INTRO_PENDING_LOOKS)
            return
        state.intro_pending = False
        if state.intro_item_id is None:
            state.intro_item_id = self._current_queue_item_id(queue_id)
        # breaks are held back until the intro is placed, or the two would talk over each other
        state.intro_until = asyncio.get_running_loop().time() + INTRO_RENDER_TIMEOUT_SECONDS
        self.mass.call_later(
            INTRO_RENDER_TIMEOUT_SECONDS,
            self._release_intro_hold,
            queue_id,
            state,
            task_id=f"ai_radio_intro_hold_{queue_id}",
        )
        self.mass.create_task(
            self._place_intro_overlay(queue_id, state), task_id=f"ai_radio_intro_{queue_id}"
        )

    def _intro_owes_a_retry(self, state: DJQueueState) -> bool:
        """Return whether the intro is allowed and only waits for playback that can air it."""
        if not self.mass.streams.program_bus_enabled():
            # no look can find a stream that mixes programs while the switch is off
            return False
        return (
            bool(
                self.get_config_value(
                    CONF_PLAY_INTRO_ON_ACTIVATE,
                    DEFAULT_PLAY_INTRO_ON_ACTIVATE,
                    return_type=bool,
                )
            )
            and host_plays_intro(self._hosts.get(state.host_id))
            and self._host_has_intro(state.host_id)
        )

    def _intro_belongs_to_playback(self, queue_id: str, state: DJQueueState) -> bool:
        """
        Return whether the queue still holds the item the intro was owed for.

        :param queue_id: The queue the intro would be spoken over.
        :param state: The DJ state the intro belongs to.
        """
        if state.intro_item_id is None:
            # the queue had no item to bind to when the intro came due, so nothing moved on
            return True
        live_item_id = self._current_queue_item_id(queue_id)
        # a queue that reports no playhead has not moved on to another item, it is on its way
        # to the one the intro was owed for
        return live_item_id is None or live_item_id == state.intro_item_id

    def _intro_playback_is_live(self, queue_id: str, state: DJQueueState) -> bool:
        """
        Return whether the playback the intro was rendered for still runs.

        :param queue_id: The queue the intro would be spoken over.
        :param state: The DJ state the intro belongs to.
        """
        queue = self.mass.player_queues.get(queue_id)
        if queue is None or not self._queue_plays(queue):
            return False
        live_item_id = queue.current_item.queue_item_id if queue.current_item else None
        return live_item_id == state.intro_item_id

    def _current_queue_item_id(self, queue_id: str) -> str | None:
        """
        Return the queue item the queue is playing, if any.

        :param queue_id: The queue to read the playhead of.
        """
        queue = self.mass.player_queues.get(queue_id)
        if queue is None or queue.current_item is None:
            return None
        return queue.current_item.queue_item_id

    def _book_intro_retry(self, queue_id: str, state: DJQueueState, looks: int) -> None:
        """Look again for the stream an owed intro needs, a moment from now."""
        if looks <= 0:
            return
        self.mass.call_later(
            INTRO_RETRY_SECONDS,
            self._retry_intro,
            queue_id,
            state,
            looks,
            task_id=f"ai_radio_intro_retry_{queue_id}",
        )

    def _retry_intro(self, queue_id: str, state: DJQueueState, looks: int) -> None:
        """Speak an owed intro once its queue plays and its stream carries programs."""
        if self._dj_queues.get(queue_id) is not state or not state.intro_pending:
            return
        if (reason := self._intro_skipped_reason(queue_id, state)) is not None:
            self.logger.debug("Queue %s still owes its intro: %s", queue_id, reason)
            if (queue := self.mass.player_queues.get(queue_id)) is not None and self._queue_plays(
                queue
            ):
                self._book_intro_retry(queue_id, state, looks - 1)
            return
        self._start_intro_overlay(queue_id, state)

    def _intro_holds_breaks(self, queue_id: str, state: DJQueueState) -> bool:
        """Return whether the queue's intro is still rendering, so no break may be planned."""
        if state.intro_until is None:
            return False
        if asyncio.get_running_loop().time() < state.intro_until:
            # the intro is still rendering and a break would talk over it
            self.logger.debug("Queue %s holds its breaks until the intro is placed", queue_id)
            return True
        # the deadline timer has not run yet, and this pass can let go of the hold itself
        self._release_intro_hold(queue_id, state, replan=False)
        return False

    def _release_intro_hold(
        self, queue_id: str, state: DJQueueState, *, replan: bool = True
    ) -> None:
        """Let the breaks plan again at the intro's deadline, whatever the render is doing."""
        if self._dj_queues.get(queue_id) is not state or state.intro_until is None:
            return
        state.intro_until = None
        # a hanging render must not hold the breaks back for the rest of the session
        self.logger.warning(
            "Queue %s intro did not render within %ss - planning its breaks anyway",
            queue_id,
            INTRO_RENDER_TIMEOUT_SECONDS,
        )
        if replan:
            self._schedule_replan(queue_id)

    def _intro_skipped_reason(self, queue_id: str, state: DJQueueState) -> str | None:
        """
        Return why the queue is not spoken an intro, or None when it should be.

        :param queue_id: The queue whose DJ was switched on.
        :param state: The DJ state the switch armed.
        """
        if not self.get_config_value(
            CONF_PLAY_INTRO_ON_ACTIVATE,
            DEFAULT_PLAY_INTRO_ON_ACTIVATE,
            return_type=bool,
        ):
            return "the intro is switched off"
        if not host_plays_intro(self._hosts.get(state.host_id)):
            return "the host switches its intro off"
        if not self._host_has_intro(state.host_id):
            return "the host has no start of playlist section"
        queue = self.mass.player_queues.get(queue_id)
        if queue is None or not self._queue_plays(queue):
            return "the queue is not playing"
        if not self.mass.streams.program_bus_enabled():
            return "the streams program bus is switched off"
        if not self.mass.streams.program_bus_active(queue_id):
            return "the queue's flow stream is not mixing programs"
        return None

    def _host_has_intro(self, host_id: str) -> bool:
        """Return whether the host plans a section for the start of the playlist."""
        host = self._hosts.get(host_id) or {}
        return any(
            str(rule.get("when", "")).strip() == "start_of_playlist"
            for rule in host.get("section_order", [])
        )

    async def _place_intro_overlay(self, queue_id: str, state: DJQueueState) -> None:
        """Mix the rendered intro into the queue's stream, then let the breaks plan again."""
        try:
            pcm = await self._render_intro_pcm(queue_id, state)
            if not pcm:
                return
            if self._dj_queues.get(queue_id) is not state:
                # the switch that armed this state was replaced while the intro rendered
                return
            if not self._intro_playback_is_live(queue_id, state):
                # the render takes seconds, so the listener may have skipped or the playback
                # may have ended meanwhile; the intro is not spoken over what plays instead
                self.logger.debug(
                    "Queue %s left the playback its intro rendered for - the intro is dropped",
                    queue_id,
                )
                return
            result = self.mass.streams.schedule_overlay(
                queue_id, pcm, label=OVERLAY_LABEL_INTRO, kind=ProgramKind.INTRO
            )
            if not result.scheduled:
                self.logger.info("Queue %s did not take its intro (%s)", queue_id, result.reason)
                return
            state.intro_placed = True
            self.logger.info("Queue %s spoke the intro of host %s", queue_id, state.host_id)
        except Exception:
            # an engine that fails the intro must not take the DJ's breaks down with it
            self.logger.exception("Queue %s could not speak its intro", queue_id)
        finally:
            if self._dj_queues.get(queue_id) is state:
                state.intro_until = None
                self._schedule_replan(queue_id)

    async def _render_intro_pcm(self, queue_id: str, state: DJQueueState) -> bytes:
        """
        Render the host's intro sections into the format the queue's stream produces.

        :param queue_id: The queue that would hear the intro.
        :param state: The DJ state the intro belongs to.
        :return: The intro's audio, or empty when there is nothing to speak into.
        """
        pcm_format = self.mass.streams.program_bus_format(queue_id)
        if pcm_format is None:
            return b""
        host = self._hosts.get(state.host_id) or {}
        program = self._build_program(
            {"id": state.station_id, "name": f"AI DJ {host['name']}"}, host
        )
        sections = await self._plan_intro_sections(state, program)
        if not sections:
            return b""
        return await self._render_mixed_program(queue_id, program, sections, pcm_format)

    async def _plan_intro_sections(
        self, state: DJQueueState, program: dict[str, Any]
    ) -> list[PlannedSection]:
        """
        Return the sections the host speaks as its intro, in order.

        :param state: The DJ state the intro belongs to.
        :param program: The station and host the sections belong to.
        """
        runtime_tokens = await self._prepare_runtime_tokens(program)
        planned, _history = self._plan_sections(
            session_id=state.dj_session_id,
            tracks=self._overlay_window_tracks(state.queue_id),
            program=program,
            track_index_offset=0,
            minute_offset=0.0,
            history_state={},
            allowed_slot_when=["start_of_playlist"],
            runtime_tokens=runtime_tokens,
        )
        return planned

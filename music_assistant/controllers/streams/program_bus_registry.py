"""Registration of the program bus of every queue's live flow session."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from music_assistant.controllers.streams.program_ledger import OverlayProgramState, ProgramLedger
from music_assistant.controllers.streams.program_levels import ProgramLevels
from music_assistant.controllers.streams.program_mixer import FlowProgramBus
from music_assistant.controllers.streams.program_schedule import ScheduleReason

if TYPE_CHECKING:
    from music_assistant_models.media_items import AudioFormat


class ProgramBusChange(StrEnum):
    """How a queue's program bus changed, for listeners that anchor programs on it."""

    REGISTERED = "registered"
    REPLACED = "replaced"
    RELEASED = "released"


@dataclass(frozen=True)
class ProgramBusEvent:
    """A queue's program bus appeared, was replaced by another session's bus, or ended."""

    queue_id: str
    change: ProgramBusChange


class ProgramBusRegistry:
    """
    The program bus of every queue's live flow session.

    A session that takes over from another one replaces its bus. What the old bus still had
    to play is not dropped blindly: a term belongs to the session that took it, while an
    announcement whose audio is already rendered may still be anchored on the new bus.
    The level a queue mixes its programs at belongs to the queue, not to one session, so it
    survives a flow restart and is handed to every bus that opens for it.
    """

    def __init__(
        self,
        ledger: ProgramLedger,
        heard_frame: Callable[[str, FlowProgramBus], int | None],
        logger: logging.Logger,
    ) -> None:
        """
        Initialize the registry.

        :param ledger: The ledger whose programs end with their bus.
        :param heard_frame: Reader for the frame a queue's listener has heard on a bus.
        :param logger: Logger for the bus changes and listener failures.
        """
        self._ledger = ledger
        self._heard_frame = heard_frame
        self._logger = logger
        self._buses: dict[str, FlowProgramBus] = {}
        self._levels: dict[str, ProgramLevels] = {}
        self._listeners: list[Callable[[ProgramBusEvent], None]] = []

    def get(self, queue_id: str) -> FlowProgramBus | None:
        """
        Return the queue's live program bus, if it has one.

        :param queue_id: Queue to look up.
        """
        return self._buses.get(queue_id)

    def active(self, queue_id: str) -> bool:
        """
        Return whether the queue has a flow session that can mix a program right now.

        :param queue_id: Queue to check.
        """
        return queue_id in self._buses

    def pcm_format(self, queue_id: str) -> AudioFormat | None:
        """
        Return the PCM format a program for the queue has to be delivered in.

        :param queue_id: Queue to check.
        :return: The PCM format of the queue's live flow session, or None without one.
        """
        if (program_bus := self._buses.get(queue_id)) is None:
            return None
        return program_bus.pcm_format

    def levels(self, queue_id: str) -> ProgramLevels:
        """
        Return the level the queue mixes its programs at.

        :param queue_id: Queue to look up.
        """
        return self._levels.get(queue_id, ProgramLevels())

    def set_levels(self, queue_id: str, levels: ProgramLevels) -> None:
        """
        Set the level the queue mixes its programs at, on its bus if one is running.

        :param queue_id: Queue whose programs are mixed at the given level.
        :param levels: Voice level and ducking depth of the queue's programs.
        """
        self._levels[queue_id] = levels
        if (program_bus := self._buses.get(queue_id)) is not None:
            self._apply_levels(queue_id, program_bus)

    def register(self, queue_id: str, program_bus: FlowProgramBus) -> None:
        """
        Publish the program bus of a flow session that just started for the queue.

        :param queue_id: Queue the session streams.
        :param program_bus: The session's program bus.
        """
        replaced = self._buses.get(queue_id)
        if replaced is not None and replaced is not program_bus:
            # a program the listener already heard must not be anchored again, so bring the
            # deadlines up to date before the old bus gives its programs away
            self._ledger.refresh(queue_id, lambda bus: self._heard_frame(queue_id, bus))
            # the old bus is still the published one here, so what the ledger drops is
            # cancelled on it; only then does the new session take over
            self._ledger.close_bus(
                queue_id,
                replaced,
                terminal_state=OverlayProgramState.DROPPED,
                terminal_reason=ScheduleReason.BUS_REPLACED,
                keep_announcements=True,
                heard_frame=lambda bus: self._heard_frame(queue_id, bus),
            )
            self._apply_levels(queue_id, program_bus)
            self._buses[queue_id] = program_bus
            self._notify(queue_id, ProgramBusChange.REPLACED)
            return
        self._apply_levels(queue_id, program_bus)
        self._buses[queue_id] = program_bus
        self._notify(queue_id, ProgramBusChange.REGISTERED)

    def forget_queue(self, queue_id: str) -> None:
        """
        Forget everything the registry kept for a queue that is gone.

        A queue that comes back under the same id has to start empty: a bus the removed
        session left behind would read as one that is already mixing programs, and the level
        the removed queue set would be inherited by the session that follows it.

        :param queue_id: Queue that was removed.
        """
        self._levels.pop(queue_id, None)
        if (program_bus := self._buses.get(queue_id)) is not None:
            # end what the bus still had while it is still published, so the ledger drops
            # its programs on it instead of leaving them scheduled for a bus that is gone
            self.release(queue_id, program_bus)

    def release(self, queue_id: str, program_bus: FlowProgramBus) -> None:
        """
        Withdraw the program bus of a flow session of the queue that ended.

        A session that was superseded by a newer one must not take that session's bus
        with it, so only the bus that is still published is released.

        :param queue_id: Queue the session streamed.
        :param program_bus: The session's program bus.
        """
        if self._buses.get(queue_id) is not program_bus:
            return
        # a session that ends can not air what it still had; close it while it is still the
        # published bus, so what the ledger drops is cancelled on it
        self._ledger.refresh(queue_id, lambda bus: self._heard_frame(queue_id, bus))
        self._ledger.close_bus(
            queue_id,
            program_bus,
            terminal_state=OverlayProgramState.DROPPED,
            terminal_reason=ScheduleReason.FLOW_ENDED,
            keep_announcements=True,
            heard_frame=lambda bus: self._heard_frame(queue_id, bus),
        )
        del self._buses[queue_id]
        self._notify(queue_id, ProgramBusChange.RELEASED)

    def add_listener(self, listener: Callable[[ProgramBusEvent], None]) -> Callable[[], None]:
        """
        Subscribe to the queue buses appearing, being replaced and ending.

        :param listener: Called with the queue and the change for every bus change.
        :return: A callback that removes the listener again.
        """
        self._listeners.append(listener)

        def _remove() -> None:
            with suppress(ValueError):
                self._listeners.remove(listener)

        return _remove

    def _apply_levels(self, queue_id: str, program_bus: FlowProgramBus) -> None:
        """Hand the queue's level to a bus, so it mixes the programs the queue asked for."""
        levels = self.levels(queue_id)
        program_bus.set_levels(voice_gain=levels.voice_gain, duck_depth=levels.duck_depth)

    def _notify(self, queue_id: str, change: ProgramBusChange) -> None:
        """Tell the listeners what happened to the queue's program bus."""
        event = ProgramBusEvent(queue_id=queue_id, change=change)
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:
                self._logger.exception(
                    "A program bus listener failed for queue %s (%s)", queue_id, change
                )

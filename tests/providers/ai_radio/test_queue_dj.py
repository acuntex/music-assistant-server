"""Unit tests for the AI Radio sticky queue DJ state."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from music_assistant_models.enums import (
    ContentType,
    CrossfadeMode,
    EventType,
    MediaType,
    PlaybackState,
)
from music_assistant_models.errors import (
    InsufficientPermissions,
    InvalidDataError,
    MusicAssistantError,
)
from music_assistant_models.event import MassEvent
from music_assistant_models.media_items import AudioFormat

from music_assistant.controllers.streams.constants import LEAD_MIN_LEAD_SECONDS
from music_assistant.controllers.streams.controller import ScheduleReason, ScheduleResult
from music_assistant.controllers.streams.program_schedule import ProgramKind
from music_assistant.providers.ai_radio import queue_dj as queue_dj_module
from music_assistant.providers.ai_radio.constants import (
    ATTR_GAP_NEXT_ID,
    ATTR_HOST_ID,
    ATTR_QUEUE_DJ,
    ATTR_SESSION_ID,
    CONF_PLAY_INTRO_ON_ACTIVATE,
    EVENT_QUEUE_DJ_UPDATED,
    INTRO_RENDER_TIMEOUT_SECONDS,
    OVERLAY_LABEL_INTRO,
    OVERLAY_LABEL_TRANSITION,
    PROGRAM_BUS_OFF_HINT,
    PROGRAM_BUS_OFF_LINK,
    RENDER_FAILURES_BEFORE_COOLDOWN,
    TTS_COOLDOWN_SECONDS,
)
from music_assistant.providers.ai_radio.intro import AIRadioIntroMixin
from music_assistant.providers.ai_radio.media import AIRadioMediaMixin
from music_assistant.providers.ai_radio.models import (
    DJQueueState,
    EngineRenderError,
    PlannedSection,
    RenderPart,
)
from music_assistant.providers.ai_radio.queue_dj import AIRadioQueueDJMixin, _gap_can_air
from music_assistant.providers.ai_radio.runtime import AIRadioRuntimeMixin
from music_assistant.providers.ai_radio.storage import AIRadioStorageMixin
from music_assistant.providers.ai_radio.term_engine import BUS_GRACE_SECONDS, AIRadioTermEngineMixin
from music_assistant.providers.ai_radio.term_timing import (
    TERM_SCHEDULE_TOLERANCE_SECONDS,
    TERM_START_BIAS_SECONDS,
)
from music_assistant.providers.ai_radio.terms import BoundaryState, ProgramTerm, TermState

from .events import ProviderEventRecorder

# the PCM format a live flow session delivers a program in
PLAYBACK_PCM_FORMAT = AudioFormat(
    content_type=ContentType.PCM_S16LE,
    sample_rate=48000,
    bit_depth=16,
    channels=2,
)
# seconds of a spoken overlay stand-in a test renders unless it says otherwise
OVERLAY_PCM_SECONDS = 3.0
# the running track's length and how far into it the queue stands in the overlay tests
TRACK_SECONDS = 200
POSITION_SECONDS = 120


class FakeQueue:
    """Minimal PlayerQueue stand-in."""

    def __init__(
        self,
        queue_id: str,
        current_index: int | None,
        index_in_buffer: int | None,
        sources: list[Any] | None = None,
        ended: bool = False,
        state: PlaybackState = PlaybackState.PLAYING,
        current_item: FakeQueueItem | None = None,
        elapsed_time: float = 0.0,
    ) -> None:
        """Initialize the fake queue with its playback pointers."""
        self.queue_id = queue_id
        self.current_index = current_index
        self.index_in_buffer = index_in_buffer
        self.sources = sources or []
        self.ended = ended
        self.state = state
        self.current_item = current_item
        self.corrected_elapsed_time = elapsed_time


class FakeQueueItem:
    """Minimal QueueItem stand-in."""

    _counter = 0

    def __init__(
        self, name: str, duration: int | None = 200, extra: dict[str, Any] | None = None
    ) -> None:
        """Initialize the fake queue item with a unique id."""
        FakeQueueItem._counter += 1
        self.queue_item_id = f"qi{FakeQueueItem._counter}"
        # mirrors the insert order a real queue restores when it is un-shuffled
        self.sort_index = FakeQueueItem._counter
        self.name = name
        self.duration = duration
        self.media_item = None
        self.streamdetails = None
        self.available = True
        self.extra_attributes: dict[str, Any] = dict(extra or {})


class FakePlayerQueues:
    """Minimal PlayerQueues controller stand-in."""

    def __init__(self, queue: FakeQueue, items: list[FakeQueueItem]) -> None:
        """Initialize with one queue and its items."""
        self._queue = queue
        self._items = items
        # what each update_items call added, as (items, index), and what it dropped. the real
        # controller only ever replaces the whole list, so both are derived from the diff
        self.loads: list[tuple[list[Any], int]] = []
        self.deleted: list[str] = []
        self.update_calls = 0
        # stands in for the QUEUE_ITEMS_UPDATED event a real update_items emits
        self.on_items_updated: Callable[[], None] | None = None

    def get(self, queue_id: str) -> FakeQueue | None:
        """Return the queue when the id matches."""
        return self._queue if queue_id == self._queue.queue_id else None

    def items(self, queue_id: str, limit: int = 500, offset: int = 0) -> list[Any]:
        """Return one page of queue items."""
        return self._items[offset : offset + limit]

    def update_items(self, queue_id: str, queue_items: list[Any]) -> None:
        """Replace the queue items with the given list."""
        self.update_calls += 1
        previous_ids = {item.queue_item_id for item in self._items}
        current_ids = {item.queue_item_id for item in queue_items}
        self.deleted.extend(
            item.queue_item_id for item in self._items if item.queue_item_id not in current_ids
        )
        self.loads.extend(
            ([item], index)
            for index, item in enumerate(queue_items)
            if item.queue_item_id not in previous_ids
        )
        self._items = queue_items
        if self.on_items_updated is not None:
            self.on_items_updated()


class FakeMass:
    """Minimal MusicAssistant stand-in for the queue DJ mixin."""

    def __init__(self, player_queues: FakePlayerQueues) -> None:
        """Initialize with the fake player queues controller."""
        self.player_queues = player_queues
        self.config = StubConfig()
        self.tasks: list[asyncio.Task[Any]] = []
        self._tasks_by_id: dict[str, asyncio.Task[Any]] = {}
        self.timers: list[tuple[float, Callable[..., Any], tuple[Any, ...], str | None]] = []
        # (queue_id, required) for every boundary-mix requirement the DJ placed or released
        self.boundary_mix_calls: list[tuple[str, bool]] = []
        # the same, for the claim that holds a queue's tails for a lead
        self.lead_hold_calls: list[tuple[str, bool]] = []
        # the players the DJ asked to keep
        self.kept_players: list[str] = []
        self.players = SimpleNamespace(keep_player=self.kept_players.append)
        # and for the claim that streams the queue in flow mode, which the program bus needs
        self.flow_mode_calls: list[tuple[str, bool]] = []
        # the owner every flow claim was placed under
        self.flow_claim_owners: list[tuple[str, str]] = []

        def _set_boundary_mix(queue_id: str, _owner: str, required: bool) -> None:
            self.boundary_mix_calls.append((queue_id, required))

        def _set_lead_hold(queue_id: str, _owner: str, required: bool) -> None:
            self.lead_hold_calls.append((queue_id, required))

        def _set_flow_mode(queue_id: str, owner: str, required: bool) -> None:
            self.flow_mode_calls.append((queue_id, required))
            if required:
                self.flow_claim_owners.append((queue_id, owner))

        self.streams = SimpleNamespace(
            set_boundary_mix_required=_set_boundary_mix,
            set_lead_hold_required=_set_lead_hold,
            set_flow_mode_required=_set_flow_mode,
        )
        self._init_program_bus()

    def _init_program_bus(self) -> None:
        """Wire the program bus stubs the overlay paths talk to."""
        self.streams.program_bus_enabled_flag = False
        self.bus_queues: set[str] = set()
        # the seconds the queue's producer runs ahead of what the listener heard, as the
        # bus reports them; None is a bus that can not tell
        self.streams.program_bus_lead_value = None
        # the queue's effective crossfade mode, which moves a boundary earlier
        self.streams.crossfade_mode = CrossfadeMode.DISABLED
        # what the streams ledger reports about the programs placed on the queue
        self.streams.overlay_states: list[Any] = []
        # every program the DJ took back from the bus
        self.streams.cancelled: list[tuple[str, str, str, int | None]] = []
        # (queue_id, pcm, label, start_in_seconds) for every program the DJ mixed in
        self.overlays: list[tuple[str, bytes, str, float]] = []
        # the priority and kind the caller claimed for each of them
        self.overlay_priorities: list[object] = []
        self.overlay_kinds: list[object] = []
        # the head each program was planned to start with before its boundary
        self.overlay_heads: list[float] = []
        # (queue_id, voice db, duck db) for every level the DJ handed to the queue's bus
        self.program_levels: list[tuple[str, float, float | None]] = []

        def _program_bus_enabled() -> bool:
            return cast("bool", self.streams.program_bus_enabled_flag)

        def _program_bus_active(queue_id: str) -> bool:
            return queue_id in self.bus_queues

        def _program_bus_format(queue_id: str) -> AudioFormat | None:
            return PLAYBACK_PCM_FORMAT if queue_id in self.bus_queues else None

        def _program_bus_lead_seconds(_queue_id: str) -> float | None:
            return cast("float | None", self.streams.program_bus_lead_value)

        def _schedule_overlay(
            queue_id: str,
            pcm: bytes,
            *,
            label: str,
            start_in_seconds: float = 0.0,
            priority: object = None,
            kind: object = None,
            head_seconds: float = 0.0,
        ) -> ScheduleResult:
            if queue_id not in self.bus_queues:
                return ScheduleResult(
                    scheduled=False,
                    anchor_frame=0,
                    frames=0,
                    late_by_seconds=0.0,
                    lead_shortfall_seconds=0.0,
                    reason=ScheduleReason.NO_BUS,
                )
            self.overlays.append((queue_id, pcm, label, start_in_seconds))
            self.overlay_priorities.append(priority)
            self.overlay_kinds.append(kind)
            self.overlay_heads.append(head_seconds)
            return ScheduleResult(
                scheduled=True,
                anchor_frame=0,
                frames=0,
                late_by_seconds=0.0,
                lead_shortfall_seconds=0.0,
                reason=None,
            )

        def _get_crossfade_mode(_queue: Any) -> Any:
            return self.streams.crossfade_mode

        def _overlay_state(_queue_id: str) -> list[Any]:
            return self.streams.overlay_states

        def _set_program_levels(
            queue_id: str, *, voice_db: float = 0.0, duck_db: float | None = None
        ) -> None:
            self.program_levels.append((queue_id, voice_db, duck_db))

        def _cancel_overlay(
            queue_id: str, label: str, reason: str, *, start_frame: int | None = None
        ) -> bool:
            self.streams.cancelled.append((queue_id, label, reason, start_frame))
            return True

        self.streams.program_bus_enabled = _program_bus_enabled
        self.streams.program_bus_active = _program_bus_active
        self.streams.program_bus_format = _program_bus_format
        self.streams.program_bus_lead_seconds = _program_bus_lead_seconds
        self.streams.schedule_overlay = _schedule_overlay
        self.streams.get_crossfade_mode = _get_crossfade_mode
        self.streams.overlay_state = _overlay_state
        self.streams.cancel_overlay = _cancel_overlay
        self.streams.set_program_levels = _set_program_levels

    def activate_program_bus(self, queue_id: str) -> None:
        """Let the queue report a live flow session that mixes programs."""
        self.streams.program_bus_enabled_flag = True
        self.bus_queues.add(queue_id)

    def drop_program_bus(self, queue_id: str) -> None:
        """Withdraw the queue's flow session, as a stream that ended does."""
        self.bus_queues.discard(queue_id)

    def create_task(
        self, target: Coroutine[Any, Any, Any], task_id: str | None = None
    ) -> asyncio.Task[Any]:
        """Run the given coroutine as a task, deduplicating on task id like mass does."""
        if task_id and (existing := self._tasks_by_id.get(task_id)) and not existing.done():
            target.close()
            return existing
        task = asyncio.ensure_future(target)
        self.tasks.append(task)
        if task_id:
            self._tasks_by_id[task_id] = task
        return task

    def call_later(
        self, delay: float, target: Callable[..., Any], *args: Any, task_id: str | None = None
    ) -> None:
        """Record the delayed call instead of scheduling it."""
        self.timers.append((delay, target, args, task_id))

    def subscribe(self, *args: Any, **kwargs: Any) -> Any:
        """Return a no-op unsubscribe callback."""
        return lambda: None


class StubConfig:
    """Minimal ProviderConfig stand-in exposing get_value."""

    def get_value(self, key: str, default: Any = None) -> Any:
        """Return the default for every config key."""
        return default

    def get_raw_core_config_value(self, domain: str, key: str, default: Any = None) -> Any:
        """Return the default for every core config value, the crossfade length included."""
        return default


class OverlayStubs:
    """Config and engine stand-ins shared by the harnesses that mix programs in."""

    def _init_overlay_stubs(self) -> None:
        """Reset what a test configured and what the harness rendered."""
        self.config_values: dict[str, Any] = {}
        # (host id, section id) of every section rendered for an overlay
        self.rendered: list[tuple[str, str]] = []
        self.fail_overlay_render = False
        # which engine a failing overlay render stands in for
        self.fail_overlay_part = RenderPart.TTS

    def get_config_value(
        self,
        key: str,
        default: Any = None,
        *,
        return_type: Any = None,
    ) -> Any:
        """Return what the test configured for the key, or the default."""
        return self.config_values.get(key, default)

    async def _render_section_pcm(
        self,
        queue_id: str,
        program: dict[str, Any],
        section: PlannedSection,
        pcm_format: AudioFormat,
    ) -> bytes:
        """Return a stand-in clip in the flow's format, or fail as a broken engine does."""
        self.rendered.append((str(program.get("host_id") or ""), section.section_id))
        if self.fail_overlay_render:
            # an engine that is down fails the render, and only its own bucket pauses
            raise EngineRenderError(
                self.fail_overlay_part,
                f"{self.fail_overlay_part.label} 'engine-x'",
                "engine down",
            )
        frames = int(OVERLAY_PCM_SECONDS * pcm_format.sample_rate)
        return b"\x00\x00" * (frames * pcm_format.channels)


class DummyQueueDJ(
    OverlayStubs,
    ProviderEventRecorder,
    AIRadioRuntimeMixin,
    AIRadioTermEngineMixin,
    AIRadioIntroMixin,
    AIRadioQueueDJMixin,
    AIRadioStorageMixin,
    AIRadioMediaMixin,
):
    """Minimal harness for queue DJ state tests."""

    instance_id = "ai_radio_test"

    def __init__(self, tmp_path: Path) -> None:
        """Initialize dummy mixin state."""
        self.logger = logging.getLogger(__name__)
        self.config = cast("Any", StubConfig())
        self._init_overlay_stubs()
        self._hosts: dict[str, dict[str, Any]] = {
            "rick": {"id": "rick", "name": "Rick", "instructions": "x", "tts_engine": ""},
        }
        self._stations: dict[str, dict[str, Any]] = {}
        self._sections: dict[str, dict[str, Any]] = {}
        self._armed_show_queues: dict[str, str] = {}
        self._show_library_ids: dict[str, str] = {}
        self._dj_queues: dict[str, Any] = {}
        self._dj_file = tmp_path / "queue_dj.json"
        self._dj_lock = asyncio.Lock()
        self._unloading = False
        self.rendered_ahead: list[str] = []
        # the disable path reaches the real clip cleanup, which needs a queue layer;
        # this one holds no queue at all, so cleanup finds nothing to do
        # the mixin declares `mass: MusicAssistant`; a fake stands in for tests, so its own
        # attributes (not MusicAssistant's) are what callers here actually see
        self.mass: FakeMass = FakeMass(  # type: ignore[assignment]
            FakePlayerQueues(FakeQueue("other-queue", None, None), [])
        )

    def _schedule_replan(self, queue_id: str) -> None:
        """Record replan requests instead of running them."""
        self.replanned = getattr(self, "replanned", [])
        self.replanned.append(queue_id)

    def _render_ahead(self, queue_id: str) -> None:
        """Record render-ahead requests instead of rendering."""
        self.rendered_ahead.append(queue_id)


def _dj_harness(tmp_path: Path) -> DummyQueueDJ:
    """Build a queue DJ harness with a morning_show station bound to host amy."""
    dj = DummyQueueDJ(tmp_path)
    dj._hosts["amy"] = {"id": "amy", "name": "Amy", "instructions": "x", "tts_engine": ""}
    dj._stations["morning_show"] = {"id": "morning_show", "host_id": "amy"}
    return dj


def _player_removed(queue_id: str) -> MassEvent:
    """Build the PLAYER_REMOVED event of the player of a queue."""
    return MassEvent(EventType.PLAYER_REMOVED, queue_id)


def _fake_queue_with_sources(
    dj: DummyQueueDJ | ReplanQueueDJ, queue_id: str, source_uris: list[str]
) -> FakeQueue:
    """Register a fake queue exposing the given source uris on the harness's player_queues."""
    sources = [SimpleNamespace(uri=uri) for uri in source_uris]
    queue = FakeQueue(queue_id, None, None, sources=sources)
    dj.mass.player_queues._queue = queue
    return queue


class ReplanQueueDJ(
    OverlayStubs,
    ProviderEventRecorder,
    AIRadioRuntimeMixin,
    AIRadioTermEngineMixin,
    AIRadioIntroMixin,
    AIRadioQueueDJMixin,
    AIRadioStorageMixin,
    AIRadioMediaMixin,
):
    """Harness combining the queue DJ mixin with the real planner and clip builder."""

    instance_id = "ai_radio_test"
    domain = "ai_radio"

    def __init__(
        self,
        tmp_path: Path,
        queue: FakeQueue,
        items: list[FakeQueueItem],
        host: dict[str, Any],
    ) -> None:
        """Initialize the harness around one fake queue."""
        self.logger = logging.getLogger(__name__)
        self.config = cast("Any", StubConfig())
        self._init_overlay_stubs()
        self._sections = {_transition_section()["id"]: _transition_section()}
        self._hosts: dict[str, dict[str, Any]] = {host["id"]: host}
        self._stations: dict[str, dict[str, Any]] = {}
        self._armed_show_queues: dict[str, str] = {}
        self._show_library_ids: dict[str, str] = {}
        self._dj_queues: dict[str, Any] = {}
        self._dj_file = tmp_path / "queue_dj.json"
        self._dj_lock = asyncio.Lock()
        self._unloading = False
        self.player_queues = FakePlayerQueues(queue, items)
        # see DummyQueueDJ.__init__ for why this needs its own annotation
        self.mass: FakeMass = FakeMass(self.player_queues)  # type: ignore[assignment]

    def _render_ahead(self, queue_id: str) -> None:
        """Skip rendering ahead, which these tests do not exercise."""


def _transition_section() -> dict[str, Any]:
    """Return the single shared section used by the queue DJ test hosts."""
    return {
        "id": "Song_Transition",
        "name": "Song Transition",
        "type": "ai_text",
        "web_search": "disabled",
        "prompt": "From <prev_songinfo> to <next_songinfo>",
        "constraints": {"max_chars": 200},
    }


def _must_host() -> dict[str, Any]:
    """Return a host that always plans a section between songs."""
    return {
        "id": "rick",
        "name": "Rick",
        "instructions": "keep it short",
        "tts_engine": "",
        "section_ids": ["Song_Transition"],
        "section_order": [{"when": "between_songs", "flow": [{"MUST": "Song_Transition"}]}],
        "merge_section_id": "",
    }


def _optional_host(min_gap_songs: int) -> dict[str, Any]:
    """Return a host whose section is certain to fire but guarded by a song gap."""
    host = _must_host()
    host["section_order"] = [
        {
            "when": "between_songs",
            "flow": [
                {
                    "OPTIONAL": {
                        "section": "Song_Transition",
                        "chance": 1.0,
                        "guards": {"min_gap_songs": min_gap_songs},
                    }
                }
            ],
        }
    ]
    return host


def _hourly_sections() -> list[dict[str, Any]]:
    """Return two once-per-hour sections plus the ai_meta section that merges them."""
    return [
        {
            "id": "Weather",
            "name": "Weather",
            "type": "ai_text",
            "web_search": "disabled",
            "prompt": "Give the forecast",
            "constraints": {"max_chars": 200},
        },
        {
            "id": "News",
            "name": "News",
            "type": "ai_text",
            "web_search": "disabled",
            "prompt": "Give the headlines",
            "constraints": {"max_chars": 200},
        },
        {
            "id": "Smoother",
            "name": "Between Songs Mix",
            "type": "ai_meta",
            "prompt": "Combine these: <section_drafts>",
        },
    ]


def _hourly_host() -> dict[str, Any]:
    """Return a host whose two hourly sections merge into a single clip per gap."""
    host = _must_host()
    host["section_ids"] = ["Weather", "News", "Smoother"]
    host["section_order"] = [
        {
            "when": "between_songs",
            "flow": [
                {
                    "OPTIONAL": {
                        "section": "Weather",
                        "chance": 1.0,
                        "guards": {"max_per_60min": 1},
                    }
                },
                {
                    "OPTIONAL": {"section": "News", "chance": 1.0, "guards": {"max_per_60min": 1}},
                },
            ],
        }
    ]
    host["merge_section_id"] = "Smoother"
    return host


def _track(index: int) -> FakeQueueItem:
    """Return a fake music queue item."""
    return FakeQueueItem(f"Artist {index} - Song {index}")


def _dj_clip(gap_next_id: str, session_id: str) -> FakeQueueItem:
    """Return a fake DJ clip queue item announcing the given track."""
    return FakeQueueItem(
        "Song Transition",
        duration=30,
        extra={
            ATTR_QUEUE_DJ: True,
            ATTR_GAP_NEXT_ID: gap_next_id,
            ATTR_SESSION_ID: session_id,
        },
    )


def _make_replan_dj(
    tmp_path: Path,
    items: list[FakeQueueItem],
    current_index: int | None = 0,
    index_in_buffer: int | None = 0,
    host: dict[str, Any] | None = None,
) -> ReplanQueueDJ:
    """Build an armed replan harness around the given queue items."""
    queue = FakeQueue("queue-1", current_index, index_in_buffer)
    dummy = ReplanQueueDJ(tmp_path, queue, items, host or _must_host())
    # set_queue_dj marks a state plannable once its clip cleanup ran, and these harnesses
    # start from a queue whose switch already settled
    dummy._arm_dj_state("queue-1", "rick").ready = True
    return dummy


def _status_entry(
    host_id: str,
    station_id: str = "",
    voice_level_db: float = 0.0,
    duck_db: float | None = None,
    *,
    bus_off: bool = True,
) -> dict[str, Any]:
    """Return the status entry of one armed queue DJ, with the bus off by default."""
    entry: dict[str, Any] = {
        "host_id": host_id,
        "station_id": station_id,
        "voice_level_db": voice_level_db,
        "duck_db": duck_db,
        "bus_enabled": not bus_off,
        "hint": PROGRAM_BUS_OFF_HINT if bus_off else None,
        "hint_link": PROGRAM_BUS_OFF_LINK if bus_off else None,
    }
    return entry


async def test_arming_the_dj_with_the_bus_off_hints_at_the_switch(tmp_path: Path) -> None:
    """The hint is shown, and the DJ is armed anyway: the switch is never refused."""
    dummy = DummyQueueDJ(tmp_path)

    entry = (await dummy.set_queue_dj("queue-1", "rick"))["queue-1"]

    assert entry["bus_enabled"] is False
    assert entry["hint"]
    assert entry["hint_link"] == PROGRAM_BUS_OFF_LINK
    # the DJ speaks as soon as the bus is on, so arming it was the right thing to do
    assert dummy._dj_queues["queue-1"].host_id == "rick"
    assert dummy.replanned == ["queue-1"]


async def test_arming_the_dj_with_the_bus_on_hints_at_nothing(tmp_path: Path) -> None:
    """A queue whose stream mixes programs needs no hint."""
    dummy = DummyQueueDJ(tmp_path)
    dummy.mass.activate_program_bus("queue-1")

    entry = (await dummy.set_queue_dj("queue-1", "rick"))["queue-1"]

    assert entry["bus_enabled"] is True
    assert entry["hint"] is None
    assert entry["hint_link"] is None


async def test_the_status_answer_carries_the_same_hint(tmp_path: Path) -> None:
    """A client that only reads the status sees the hint the switch reported."""
    dummy = DummyQueueDJ(tmp_path)

    armed = (await dummy.set_queue_dj("queue-1", "rick"))["queue-1"]
    status = (await dummy.get_queue_dj_status())["queue-1"]

    assert status["hint"] == armed["hint"]
    assert status["bus_enabled"] is False


async def test_the_hint_leaves_the_other_answer_fields_alone(tmp_path: Path) -> None:
    """The answer keeps every field a client already reads, and adds the hint to them."""
    dummy = DummyQueueDJ(tmp_path)

    entry = (await dummy.set_queue_dj("queue-1", "rick"))["queue-1"]

    assert entry | {"bus_enabled": None, "hint": None, "hint_link": None} == {
        "host_id": "rick",
        "station_id": "",
        "voice_level_db": 0.0,
        "duck_db": None,
        "bus_enabled": None,
        "hint": None,
        "hint_link": None,
    }


async def test_set_queue_dj_enables_and_persists(tmp_path: Path) -> None:
    """Arm a queue DJ, persist it, and reload it into a fresh instance."""
    dummy = DummyQueueDJ(tmp_path)
    mapping = await dummy.set_queue_dj("queue-1", "rick")
    assert mapping == {"queue-1": _status_entry("rick")}
    assert dummy._dj_queues["queue-1"].host_id == "rick"
    assert dummy._dj_queues["queue-1"].dj_session_id
    assert dummy._dj_queues["queue-1"].ready is True
    assert dummy.replanned == ["queue-1"]
    assert dummy.mass.kept_players == ["queue-1"]
    assert dummy._dj_file.exists()

    fresh = DummyQueueDJ(tmp_path)
    await fresh._load_queue_dj()
    assert fresh._dj_queues["queue-1"].host_id == "rick"
    # nothing has to be cleaned up before a boot arm, so it may plan on the first event
    assert fresh._dj_queues["queue-1"].ready is True


async def test_set_queue_dj_rejects_unknown_host(tmp_path: Path) -> None:
    """Reject arming a queue DJ with an unknown host id."""
    dummy = DummyQueueDJ(tmp_path)
    with pytest.raises(InvalidDataError):
        await dummy.set_queue_dj("queue-1", "nobody")


async def test_the_legacy_cleanup_gets_the_live_session_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retried cleanup has to know which clips the session that runs now has spliced."""
    dummy = DummyQueueDJ(tmp_path)
    dummy._dj_queues["queue-1"] = DJQueueState(
        queue_id="queue-1", host_id="rick", dj_session_id="dj-live"
    )
    seen: list[tuple[str, dict[str, str]]] = []

    async def _record(
        _mass: Any, _logger: Any, instance_id: str, live_session_ids: dict[str, str]
    ) -> bool:
        seen.append((instance_id, dict(live_session_ids)))
        return False

    monkeypatch.setattr(queue_dj_module, "remove_legacy_dj_items", _record)

    await dummy._cleanup_legacy_dj_items()

    assert seen == [("ai_radio_test", {"queue-1": "dj-live"})]


async def test_set_queue_dj_none_disables(tmp_path: Path) -> None:
    """Disable an armed queue DJ by passing host_id=None."""
    dummy = DummyQueueDJ(tmp_path)
    await dummy.set_queue_dj("queue-1", "rick")
    mapping = await dummy.set_queue_dj("queue-1", None)
    assert mapping == {}
    assert dummy._dj_queues == {}


async def test_status_returns_mapping(tmp_path: Path) -> None:
    """Return the queue-to-host mapping for an armed queue DJ."""
    dummy = DummyQueueDJ(tmp_path)
    await dummy.set_queue_dj("queue-1", "rick")
    assert await dummy.get_queue_dj_status() == {"queue-1": _status_entry("rick")}


async def test_a_queue_mixes_at_the_defaults_until_it_sets_its_own(tmp_path: Path) -> None:
    """A freshly armed DJ speaks at full level and inherits the global duck."""
    dummy = DummyQueueDJ(tmp_path)

    mapping = await dummy.set_queue_dj("queue-1", "rick")

    assert mapping["queue-1"]["voice_level_db"] == 0.0
    assert mapping["queue-1"]["duck_db"] is None
    # the queue's bus is told where to mix the voice, so the announcement path agrees
    assert ("queue-1", 0.0, None) in dummy.mass.program_levels


async def test_setting_the_levels_persists_them_and_tells_the_bus(tmp_path: Path) -> None:
    """A queue keeps its own voice level and ducking depth, and the bus mixes at them."""
    dummy = DummyQueueDJ(tmp_path)
    await dummy.set_queue_dj("queue-1", "rick")

    await dummy.set_queue_dj_voice_level("queue-1", voice_level_db=-6)
    mapping = await dummy.set_queue_dj_duck("queue-1", duck_db=-12)

    assert mapping == {"queue-1": _status_entry("rick", voice_level_db=-6.0, duck_db=-12.0)}
    assert dummy.mass.program_levels[-1] == ("queue-1", -6.0, -12.0)


async def test_setting_only_the_duck_keeps_the_voice_level_it_had(tmp_path: Path) -> None:
    """A ui that saves one slider does not reset the other setting to its default."""
    dummy = DummyQueueDJ(tmp_path)
    await dummy.set_queue_dj("queue-1", "rick")
    await dummy.set_queue_dj_voice_level("queue-1", voice_level_db=-6)

    mapping = await dummy.set_queue_dj_duck("queue-1", duck_db=-3)

    assert (mapping["queue-1"]["voice_level_db"], mapping["queue-1"]["duck_db"]) == (-6.0, -3.0)
    assert dummy.mass.program_levels[-1] == ("queue-1", -6.0, -3.0)


async def test_setting_only_the_voice_level_keeps_the_duck_it_had(tmp_path: Path) -> None:
    """The other side of the same coin: the duck stays where the queue put it."""
    dummy = DummyQueueDJ(tmp_path)
    await dummy.set_queue_dj("queue-1", "rick")
    await dummy.set_queue_dj_duck("queue-1", duck_db=-12)

    mapping = await dummy.set_queue_dj_voice_level("queue-1", voice_level_db=-4.5)

    assert (mapping["queue-1"]["voice_level_db"], mapping["queue-1"]["duck_db"]) == (
        -4.5,
        -12.0,
    )
    assert dummy.mass.program_levels[-1] == ("queue-1", -4.5, -12.0)


async def test_both_commands_set_the_two_independent_levels(tmp_path: Path) -> None:
    """A settings dialog may still carry both sliders, one call per setter."""
    dummy = DummyQueueDJ(tmp_path)
    await dummy.set_queue_dj("queue-1", "rick")
    await dummy.set_queue_dj_voice_level("queue-1", voice_level_db=-6)

    await dummy.set_queue_dj_voice_level("queue-1", voice_level_db=-9)
    mapping = await dummy.set_queue_dj_duck("queue-1", duck_db=-12)

    assert (mapping["queue-1"]["voice_level_db"], mapping["queue-1"]["duck_db"]) == (-9.0, -12.0)


async def test_an_explicit_null_returns_the_voice_level_to_its_default(tmp_path: Path) -> None:
    """None is a value, not an omission: the queue speaks at 0 dB again, the duck stays."""
    dummy = DummyQueueDJ(tmp_path)
    await dummy.set_queue_dj("queue-1", "rick")
    await dummy.set_queue_dj_voice_level("queue-1", voice_level_db=-6)
    await dummy.set_queue_dj_duck("queue-1", duck_db=-12)

    mapping = await dummy.set_queue_dj_voice_level("queue-1", voice_level_db=None)

    assert mapping["queue-1"]["voice_level_db"] == 0.0
    assert mapping["queue-1"]["duck_db"] == -12.0


async def test_an_explicit_null_returns_the_duck_to_the_global_depth(tmp_path: Path) -> None:
    """The duck's own default is 'inherit the depth the bus mixes every queue with'."""
    dummy = DummyQueueDJ(tmp_path)
    await dummy.set_queue_dj("queue-1", "rick")
    await dummy.set_queue_dj_voice_level("queue-1", voice_level_db=-6)
    await dummy.set_queue_dj_duck("queue-1", duck_db=-12)

    mapping = await dummy.set_queue_dj_duck("queue-1", duck_db=None)

    assert mapping["queue-1"]["duck_db"] is None
    assert mapping["queue-1"]["voice_level_db"] == -6.0
    assert dummy.mass.program_levels[-1] == ("queue-1", -6.0, None)


async def test_the_levels_survive_a_restart(tmp_path: Path) -> None:
    """A restored DJ reads its levels back and hands them to the new stream's bus."""
    dummy = DummyQueueDJ(tmp_path)
    await dummy.set_queue_dj("queue-1", "rick")
    await dummy.set_queue_dj_voice_level("queue-1", voice_level_db=-4.5)
    await dummy.set_queue_dj_duck("queue-1", duck_db=-12)

    fresh = DummyQueueDJ(tmp_path)
    await fresh._load_queue_dj()

    state = fresh._dj_queues["queue-1"]
    assert (state.voice_level_db, state.duck_db) == (-4.5, -12.0)
    assert fresh.mass.program_levels[-1] == ("queue-1", -4.5, -12.0)


async def test_a_stored_dj_state_without_the_levels_reads_the_defaults(tmp_path: Path) -> None:
    """A queue DJ file written before the settings existed still loads its host."""
    dummy = DummyQueueDJ(tmp_path)
    dummy._dj_file.write_text(
        '{"version": 1, "queues": {"queue-1": {"host_id": "rick", "station_id": ""}}}'
    )

    await dummy._load_queue_dj()

    state = dummy._dj_queues["queue-1"]
    assert state.host_id == "rick"
    assert (state.voice_level_db, state.duck_db) == (0.0, None)
    assert dummy.mass.program_levels == [("queue-1", 0.0, None)]


async def test_setting_the_levels_of_a_queue_without_a_dj_is_refused(tmp_path: Path) -> None:
    """Levels belong to an armed DJ, so a queue that runs none can not set them."""
    dummy = DummyQueueDJ(tmp_path)

    with pytest.raises(InvalidDataError, match="queue-1"):
        await dummy.set_queue_dj_voice_level("queue-1", voice_level_db=-6)
    with pytest.raises(InvalidDataError, match="queue-1"):
        await dummy.set_queue_dj_duck("queue-1", duck_db=-6)


@pytest.mark.parametrize("voice_level_db", [-9.1, 0.5, "loud", True])
async def test_a_voice_level_outside_its_range_is_refused(
    tmp_path: Path, voice_level_db: Any
) -> None:
    """A value outside its range or of the wrong type is an error, not a clamp."""
    dummy = DummyQueueDJ(tmp_path)
    await dummy.set_queue_dj("queue-1", "rick")

    with pytest.raises(InvalidDataError):
        await dummy.set_queue_dj_voice_level("queue-1", voice_level_db=voice_level_db)


@pytest.mark.parametrize("duck_db", [-12.1, 0.5, "deep", True])
async def test_a_duck_outside_its_range_is_refused(tmp_path: Path, duck_db: Any) -> None:
    """The ducking depth has its own range, and the same refusal."""
    dummy = DummyQueueDJ(tmp_path)
    await dummy.set_queue_dj("queue-1", "rick")

    with pytest.raises(InvalidDataError):
        await dummy.set_queue_dj_duck("queue-1", duck_db=duck_db)


@pytest.mark.parametrize("voice_level_db", [0, -9, -4.5])
async def test_the_bounds_of_the_voice_range_are_valid(tmp_path: Path, voice_level_db: Any) -> None:
    """0 dB and the floor are settings a user may pick."""
    dummy = DummyQueueDJ(tmp_path)
    await dummy.set_queue_dj("queue-1", "rick")

    mapping = await dummy.set_queue_dj_voice_level("queue-1", voice_level_db=voice_level_db)

    assert mapping["queue-1"]["voice_level_db"] == pytest.approx(float(voice_level_db))


@pytest.mark.parametrize("duck_db", [0, -12, -4.5, None])
async def test_the_bounds_of_the_duck_range_are_valid(tmp_path: Path, duck_db: Any) -> None:
    """0 dB and the floor are settings a user may pick, and None is 'inherit'."""
    dummy = DummyQueueDJ(tmp_path)
    await dummy.set_queue_dj("queue-1", "rick")

    mapping = await dummy.set_queue_dj_duck("queue-1", duck_db=duck_db)

    expected_duck = None if duck_db is None else float(duck_db)
    if expected_duck is None:
        assert mapping["queue-1"]["duck_db"] is None
    else:
        assert mapping["queue-1"]["duck_db"] == pytest.approx(expected_duck)


async def test_a_host_switch_keeps_the_levels_of_the_queue(tmp_path: Path) -> None:
    """The level belongs to the queue, so switching its host does not reset it."""
    dummy = DummyQueueDJ(tmp_path)
    dummy._hosts["amy"] = {"id": "amy", "name": "Amy", "instructions": "x", "tts_engine": ""}
    await dummy.set_queue_dj("queue-1", "rick")
    await dummy.set_queue_dj_voice_level("queue-1", voice_level_db=-6)
    await dummy.set_queue_dj_duck("queue-1", duck_db=-12)

    mapping = await dummy.set_queue_dj("queue-1", "amy")

    assert mapping["queue-1"] == _status_entry("amy", voice_level_db=-6.0, duck_db=-12.0)


async def test_disabling_the_dj_hands_the_defaults_back_to_the_bus(tmp_path: Path) -> None:
    """A disabled DJ leaves no gain behind on the queue's announcements."""
    dummy = DummyQueueDJ(tmp_path)
    await dummy.set_queue_dj("queue-1", "rick")
    await dummy.set_queue_dj_voice_level("queue-1", voice_level_db=-9)
    await dummy.set_queue_dj_duck("queue-1", duck_db=-12)

    await dummy.set_queue_dj("queue-1", None)

    assert dummy.mass.program_levels[-1] == ("queue-1", 0.0, None)


async def test_a_show_that_detaches_and_replays_keeps_the_level_of_its_queue(
    tmp_path: Path,
) -> None:
    """The level belongs to the queue, so re-arming the show after a detach mixes it again."""
    dj = _dj_harness(tmp_path)
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])
    await dj._ensure_show_dj("q1")
    await dj.set_queue_dj_voice_level("q1", voice_level_db=-6)
    await dj.set_queue_dj_duck("q1", duck_db=-12)

    # the show ran out: its DJ detaches on the next event without the user asking for it
    dj.mass.player_queues._queue.ended = True
    await dj._maybe_detach_show_dj("q1")
    assert "q1" not in dj._dj_queues

    # the listener plays the show again
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])
    await dj._ensure_show_dj("q1")

    state = dj._dj_queues["q1"]
    assert state.host_id == "amy"
    assert (state.voice_level_db, state.duck_db) == (-6.0, -12.0)
    assert dj.mass.program_levels[-1] == ("q1", -6.0, -12.0)


async def test_the_level_of_a_queue_survives_its_dj_being_switched_off_and_on(
    tmp_path: Path,
) -> None:
    """A disabled DJ hands the bus defaults, and the queue's own level is not thrown away."""
    dummy = DummyQueueDJ(tmp_path)
    await dummy.set_queue_dj("queue-1", "rick")
    await dummy.set_queue_dj_voice_level("queue-1", voice_level_db=-4.5)

    await dummy.set_queue_dj("queue-1", None)
    assert dummy.mass.program_levels[-1] == ("queue-1", 0.0, None)

    mapping = await dummy.set_queue_dj("queue-1", "rick")

    assert mapping["queue-1"]["voice_level_db"] == pytest.approx(-4.5)
    assert dummy.mass.program_levels[-1] == ("queue-1", -4.5, None)


async def test_a_removed_player_forgets_the_level_of_its_queue(tmp_path: Path) -> None:
    """A player that is gone takes its queue's level with it instead of holding it forever."""
    dummy = DummyQueueDJ(tmp_path)
    await dummy.set_queue_dj("queue-1", "rick")
    await dummy.set_queue_dj_voice_level("queue-1", voice_level_db=-9)

    await dummy._on_dj_queue_event(_player_removed("queue-1"))

    mapping = await dummy.set_queue_dj("queue-1", "rick")
    assert mapping["queue-1"]["voice_level_db"] == 0.0


async def test_the_item_path_keeps_its_clips_with_levels_set(tmp_path: Path) -> None:
    """With the program bus off, the levels change nothing about the clips it splices."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, tracks)
    await dummy.set_queue_dj_voice_level("queue-1", voice_level_db=-6)
    await dummy.set_queue_dj_duck("queue-1", duck_db=-12)

    await dummy._replan_queue("queue-1")

    assert dummy.mass.overlays == []
    assert {items[0].extra_attributes[ATTR_GAP_NEXT_ID] for items, _ in dummy.player_queues.loads}


async def test_ensure_show_dj_arms_host_for_show_queue(tmp_path: Path) -> None:
    """Playing a show with no DJ armed yet arms its station's host."""
    dj = _dj_harness(tmp_path)
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])

    await dj._ensure_show_dj("q1")

    state = dj._dj_queues["q1"]
    assert state.host_id == "amy"
    assert state.station_id == "morning_show"
    assert dj._armed_show_queues["q1"] == "morning_show"


def _let_amy_post(dj: DummyQueueDJ) -> None:
    """Give host amy one section whose break carries over the next track."""
    dj._sections["talk_over"] = {"id": "talk_over", "allow_post": True}
    dj._hosts["amy"]["section_ids"] = ["talk_over"]


async def test_arming_a_posting_show_requires_mixed_boundaries(tmp_path: Path) -> None:
    """A show whose host carries a break over the next track asks for mixed boundaries."""
    dj = _dj_harness(tmp_path)
    _let_amy_post(dj)
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])

    await dj._ensure_show_dj("q1")

    assert dj.mass.boundary_mix_calls == [("q1", True)]
    # its break only plays into the item after the boundary, so no tail is held for a lead
    assert dj.mass.lead_hold_calls == [("q1", False)]


def _let_amy_lead(dj: DummyQueueDJ) -> None:
    """Give host amy one section whose break starts over the outgoing track's tail."""
    dj._sections["talk_over"] = {"id": "talk_over", "allow_pre": True}
    dj._hosts["amy"]["section_ids"] = ["talk_over"]


async def test_arming_a_leading_show_holds_the_queue_tails(tmp_path: Path) -> None:
    """A show whose break starts over the outgoing tail claims the hold that carries a lead."""
    dj = _dj_harness(tmp_path)
    _let_amy_lead(dj)
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])

    await dj._ensure_show_dj("q1")

    assert dj.mass.lead_hold_calls == [("q1", True)]
    # the break straddles the boundary as well, so both claims are placed
    assert dj.mass.boundary_mix_calls == [("q1", True)]


async def test_arming_a_show_without_posting_sections_asks_for_nothing(tmp_path: Path) -> None:
    """A show whose breaks never carry over places no requirement on the queue."""
    dj = _dj_harness(tmp_path)
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])

    await dj._ensure_show_dj("q1")

    assert dj.mass.boundary_mix_calls == [("q1", False)]
    assert dj.mass.lead_hold_calls == [("q1", False)]


async def test_disabling_the_show_dj_releases_the_requirement(tmp_path: Path) -> None:
    """Detaching or disabling the show's host lets go of the mixed boundaries."""
    dj = _dj_harness(tmp_path)
    _let_amy_post(dj)
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])
    await dj._ensure_show_dj("q1")

    await dj.set_queue_dj("q1", None)

    assert dj.mass.boundary_mix_calls == [("q1", True), ("q1", False)]


async def test_ensure_show_dj_skips_when_already_armed_this_play(tmp_path: Path) -> None:
    """A show already recorded as armed this play (e.g. a manual disable) is not re-armed."""
    dj = _dj_harness(tmp_path)
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])
    dj._armed_show_queues["q1"] = "morning_show"

    await dj._ensure_show_dj("q1")

    assert "q1" not in dj._dj_queues


async def test_ensure_show_dj_leaves_a_manual_dj_alone(tmp_path: Path) -> None:
    """A manual DJ (no station binding) is never bound to the show nor auto-detached."""
    dj = _dj_harness(tmp_path)
    queue = _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])
    state = dj._arm_dj_state("q1", "bob")
    state.ready = True

    await dj._ensure_show_dj("q1")

    assert dj._dj_queues["q1"].host_id == "bob"
    assert dj._dj_queues["q1"].station_id == ""
    assert "q1" not in dj._armed_show_queues

    queue.ended = True
    await dj._maybe_detach_show_dj("q1")

    assert dj._dj_queues["q1"] is state


async def test_show_binding_survives_a_restart(tmp_path: Path) -> None:
    """The station binding of an auto-armed show DJ is persisted and restored."""
    dj = _dj_harness(tmp_path)
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])
    await dj._ensure_show_dj("q1")

    fresh = _dj_harness(tmp_path)
    await fresh._load_queue_dj()

    assert fresh._dj_queues["q1"].host_id == "amy"
    assert fresh._dj_queues["q1"].station_id == "morning_show"


async def test_host_switch_keeps_the_show_binding(tmp_path: Path) -> None:
    """Switching the host mid-show keeps the station binding of the replaced state."""
    dj = _dj_harness(tmp_path)
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])
    await dj._ensure_show_dj("q1")

    await dj.set_queue_dj("q1", "rick")

    assert dj._dj_queues["q1"].host_id == "rick"
    assert dj._dj_queues["q1"].station_id == "morning_show"


async def test_manual_disable_mid_show_is_not_rearmed(tmp_path: Path) -> None:
    """Turning off the show's own DJ mid-show must not bring it back on the next check."""
    dj = _dj_harness(tmp_path)
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])
    await dj._ensure_show_dj("q1")
    assert "q1" in dj._dj_queues

    await dj.set_queue_dj("q1", None)
    await dj._ensure_show_dj("q1")

    assert "q1" not in dj._dj_queues


async def test_replacing_show_a_with_show_b_arms_bs_host(tmp_path: Path) -> None:
    """Playing show B over show A detaches A's DJ and arms B's on the following event."""
    dj = _dj_harness(tmp_path)
    dj._hosts["bob"] = {"id": "bob", "name": "Bob", "instructions": "x", "tts_engine": ""}
    dj._stations["evening_show"] = {"id": "evening_show", "host_id": "bob"}
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])
    await dj._ensure_show_dj("q1")
    assert dj._dj_queues["q1"].host_id == "amy"

    # the queue's sources now name show B; the same event ensures, then detaches
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/evening_show"])
    await dj._ensure_show_dj("q1")
    # A's binding must not have stamped B as already armed
    assert dj._armed_show_queues.get("q1") != "evening_show"
    await dj._maybe_detach_show_dj("q1")
    assert "q1" not in dj._dj_queues

    # the next event arms B's host afresh
    await dj._ensure_show_dj("q1")
    assert dj._dj_queues["q1"].host_id == "bob"
    assert dj._dj_queues["q1"].station_id == "evening_show"


async def test_auto_armed_dj_detaches_and_forgets_the_armed_record_when_source_gone(
    tmp_path: Path,
) -> None:
    """An auto-armed DJ whose show left the queue's sources is disarmed and forgotten."""
    dj = _dj_harness(tmp_path)
    _fake_queue_with_sources(dj, "q1", [])
    state = dj._arm_dj_state("q1", "amy")
    state.ready = True
    state.station_id = "morning_show"
    dj._armed_show_queues["q1"] = "morning_show"

    await dj._maybe_detach_show_dj("q1")

    assert "q1" not in dj._dj_queues
    assert "q1" not in dj._armed_show_queues


async def test_show_dj_survives_its_player_dropping(tmp_path: Path) -> None:
    """A show DJ and its arm-once record stay while the queue is temporarily unregistered."""
    dj = _dj_harness(tmp_path)
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])
    await dj._ensure_show_dj("q1")
    state = dj._dj_queues["q1"]

    _fake_queue_with_sources(dj, "other", [])
    await dj._maybe_detach_show_dj("q1")
    dj._forget_armed_show("q1")

    assert dj._dj_queues["q1"] is state
    assert dj._armed_show_queues["q1"] == "morning_show"


async def test_manually_armed_dj_is_never_auto_detached(tmp_path: Path) -> None:
    """A DJ with no station binding (a manual pick) is left alone by the auto-detach."""
    dj = _dj_harness(tmp_path)
    _fake_queue_with_sources(dj, "q1", [])
    state = dj._arm_dj_state("q1", "amy")
    state.ready = True

    await dj._maybe_detach_show_dj("q1")

    assert "q1" in dj._dj_queues


async def test_detaching_a_show_dj_removes_its_pending_clips(tmp_path: Path) -> None:
    """An auto-detach takes the show's pending clips out of the queue along with its state."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    state = dummy._dj_queues["queue-1"]
    state.station_id = "morning_show"
    dummy._armed_show_queues["queue-1"] = "morning_show"
    first = _dj_clip(tracks[2].queue_item_id, state.dj_session_id)
    second = _dj_clip(tracks[3].queue_item_id, state.dj_session_id)
    dummy.player_queues._items = [tracks[0], tracks[1], first, tracks[2], second, tracks[3]]

    await dummy._maybe_detach_show_dj("queue-1")

    assert "queue-1" not in dummy._dj_queues
    assert "queue-1" not in dummy._armed_show_queues
    assert {first.queue_item_id, second.queue_item_id} <= set(dummy.player_queues.deleted)


async def test_status_includes_station_binding(tmp_path: Path) -> None:
    """The status payload surfaces the station a queue DJ is bound to."""
    dj = _dj_harness(tmp_path)
    state = dj._arm_dj_state("q1", "amy")
    state.station_id = "morning_show"

    status = await dj.get_queue_dj_status()

    assert status == {"q1": _status_entry("amy", station_id="morning_show")}


async def test_on_dj_queue_event_auto_arms_a_show_queue(tmp_path: Path) -> None:
    """A queue event for a freshly loaded show arms its host without an explicit dj call."""
    dj = _dj_harness(tmp_path)
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])

    await dj._on_dj_queue_event(
        cast("Any", SimpleNamespace(event=EventType.QUEUE_ADDED, object_id="q1"))
    )

    state = dj._dj_queues["q1"]
    assert state.host_id == "amy"
    assert state.station_id == "morning_show"
    # armed twice: once by set_queue_dj's own arm, once by the event's usual replan request
    assert dj.replanned == ["q1", "q1"]
    assert dj.rendered_ahead == ["q1"]


async def test_on_dj_queue_event_renders_ahead_only_once_the_dj_is_ready(tmp_path: Path) -> None:
    """A DJ still clearing the previous host's clips renders nothing ahead."""
    dj = _dj_harness(tmp_path)
    _fake_queue_with_sources(dj, "q1", [])
    state = dj._arm_dj_state("q1", "amy")
    event = cast("Any", SimpleNamespace(event=EventType.QUEUE_UPDATED, object_id="q1"))

    await dj._on_dj_queue_event(event)
    assert dj.rendered_ahead == []

    state.ready = True
    await dj._on_dj_queue_event(event)
    assert dj.rendered_ahead == ["q1"]


async def test_on_dj_queue_event_does_not_rearm_a_manually_disabled_show(tmp_path: Path) -> None:
    """A user disabling the DJ mid-show must not get it re-armed by the next queue event."""
    dj = _dj_harness(tmp_path)
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])
    await dj._ensure_show_dj("q1")
    assert "q1" in dj._dj_queues

    await dj.set_queue_dj("q1", None)
    assert "q1" not in dj._dj_queues

    await dj._on_dj_queue_event(
        cast("Any", SimpleNamespace(event=EventType.QUEUE_ITEMS_UPDATED, object_id="q1"))
    )

    assert "q1" not in dj._dj_queues
    assert dj._armed_show_queues["q1"] == "morning_show"
    assert dj.rendered_ahead == []


async def test_ending_the_queue_detaches_and_clears_the_armed_record(tmp_path: Path) -> None:
    """An ended queue's auto-armed DJ detaches, and a later replay can arm it again."""
    dj = _dj_harness(tmp_path)
    queue = _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])
    queue.ended = True

    await dj._on_dj_queue_event(
        cast("Any", SimpleNamespace(event=EventType.QUEUE_ITEMS_UPDATED, object_id="q1"))
    )

    assert "q1" not in dj._dj_queues
    assert "q1" not in dj._armed_show_queues
    # the detached DJ's clips are not rendered ahead on its way out
    assert dj.rendered_ahead == []

    # a genuine replay resets the queue before the show is enqueued on it again
    queue.ended = False
    await dj._ensure_show_dj("q1")

    assert dj._dj_queues["q1"].host_id == "amy"


async def test_a_show_replay_earns_a_fresh_outro(tmp_path: Path) -> None:
    """A replay of a finished show plans its outro again, exactly once."""
    tracks = [_track(0), _track(1)]
    queue = FakeQueue(
        "queue-1", -1, -1, sources=[SimpleNamespace(uri="ai_radio_test://radio/morning_show")]
    )
    dummy = ReplanQueueDJ(tmp_path, queue, list(tracks), _outro_host())
    dummy._stations["morning_show"] = {"id": "morning_show", "host_id": "rick"}

    await dummy._ensure_show_dj("queue-1")
    await _drain_tasks(dummy.mass)
    assert len(_outro_clips(dummy)) == 1

    # the show runs out, so its DJ detaches and the replay enqueues the songs again
    queue.ended = True
    await dummy._maybe_detach_show_dj("queue-1")
    assert "queue-1" not in dummy._dj_queues

    queue.ended = False
    dummy.player_queues._items = [_track(10), _track(11)]
    await dummy._ensure_show_dj("queue-1")
    await _drain_tasks(dummy.mass)

    outros = _outro_clips(dummy)
    assert len(outros) == 1
    assert dummy.player_queues.items("queue-1")[-1] is outros[0]


async def test_on_dj_queue_event_forgets_armed_show_on_player_removed(tmp_path: Path) -> None:
    """Removing the player behind a show-bound DJ also forgets its arm-once record."""
    dj = _dj_harness(tmp_path)
    _fake_queue_with_sources(dj, "q1", [])
    state = dj._arm_dj_state("q1", "amy")
    state.station_id = "morning_show"
    dj._armed_show_queues["q1"] = "morning_show"

    await dj._on_dj_queue_event(_player_removed("q1"))

    assert "q1" not in dj._dj_queues
    assert "q1" not in dj._armed_show_queues


async def test_a_show_dj_outlives_a_disabled_player(tmp_path: Path) -> None:
    """
    A show-bound DJ and its arm-once record stay while the player is disabled, and run again.

    A disable takes the queue away and sends a last QUEUE_UPDATED for it, no PLAYER_REMOVED.
    """
    dj = _dj_harness(tmp_path)
    show = [f"{dj.instance_id}://radio/morning_show"]
    _fake_queue_with_sources(dj, "q1", show)
    await dj._ensure_show_dj("q1")
    _fake_queue_with_sources(dj, "other-queue", [])

    await dj._on_dj_queue_event(MassEvent(EventType.QUEUE_UPDATED, "q1"))

    assert dj._dj_queues["q1"].station_id == "morning_show"
    assert dj._armed_show_queues["q1"] == "morning_show"

    dj.replanned.clear()
    dj.rendered_ahead.clear()
    _fake_queue_with_sources(dj, "q1", show)
    await dj._on_dj_queue_event(MassEvent(EventType.QUEUE_ADDED, "q1"))

    assert dj._dj_queues["q1"].station_id == "morning_show"
    assert dj._armed_show_queues["q1"] == "morning_show"
    assert dj.replanned == ["q1"]
    assert dj.rendered_ahead == ["q1"]


async def test_a_show_dj_goes_with_a_removed_player(tmp_path: Path) -> None:
    """A removal takes the queue away as a disable does, then PLAYER_REMOVED drops both."""
    dj = _dj_harness(tmp_path)
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])
    await dj._ensure_show_dj("q1")
    _fake_queue_with_sources(dj, "other-queue", [])

    await dj._on_dj_queue_event(MassEvent(EventType.QUEUE_UPDATED, "q1"))
    await dj._on_dj_queue_event(_player_removed("q1"))

    assert "q1" not in dj._dj_queues
    assert "q1" not in dj._armed_show_queues


async def test_ensure_show_dj_arms_via_the_shows_library_uri(tmp_path: Path) -> None:
    """A queue sourcing the show by its library identity still arms the show's host."""
    dj = _dj_harness(tmp_path)
    dj._show_library_ids = {"7": "morning_show"}
    _fake_queue_with_sources(dj, "q1", ["library://radio/7"])

    await dj._ensure_show_dj("q1")

    state = dj._dj_queues["q1"]
    assert state.host_id == "amy"
    assert state.station_id == "morning_show"


async def test_ensure_show_dj_skips_arming_for_a_deleted_host(tmp_path: Path) -> None:
    """A show whose host was deleted is not armed, and the event handler does not raise."""
    dj = _dj_harness(tmp_path)
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])
    del dj._hosts["amy"]

    await dj._ensure_show_dj("q1")

    assert "q1" not in dj._dj_queues


async def test_forget_armed_show_keeps_the_record_while_sources_still_match(
    tmp_path: Path,
) -> None:
    """The arm-once record survives a queue event as long as sources still name the show."""
    dj = _dj_harness(tmp_path)
    _fake_queue_with_sources(dj, "q1", [f"{dj.instance_id}://radio/morning_show"])
    dj._armed_show_queues["q1"] = "morning_show"

    await dj._on_dj_queue_event(
        cast("Any", SimpleNamespace(event=EventType.QUEUE_UPDATED, object_id="q1"))
    )

    assert dj._armed_show_queues["q1"] == "morning_show"


async def test_forget_armed_show_drops_the_record_once_sources_change(tmp_path: Path) -> None:
    """The independent cleanup forgets a queue's record once its sources drop the show."""
    dj = _dj_harness(tmp_path)
    _fake_queue_with_sources(dj, "q1", [])
    dj._armed_show_queues["q1"] = "morning_show"

    await dj._on_dj_queue_event(
        cast("Any", SimpleNamespace(event=EventType.QUEUE_UPDATED, object_id="q1"))
    )

    assert "q1" not in dj._armed_show_queues


async def test_gap_can_air_needs_the_outgoing_track_to_play() -> None:
    """A track that could not start is no boundary to speak over."""
    tracks = [{"item_id": "a", "available": False}, {"item_id": "b", "available": True}]

    assert _gap_can_air(tracks, 1) is False


def test_gap_can_air_needs_the_next_track_to_play() -> None:
    """Neither is a break worth writing when the track it announces cannot play."""
    tracks = [{"item_id": "a", "available": True}, {"item_id": "b", "available": False}]

    assert _gap_can_air(tracks, 1) is False


def test_gap_can_air_for_an_outro_needs_the_last_track() -> None:
    """An outro after a track that cannot play has nothing to close."""
    tracks = [{"item_id": "a", "available": True}, {"item_id": "b", "available": False}]

    assert _gap_can_air(tracks, 2) is False


def test_gap_can_air_allows_a_boundary_that_plays() -> None:
    """Both sides can play, so the gap earns its break."""
    tracks = [{"item_id": "a", "available": True}, {"item_id": "b", "available": True}]

    assert _gap_can_air(tracks, 1) is True


def test_repair_drops_a_break_behind_a_track_that_cannot_play(tmp_path: Path) -> None:
    """A break behind a track that never started is removed, and stays decided."""
    tracks = [_track(0), _track(1)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    state = dummy._dj_queues["queue-1"]
    clip = _dj_clip(tracks[1].queue_item_id, state.dj_session_id)
    tracks[0].available = False
    dummy.player_queues._items = [tracks[0], clip, tracks[1]]

    repaired = dummy._repair_dj_clips("queue-1", state, dummy.player_queues.items("queue-1"), -1)

    assert repaired is True
    assert clip.queue_item_id in dummy.player_queues.deleted
    # re-deciding the gap would render another break for the next skipped track
    assert tracks[1].queue_item_id in state.decided_gap_ids


async def test_repair_drops_a_break_that_announces_a_track_that_cannot_play(
    tmp_path: Path,
) -> None:
    """The break's announced track cannot play, so the break has nothing to hand over to."""
    tracks = [_track(0), _track(1)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    state = dummy._dj_queues["queue-1"]
    clip = _dj_clip(tracks[1].queue_item_id, state.dj_session_id)
    tracks[1].available = False
    dummy.player_queues._items = [tracks[0], clip, tracks[1]]

    repaired = dummy._repair_dj_clips("queue-1", state, dummy.player_queues.items("queue-1"), -1)

    assert repaired is True
    assert clip.queue_item_id in dummy.player_queues.deleted
    assert tracks[1].queue_item_id in state.decided_gap_ids


async def test_repair_keeps_a_break_in_front_of_a_track_that_plays(tmp_path: Path) -> None:
    """Both boundaries can play, so the break stays and its gap is not closed."""
    tracks = [_track(0), _track(1)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    state = dummy._dj_queues["queue-1"]
    clip = _dj_clip(tracks[1].queue_item_id, state.dj_session_id)
    dummy.player_queues._items = [tracks[0], clip, tracks[1]]

    repaired = dummy._repair_dj_clips("queue-1", state, dummy.player_queues.items("queue-1"), -1)

    assert repaired is False
    assert dummy.player_queues.deleted == []


async def test_repair_never_treats_a_trailing_outro_as_stale(tmp_path: Path) -> None:
    """An outro clip carries no gap-next id, so repair must not delete it for lacking one."""
    tracks = [_track(index) for index in range(2)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    state = dummy._dj_queues["queue-1"]
    outro = FakeQueueItem(
        "Song Transition",
        duration=30,
        extra={ATTR_QUEUE_DJ: True, ATTR_SESSION_ID: state.dj_session_id},
    )
    dummy.player_queues._items = [*tracks, outro]

    repaired = dummy._repair_dj_clips("queue-1", state, dummy.player_queues.items("queue-1"), -1)

    assert repaired is False
    assert dummy.player_queues.deleted == []


async def test_repair_leaves_a_clip_the_player_owns_alone(tmp_path: Path) -> None:
    """A stale clip in the player's owned head stays put, only the ones beyond it are repaired."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    state = dummy._dj_queues["queue-1"]
    owned = _dj_clip("vanished-item", state.dj_session_id)
    beyond = _dj_clip("vanished-item", state.dj_session_id)
    # the guard sits on track 1, so the clip one slot past it is still the player's to take
    dummy.player_queues._items = [tracks[0], tracks[1], owned, tracks[2], beyond, tracks[3]]

    repaired = dummy._repair_dj_clips(
        "queue-1", state, dummy.player_queues.items("queue-1"), guard_index=1
    )

    assert repaired is True
    assert dummy.player_queues.deleted == [beyond.queue_item_id]
    assert owned in dummy.player_queues.items("queue-1")


async def test_splice_dj_clip_inserts_after_the_last_window_track(tmp_path: Path) -> None:
    """An end_of_playlist section splices its clip after the target, not before it."""
    tracks = [_track(index) for index in range(2)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    state = dummy._dj_queues["queue-1"]
    program = dummy._build_program({"id": "", "name": "AI DJ Rick"}, dummy._hosts["rick"])
    section = PlannedSection(
        order=0,
        clip_id="",
        section_id="Song_Transition",
        section_name="Song Transition",
        when="end_of_playlist",
        insert_at_index=len(tracks),
        prompt="",
        max_chars=200,
        web_search_mode="disabled",
    )
    items = list(dummy.player_queues.items("queue-1"))

    outcome = dummy._splice_dj_clip(
        queue_id="queue-1",
        items=items,
        guard_index=-1,
        state=state,
        program=program,
        target={"item_id": tracks[-1].queue_item_id},
        section=section,
        after_target=True,
    )

    assert outcome == "injected"
    assert items[-1].extra_attributes[ATTR_QUEUE_DJ] is True
    assert items[-1].extra_attributes.get(ATTR_GAP_NEXT_ID) is None


def _splice_gap(
    dummy: ReplanQueueDJ,
    items: list[Any],
    target_id: str,
    *,
    guard_index: int = -1,
    after_target: bool = False,
) -> str:
    """Attempt one splice of the harness's DJ program and return its outcome."""
    section = _gap_section()
    if after_target:
        section.when = "end_of_playlist"
    return dummy._splice_dj_clip(
        queue_id="queue-1",
        items=items,
        guard_index=guard_index,
        state=dummy._dj_queues["queue-1"],
        program=_dj_program(dummy),
        target={"item_id": target_id},
        section=section,
        after_target=after_target,
    )


def test_splice_returns_too_close_before_the_guard_slot(tmp_path: Path) -> None:
    """A target one slot past the guard is not spliced, and the pass spends no clip id."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    state = dummy._dj_queues["queue-1"]
    items = list(dummy.player_queues.items("queue-1"))
    before = list(items)

    outcome = _splice_gap(dummy, items, tracks[2].queue_item_id, guard_index=1)

    assert outcome == "too_close"
    assert items == before
    assert state.clip_counter == 0


def test_splice_returns_occupied_for_a_gap_that_already_has_a_clip(tmp_path: Path) -> None:
    """A gap that already holds a clip is not spliced a second time."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    state = dummy._dj_queues["queue-1"]
    existing = _dj_clip(tracks[2].queue_item_id, state.dj_session_id)
    dummy.player_queues._items = [tracks[0], tracks[1], existing, tracks[2], tracks[3]]
    items = list(dummy.player_queues.items("queue-1"))
    before = list(items)

    outcome = _splice_gap(dummy, items, tracks[2].queue_item_id)

    assert outcome == "occupied"
    assert items == before
    assert state.clip_counter == 0


def test_splice_returns_gap_gone_for_a_target_that_left(tmp_path: Path) -> None:
    """A target that is no longer in the queue has no gap left to splice into."""
    tracks = [_track(index) for index in range(3)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    state = dummy._dj_queues["queue-1"]
    items = list(dummy.player_queues.items("queue-1"))
    before = list(items)

    outcome = _splice_gap(dummy, items, "vanished-item")

    assert outcome == "gap_gone"
    assert items == before
    assert state.clip_counter == 0


def test_outro_too_close_at_the_guard_lands_nothing(tmp_path: Path) -> None:
    """An outro whose insertion point is the guard slot itself is not spliced."""
    tracks = [_track(index) for index in range(3)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    state = dummy._dj_queues["queue-1"]
    items = list(dummy.player_queues.items("queue-1"))
    before = list(items)

    outcome = _splice_gap(dummy, items, tracks[0].queue_item_id, guard_index=0, after_target=True)

    assert outcome == "too_close"
    assert items == before
    assert state.clip_counter == 0


def test_the_clip_shares_its_targets_sort_index(tmp_path: Path) -> None:
    """A clip takes its target's sort index, so un-shuffling keeps the two together."""
    tracks = [_track(index) for index in range(3)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    items = list(dummy.player_queues.items("queue-1"))

    outcome = _splice_gap(dummy, items, tracks[1].queue_item_id)

    assert outcome == "injected"
    clip = items[1]
    assert clip.queue_item_id != tracks[1].queue_item_id
    assert clip.sort_index == tracks[1].sort_index


def _outro_host() -> dict[str, Any]:
    """Return a host that always plans a section at the end of the playlist."""
    host = _must_host()
    host["section_order"] = [{"when": "end_of_playlist", "flow": [{"MUST": "Song_Transition"}]}]
    return host


async def test_replan_plans_an_outro_at_the_queue_tail(tmp_path: Path) -> None:
    """A replan plans and splices an outro after the queue's last (non-DJ) item."""
    tracks = [_track(index) for index in range(2)]
    dummy = _make_replan_dj(
        tmp_path, list(tracks), current_index=-1, index_in_buffer=-1, host=_outro_host()
    )
    state = dummy._dj_queues["queue-1"]
    state.station_id = "station_a"

    await dummy._replan_queue("queue-1")

    final_items = dummy.player_queues.items("queue-1")
    assert final_items[-1].extra_attributes[ATTR_QUEUE_DJ] is True
    assert final_items[-1].extra_attributes.get(ATTR_GAP_NEXT_ID) is None


def _show_host() -> dict[str, Any]:
    """Return a host that speaks between songs and once more at the end of the queue."""
    host = _must_host()
    host["section_order"] = [
        {"when": "between_songs", "flow": [{"MUST": "Song_Transition"}]},
        {"when": "end_of_playlist", "flow": [{"MUST": "Song_Transition"}]},
    ]
    return host


async def test_a_host_that_switches_its_outro_off_plans_none(tmp_path: Path) -> None:
    """The outro is the host's call, so a switched-off host earns no outro at the queue end."""
    host = _show_host()
    host["play_outro"] = False
    dummy = _make_replan_dj(
        tmp_path,
        [_track(index) for index in range(3)],
        current_index=-1,
        index_in_buffer=-1,
        host=host,
    )
    dummy._dj_queues["queue-1"].station_id = "station_a"

    await dummy._replan_queue("queue-1")

    assert _outro_clips(dummy) == []
    # the between-songs breaks still plan, so the pass is not simply idle
    assert dummy.player_queues.loads


async def test_a_host_that_keeps_its_outro_on_plans_one(tmp_path: Path) -> None:
    """With the switch left on, the show keeps its outro exactly as before."""
    dummy = _make_replan_dj(
        tmp_path,
        [_track(index) for index in range(3)],
        current_index=-1,
        index_in_buffer=-1,
        host=_show_host(),
    )
    dummy._dj_queues["queue-1"].station_id = "station_a"

    await dummy._replan_queue("queue-1")

    outros = _outro_clips(dummy)
    assert len(outros) == 1
    assert outros[0].extra_attributes.get(ATTR_GAP_NEXT_ID) is None


async def test_the_outro_switch_off_keeps_the_outro_off_the_bus(tmp_path: Path) -> None:
    """A term path obeys the same switch: no outro term is decided for a host without one."""
    host = _show_host()
    host["play_outro"] = False
    tracks = [_track(index) for index in range(4)]
    for track in tracks:
        track.duration = TRACK_SECONDS
    dummy = _make_replan_dj(tmp_path, tracks, host=host)
    dummy.mass.activate_program_bus("queue-1")
    dummy._dj_queues["queue-1"].station_id = "station_a"
    dummy.player_queues._queue.current_item = tracks[0]
    dummy.player_queues._queue.corrected_elapsed_time = POSITION_SECONDS

    await dummy._replan_queue("queue-1")

    assert [term for term in _terms(dummy) if term.successor_id is None] == []
    # the between-songs boundaries are still decided, so only the outro was dropped
    assert _terms(dummy)


async def test_replan_writes_no_break_around_a_track_that_cannot_play(tmp_path: Path) -> None:
    """A track that can not play earns no break before or after it, so none is paid for."""
    tracks = [_track(index) for index in range(5)]
    tracks[2].available = False
    dummy = _make_replan_dj(tmp_path, list(tracks))

    await dummy._replan_queue("queue-1")

    final_items = dummy.player_queues.items("queue-1")
    announced = {
        item.extra_attributes.get(ATTR_GAP_NEXT_ID)
        for item in final_items
        if item.extra_attributes.get(ATTR_QUEUE_DJ)
    }
    assert tracks[2].queue_item_id not in announced
    assert tracks[3].queue_item_id not in announced
    # the gap further down still gets its break, so an idle planner cannot make this pass
    assert tracks[4].queue_item_id in announced


async def test_show_bound_two_track_show_gets_exactly_one_outro(tmp_path: Path) -> None:
    """With track 1 owned by the player, the one-track window of a show still earns its outro."""
    tracks = [_track(0), _track(1)]
    dummy = _make_replan_dj(
        tmp_path, list(tracks), current_index=0, index_in_buffer=0, host=_outro_host()
    )
    state = dummy._dj_queues["queue-1"]
    state.station_id = "station_a"

    await dummy._replan_queue("queue-1")
    await dummy._replan_queue("queue-1")

    final_items = dummy.player_queues.items("queue-1")
    assert len(final_items) == 3
    assert final_items[-2].queue_item_id == tracks[1].queue_item_id
    assert final_items[-1].extra_attributes.get(ATTR_QUEUE_DJ) is True
    assert final_items[-1].extra_attributes.get(ATTR_GAP_NEXT_ID) is None


async def test_outro_not_offered_without_a_station_binding(tmp_path: Path) -> None:
    """A queue DJ with no station binding never gets an end_of_playlist outro."""
    tracks = [_track(index) for index in range(2)]
    dummy = _make_replan_dj(
        tmp_path, list(tracks), current_index=-1, index_in_buffer=-1, host=_outro_host()
    )

    await dummy._replan_queue("queue-1")

    final_items = dummy.player_queues.items("queue-1")
    assert not any(item.extra_attributes.get(ATTR_QUEUE_DJ) for item in final_items)


async def test_outro_targets_whatever_is_currently_the_queue_tail(tmp_path: Path) -> None:
    """The outro tracks the queue's last item, even one appended behind the show's tracks."""
    tracks = [_track(0), _track(1)]
    appended = _track(2)  # user appended a track behind the show
    dummy = _make_replan_dj(
        tmp_path, [*tracks, appended], current_index=-1, index_in_buffer=-1, host=_outro_host()
    )
    state = dummy._dj_queues["queue-1"]
    state.station_id = "station_a"

    await dummy._replan_queue("queue-1")

    final_items = dummy.player_queues.items("queue-1")
    assert len(final_items) == 4
    assert final_items[-1].extra_attributes.get(ATTR_QUEUE_DJ) is True
    assert final_items[-2].queue_item_id == appended.queue_item_id


async def test_no_second_outro_behind_user_appended_tracks(tmp_path: Path) -> None:
    """A song appended behind an already-spliced outro never earns a second outro."""
    tracks = [_track(0), _track(1)]
    dummy = _make_replan_dj(
        tmp_path, list(tracks), current_index=-1, index_in_buffer=-1, host=_outro_host()
    )
    state = dummy._dj_queues["queue-1"]
    state.station_id = "station_a"
    await dummy._replan_queue("queue-1")
    assert dummy.player_queues.items("queue-1")[-1].extra_attributes.get(ATTR_QUEUE_DJ) is True

    dummy.player_queues._items = [*dummy.player_queues.items("queue-1"), _track(2)]
    await dummy._replan_queue("queue-1")

    final_items = dummy.player_queues.items("queue-1")
    dj_clips = [item for item in final_items if item.extra_attributes.get(ATTR_QUEUE_DJ)]
    assert len(dj_clips) == 1


async def test_an_outro_spends_no_decided_gap(tmp_path: Path) -> None:
    """An outro announces no successor track, so it neither takes nor reopens a decided gap."""
    tracks = [_track(0), _track(1)]
    dummy = _make_replan_dj(
        tmp_path, list(tracks), current_index=-1, index_in_buffer=-1, host=_outro_host()
    )
    state = dummy._dj_queues["queue-1"]
    state.station_id = "station_a"
    state.decided_gap_ids = {tracks[1].queue_item_id}

    await dummy._replan_queue("queue-1")

    outros = _outro_clips(dummy)
    assert len(outros) == 1
    assert outros[0].extra_attributes.get(ATTR_GAP_NEXT_ID) is None
    assert state.decided_gap_ids == {tracks[1].queue_item_id}


@pytest.mark.usefixtures("kitchen_only_user")
async def test_set_queue_dj_refuses_a_queue_the_user_has_no_access_to(tmp_path: Path) -> None:
    """A member restricted to some players can not arm or clear the DJ of another queue."""
    dummy = DummyQueueDJ(tmp_path)
    dummy._arm_dj_state("living_room", "rick")

    with pytest.raises(InsufficientPermissions, match="living_room"):
        await dummy.set_queue_dj("living_room", None)
    with pytest.raises(InsufficientPermissions, match="garage"):
        await dummy.set_queue_dj("garage", "rick")

    assert set(dummy._dj_queues) == {"living_room"}
    assert not dummy._dj_file.exists()


@pytest.mark.usefixtures("kitchen_only_user")
async def test_status_only_shows_the_queues_the_user_has_access_to(tmp_path: Path) -> None:
    """A member restricted to some players does not see the DJs of the other queues."""
    dummy = DummyQueueDJ(tmp_path)
    dummy._arm_dj_state("living_room", "rick")

    expected = {"kitchen": _status_entry("rick")}
    assert await dummy.set_queue_dj("kitchen", "rick") == expected
    assert await dummy.get_queue_dj_status() == expected
    assert set(dummy._dj_queues) == {"kitchen", "living_room"}


async def test_replan_inserts_clip_between_upcoming_tracks(tmp_path: Path) -> None:
    """Inject one DJ clip into every plannable gap ahead of the playback guards."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks))

    await dummy._replan_queue("queue-1")

    queues = dummy.player_queues
    assert len(queues.loads) == 2
    for queue_items, insert_at_index in queues.loads:
        assert insert_at_index > 1
        clip = queue_items[0]
        assert clip.extra_attributes[ATTR_QUEUE_DJ] is True
        assert clip.extra_attributes[ATTR_HOST_ID] == "rick"
    announced = {items[0].extra_attributes[ATTR_GAP_NEXT_ID] for items, _ in queues.loads}
    assert announced == {tracks[2].queue_item_id, tracks[3].queue_item_id}

    final_items = queues.items("queue-1")
    for index, item in enumerate(final_items):
        if item.extra_attributes.get(ATTR_QUEUE_DJ):
            successor = final_items[index + 1]
            assert successor.queue_item_id == item.extra_attributes[ATTR_GAP_NEXT_ID]
    state = dummy._dj_queues["queue-1"]
    assert state.clip_counter == 2
    assert state.decided_gap_ids == {tracks[2].queue_item_id, tracks[3].queue_item_id}
    assert state.songs_before_window == 1


async def test_replan_skips_gaps_that_already_have_a_clip(tmp_path: Path) -> None:
    """Leave a gap alone when it already holds a DJ clip for the following track."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    state = dummy._dj_queues["queue-1"]
    existing = _dj_clip(tracks[2].queue_item_id, state.dj_session_id)
    dummy.player_queues._items = [tracks[0], tracks[1], existing, tracks[2], tracks[3]]

    await dummy._replan_queue("queue-1")

    queues = dummy.player_queues
    assert queues.deleted == []
    assert len(queues.loads) == 1
    assert queues.loads[0][0][0].extra_attributes[ATTR_GAP_NEXT_ID] == tracks[3].queue_item_id


async def test_replan_repairs_stale_clip_after_reorder(tmp_path: Path) -> None:
    """Delete a DJ clip that no longer sits in front of the track it announces."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    state = dummy._dj_queues["queue-1"]
    stale = _dj_clip("vanished-item", state.dj_session_id)
    dummy.player_queues._items = [tracks[0], tracks[1], stale, tracks[2], tracks[3]]

    await dummy._replan_queue("queue-1")

    queues = dummy.player_queues
    assert queues.deleted == [stale.queue_item_id]
    # the freed gap is plannable again in the same pass
    announced = {items[0].extra_attributes[ATTR_GAP_NEXT_ID] for items, _ in queues.loads}
    assert announced == {tracks[2].queue_item_id, tracks[3].queue_item_id}


async def test_replan_respects_buffer_guard(tmp_path: Path) -> None:
    """Never insert into the gap right after the item already loaded into the buffer."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks), current_index=0, index_in_buffer=1)

    await dummy._replan_queue("queue-1")

    queues = dummy.player_queues
    assert len(queues.loads) == 1
    queue_items, insert_at_index = queues.loads[0]
    assert insert_at_index == 3
    assert queue_items[0].extra_attributes[ATTR_GAP_NEXT_ID] == tracks[3].queue_item_id


async def test_disable_removes_all_pending_clips(tmp_path: Path) -> None:
    """Disabling drops every upcoming DJ clip whatever session it came from, played ones stay."""
    tracks = [_track(index) for index in range(3)]
    dummy = _make_replan_dj(tmp_path, [], current_index=2, index_in_buffer=2)
    state = dummy._dj_queues["queue-1"]
    played_clip = _dj_clip(tracks[1].queue_item_id, state.dj_session_id)
    pending_clip = _dj_clip(tracks[2].queue_item_id, state.dj_session_id)
    foreign_clip = _dj_clip(tracks[2].queue_item_id, "other-session")
    dummy.player_queues._items = [
        tracks[0],
        played_clip,
        tracks[1],
        pending_clip,
        tracks[2],
        foreign_clip,
    ]

    await dummy.set_queue_dj("queue-1", None)

    assert dummy._dj_queues == {}
    assert dummy.player_queues.deleted == [pending_clip.queue_item_id, foreign_clip.queue_item_id]


async def test_switch_removes_clips_of_a_session_no_state_ever_knew(tmp_path: Path) -> None:
    """Clear a clip left behind by an earlier provider run, whose session id nothing remembers."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, [])
    stale = _dj_clip(tracks[2].queue_item_id, "dj_from_a_previous_run")
    dummy.player_queues._items = [tracks[0], tracks[1], stale, tracks[2], tracks[3]]

    daisy = _must_host()
    daisy["id"] = "daisy"
    dummy._hosts["daisy"] = daisy
    await dummy.set_queue_dj("queue-1", "daisy")

    assert stale.queue_item_id in dummy.player_queues.deleted
    await asyncio.gather(*dummy.mass.tasks)


async def test_cleanup_keeps_the_freshly_armed_sessions_clips(tmp_path: Path) -> None:
    """A clip the newly armed session already injected survives the previous host's cleanup."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, [])
    old_state = dummy._dj_queues["queue-1"]
    old_clip = _dj_clip(tracks[2].queue_item_id, old_state.dj_session_id)
    dummy.player_queues._items = [tracks[0], tracks[1], old_clip, tracks[2], tracks[3]]
    write_queue_dj = dummy._write_queue_dj

    async def _write_and_inject_racing_clip() -> None:
        # stands in for a replan of the freshly armed session landing a clip in the window
        # before the cleanup of the previous host's clips got its turn
        await write_queue_dj()
        session_id = dummy._dj_queues["queue-1"].dj_session_id
        dummy.player_queues._items.insert(4, _dj_clip(tracks[3].queue_item_id, session_id))

    dummy._write_queue_dj = _write_and_inject_racing_clip  # type: ignore[method-assign]
    daisy = _must_host()
    daisy["id"] = "daisy"
    dummy._hosts["daisy"] = daisy

    await dummy.set_queue_dj("queue-1", "daisy")

    assert dummy.player_queues.deleted == [old_clip.queue_item_id]
    await asyncio.gather(*dummy.mass.tasks)


async def test_switching_host_removes_old_hosts_pending_clips(tmp_path: Path) -> None:
    """Switching the sticky DJ to a new host clears the old host's unplayed clips."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    old_state = dummy._dj_queues["queue-1"]
    await dummy._replan_queue("queue-1")
    queues = dummy.player_queues
    old_clip_ids = {
        item.queue_item_id
        for item in queues.items("queue-1")
        if item.extra_attributes.get(ATTR_QUEUE_DJ)
    }
    assert old_clip_ids  # sanity: rick's clips actually landed

    daisy = _must_host()
    daisy["id"] = "daisy"
    daisy["name"] = "Daisy"
    dummy._hosts["daisy"] = daisy

    await dummy.set_queue_dj("queue-1", "daisy")
    await asyncio.gather(*dummy.mass.tasks)

    # rick's pending clips were removed rather than left to render under his persona
    assert old_clip_ids <= set(queues.deleted)
    remaining_dj_clips = [
        item for item in queues.items("queue-1") if item.extra_attributes.get(ATTR_QUEUE_DJ)
    ]
    assert remaining_dj_clips
    for clip in remaining_dj_clips:
        assert clip.extra_attributes[ATTR_HOST_ID] == "daisy"
        assert clip.extra_attributes[ATTR_SESSION_ID] != old_state.dj_session_id


async def test_min_gap_songs_guard_holds_across_passes(tmp_path: Path) -> None:
    """Carry the section history across passes so a min_gap_songs guard keeps holding."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks), host=_optional_host(3))

    await dummy._replan_queue("queue-1")

    queues = dummy.player_queues
    assert len(queues.loads) == 1
    assert queues.loads[0][0][0].extra_attributes[ATTR_GAP_NEXT_ID] == tracks[2].queue_item_id

    extra = [_track(4), _track(5)]
    queues._items.extend(extra)
    await dummy._replan_queue("queue-1")

    # the clip announced track 2, so the next one may land no earlier than track 5
    assert len(queues.loads) == 2
    assert queues.loads[1][0][0].extra_attributes[ATTR_GAP_NEXT_ID] == extra[1].queue_item_id


async def test_planning_axis_stays_anchored_to_real_queue_positions(tmp_path: Path) -> None:
    """Growing the queue tail must not rewind the song axis the guards count on."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks))

    await dummy._replan_queue("queue-1")
    state = dummy._dj_queues["queue-1"]
    # the window opened on track 1, so track 0 is the only song behind it
    assert state.songs_before_window == 1
    assert [song for song, _minute in state.history["Song_Transition"]] == [2, 3]

    dummy.player_queues._items.extend([_track(4), _track(5)])
    await dummy._replan_queue("queue-1")

    # the window still opens on track 1, so the appended tracks keep their absolute positions
    assert state.songs_before_window == 1
    assert [song for song, _minute in state.history["Song_Transition"]] == [2, 3, 4, 5]


async def test_forward_axis_progress_leaves_history_untouched(tmp_path: Path) -> None:
    """A pass whose planning axis only moves forward must not touch the guard history."""
    tracks = [_track(index) for index in range(6)]
    dummy = _make_replan_dj(tmp_path, list(tracks), host=_optional_host(1))

    await dummy._replan_queue("queue-1")
    state = dummy._dj_queues["queue-1"]
    assert state.songs_before_window == 1
    history_before = {section_id: list(events) for section_id, events in state.history.items()}
    assert history_before  # sanity: something is actually there to protect

    dummy.player_queues._queue.current_index = 1
    dummy.player_queues._queue.index_in_buffer = 1
    await dummy._replan_queue("queue-1")

    # more songs sit behind the window now, yet no event shifted with it
    assert state.songs_before_window == 2
    assert state.history == history_before


async def test_queue_clear_and_refill_speaks_again(tmp_path: Path) -> None:
    """A queue clear followed by a refill must not leave the DJ permanently muted."""
    tracks = [_track(index) for index in range(10)]
    dummy = _make_replan_dj(
        tmp_path, list(tracks), current_index=5, index_in_buffer=5, host=_optional_host(3)
    )

    await dummy._replan_queue("queue-1")
    queues = dummy.player_queues
    assert queues.loads
    loaded_before_clear = len(queues.loads)

    # clearing the queue leaves nothing to plan against, so the pass is a no-op
    queues._items = []
    queues._queue.current_index = None
    queues._queue.index_in_buffer = None
    await dummy._replan_queue("queue-1")
    assert len(queues.loads) == loaded_before_clear

    fresh = [_track(index) for index in range(8)]
    queues._items = fresh
    queues._queue.current_index = 0
    queues._queue.index_in_buffer = 0
    await dummy._replan_queue("queue-1")

    new_loads = queues.loads[loaded_before_clear:]
    assert new_loads
    # the discarded break never aired, so nothing but the buffer guard holds the DJ back
    assert new_loads[0][0][0].extra_attributes[ATTR_GAP_NEXT_ID] == fresh[2].queue_item_id


async def test_clear_and_refill_at_the_same_position_speaks_again(tmp_path: Path) -> None:
    """A clear and refill that lands the player back on the same position must not mute the DJ."""
    tracks = [_track(index) for index in range(10)]
    dummy = _make_replan_dj(tmp_path, list(tracks), host=_optional_host(3))

    await dummy._replan_queue("queue-1")
    queues = dummy.player_queues
    assert queues.loads
    loaded_before_clear = len(queues.loads)

    queues._items = []
    queues._queue.current_index = None
    queues._queue.index_in_buffer = None
    await dummy._replan_queue("queue-1")

    # same track count behind the window as before, so the offsets cannot tell them apart
    fresh = [_track(index) for index in range(8)]
    queues._items = fresh
    queues._queue.current_index = 0
    queues._queue.index_in_buffer = 0
    await dummy._replan_queue("queue-1")

    new_loads = queues.loads[loaded_before_clear:]
    assert new_loads
    assert new_loads[0][0][0].extra_attributes[ATTR_GAP_NEXT_ID] == fresh[2].queue_item_id


async def test_a_break_that_aired_still_holds_the_dj_back_after_a_clear(tmp_path: Path) -> None:
    """A break the listener actually heard keeps its distance across a clear and refill."""
    tracks = [_track(index) for index in range(10)]
    dummy = _make_replan_dj(tmp_path, list(tracks), host=_optional_host(3))

    await dummy._replan_queue("queue-1")
    queues = dummy.player_queues
    # playback reaches the track the first break announced, so that break has aired
    clip = next(item for item in queues._items if item.extra_attributes.get(ATTR_QUEUE_DJ))
    aired_index = next(
        index
        for index, item in enumerate(queues._items)
        if item.queue_item_id == clip.extra_attributes[ATTR_GAP_NEXT_ID]
    )
    queues._queue.current_index = aired_index
    queues._queue.index_in_buffer = aired_index
    await dummy._replan_queue("queue-1")
    loaded_before_clear = len(queues.loads)

    queues._items = []
    queues._queue.current_index = None
    queues._queue.index_in_buffer = None
    await dummy._replan_queue("queue-1")

    fresh = [_track(index) for index in range(10)]
    queues._items = fresh
    queues._queue.current_index = 0
    queues._queue.index_in_buffer = 0
    await dummy._replan_queue("queue-1")

    new_loads = queues.loads[loaded_before_clear:]
    assert new_loads
    # the aired break moved with the queue, so its guard still applies from the new start:
    # it sat one song behind the window and keeps that distance on the fresh queue
    assert new_loads[0][0][0].extra_attributes[ATTR_GAP_NEXT_ID] == fresh[3].queue_item_id


async def test_a_pending_break_at_the_buffer_edge_holds_nothing_after_a_clear(
    tmp_path: Path,
) -> None:
    """A clear that discards a break sitting right at the buffer edge frees its budget too."""
    tracks = [_track(index) for index in range(10)]
    dummy = _make_replan_dj(tmp_path, list(tracks), host=_optional_host(3))

    await dummy._replan_queue("queue-1")
    queues = dummy.player_queues
    # playback advances to the track right before the first staged clip, so the clip sits
    # one slot past the buffer and its event lands exactly on the recorded window start
    queues._queue.current_index = 1
    queues._queue.index_in_buffer = 1
    await dummy._replan_queue("queue-1")
    state = dummy._dj_queues["queue-1"]
    assert state.songs_before_window == 2
    loaded_before_clear = len(queues.loads)

    queues._items = []
    queues._queue.current_index = None
    queues._queue.index_in_buffer = None
    await dummy._replan_queue("queue-1")

    fresh = [_track(index) for index in range(8)]
    queues._items = fresh
    queues._queue.current_index = 0
    queues._queue.index_in_buffer = 0
    await dummy._replan_queue("queue-1")

    new_loads = queues.loads[loaded_before_clear:]
    assert new_loads
    # the discarded clip never aired, so nothing holds the DJ past the first allowed gap
    assert new_loads[0][0][0].extra_attributes[ATTR_GAP_NEXT_ID] == fresh[2].queue_item_id


async def test_an_airing_break_still_holds_the_dj_back_when_the_tail_is_swapped(
    tmp_path: Path,
) -> None:
    """A break the player owns keeps its guard weight when every upcoming track is swapped."""
    tracks = [_track(index) for index in range(10)]
    dummy = _make_replan_dj(tmp_path, list(tracks), host=_optional_host(3))

    await dummy._replan_queue("queue-1")
    queues = dummy.player_queues
    # the first staged clip reaches the player and starts airing
    clip_index = next(
        index
        for index, item in enumerate(queues._items)
        if item.extra_attributes.get(ATTR_QUEUE_DJ)
    )
    queues._queue.current_index = clip_index
    queues._queue.index_in_buffer = clip_index
    await dummy._replan_queue("queue-1")
    state = dummy._dj_queues["queue-1"]
    assert state.songs_before_window == 2
    loaded_before_swap = len(queues.loads)

    # everything behind the airing clip is swapped out, so no decided track is left
    head = queues._items[: clip_index + 1]
    fresh = [_track(20 + index) for index in range(8)]
    queues._items = [*head, *fresh]
    await dummy._replan_queue("queue-1")

    new_loads = queues.loads[loaded_before_swap:]
    assert new_loads
    # the listener is hearing that break right now, so the guard clears three songs later
    assert new_loads[0][0][0].extra_attributes[ATTR_GAP_NEXT_ID] == fresh[3].queue_item_id


async def test_jump_to_top_speaks_in_head_gaps_without_redeciding_the_tail(
    tmp_path: Path,
) -> None:
    """Jumping playback back to the top opens the head gaps without re-deciding the tail."""
    tracks = [_track(index) for index in range(10)]
    dummy = _make_replan_dj(
        tmp_path, list(tracks), current_index=5, index_in_buffer=5, host=_optional_host(3)
    )

    await dummy._replan_queue("queue-1")
    queues = dummy.player_queues
    assert queues.loads
    loaded_before_jump = len(queues.loads)
    tail_gap_ids = {
        item.extra_attributes[ATTR_GAP_NEXT_ID]
        for item in queues._items
        if item.extra_attributes.get(ATTR_QUEUE_DJ)
    }

    queues._queue.current_index = 0
    queues._queue.index_in_buffer = 0
    await dummy._replan_queue("queue-1")

    new_loads = queues.loads[loaded_before_jump:]
    assert new_loads
    new_gap_ids = {load[0][0].extra_attributes[ATTR_GAP_NEXT_ID] for load in new_loads}
    # the history is capped at the head's first gap, so the guard clears three songs later
    assert new_gap_ids == {tracks[4].queue_item_id}
    assert new_gap_ids.isdisjoint(tail_gap_ids)


async def test_stepping_back_a_track_keeps_staged_breaks_apart(tmp_path: Path) -> None:
    """A break staged further down the queue still spaces the ones planned behind it."""
    tracks = [_track(index) for index in range(12)]
    dummy = _make_replan_dj(
        tmp_path, list(tracks), current_index=5, index_in_buffer=5, host=_optional_host(3)
    )

    await dummy._replan_queue("queue-1")
    queues = dummy.player_queues

    # the listener steps back a track, which reopens the gaps behind the staged clips
    queues._queue.current_index = 4
    queues._queue.index_in_buffer = 4
    await dummy._replan_queue("queue-1")

    clip_positions = [
        index
        for index, item in enumerate(queues._items)
        if item.extra_attributes.get(ATTR_QUEUE_DJ)
    ]
    for before, after in pairwise(clip_positions):
        tracks_between = sum(
            1
            for item in queues._items[before + 1 : after]
            if not item.extra_attributes.get(ATTR_QUEUE_DJ)
        )
        assert tracks_between >= 3


async def test_a_probed_duration_alone_does_not_move_the_history(tmp_path: Path) -> None:
    """A track duration correcting downwards is not a queue that rewound."""
    tracks = [_track(index) for index in range(10)]
    dummy = _make_replan_dj(tmp_path, list(tracks), current_index=4, index_in_buffer=4)

    await dummy._replan_queue("queue-1")
    state = dummy._dj_queues["queue-1"]
    history_before = {section_id: list(events) for section_id, events in state.history.items()}
    minutes_before = state.minutes_before_window
    assert history_before

    # a shorter probed duration replaces the estimate, so only the minute count drops
    tracks[0].duration = 30
    await dummy._replan_queue("queue-1")

    assert state.minutes_before_window < minutes_before
    assert state.songs_before_window == 5
    assert state.history == history_before


async def test_hourly_host_speaks_again_after_a_clear_and_refill(tmp_path: Path) -> None:
    """A max_per_60min guard must not keep muting the DJ off the old queue's high-water mark."""
    old_items = [_track(index) for index in range(24)]
    dummy = _make_replan_dj(
        tmp_path, list(old_items), current_index=19, index_in_buffer=19, host=_hourly_host()
    )
    for section in _hourly_sections():
        dummy._sections[section["id"]] = section

    await dummy._replan_queue("queue-1")
    queues = dummy.player_queues
    assert queues.loads
    loaded_before_clear = len(queues.loads)

    queues._items = []
    queues._queue.current_index = None
    queues._queue.index_in_buffer = None
    await dummy._replan_queue("queue-1")
    assert len(queues.loads) == loaded_before_clear

    fresh = [_track(index) for index in range(6)]
    queues._items = fresh
    queues._queue.current_index = 0
    queues._queue.index_in_buffer = 0
    await dummy._replan_queue("queue-1")

    new_loads = queues.loads[loaded_before_clear:]
    assert new_loads
    # the discarded break never aired, so it spends none of the once-per-hour budget
    assert new_loads[0][0][0].extra_attributes[ATTR_GAP_NEXT_ID] == fresh[2].queue_item_id


async def test_max_per_60min_holds_within_one_queue(tmp_path: Path) -> None:
    """A once-per-hour guard allows one break in the hour, leaving the later gaps empty."""
    tracks = [_track(index) for index in range(6)]
    dummy = _make_replan_dj(tmp_path, list(tracks), host=_hourly_host())
    for section in _hourly_sections():
        dummy._sections[section["id"]] = section

    await dummy._replan_queue("queue-1")

    # the first gap airs the weather and news, the four behind it stay empty this hour
    assert len(dummy.player_queues.loads) == 1
    assert _announced_gaps(dummy) == {tracks[2].queue_item_id}
    assert dummy._dj_queues["queue-1"].decided_gap_ids == {
        track.queue_item_id for track in tracks[2:]
    }


async def test_replacing_the_upcoming_tracks_reanchors_the_guard(tmp_path: Path) -> None:
    """Swapping every upcoming track leaves the DJ speaking at its normal gap, not silent."""
    tracks_v1 = [_track(index) for index in range(6)]
    dummy = _make_replan_dj(
        tmp_path, list(tracks_v1), current_index=0, index_in_buffer=0, host=_optional_host(2)
    )

    await dummy._replan_queue("queue-1")
    queues = dummy.player_queues
    assert queues.loads
    loaded_after_pass1 = len(queues.loads)

    # the playing track stays, so the axis holds still and only the history's tracks go
    new_tail = [_track(20 + index) for index in range(5)]
    queues._items = [tracks_v1[0], *new_tail]
    await dummy._replan_queue("queue-1")

    new_loads = queues.loads[loaded_after_pass1:]
    assert new_loads
    assert new_loads[0][0][0].extra_attributes[ATTR_GAP_NEXT_ID] == new_tail[1].queue_item_id


async def test_scheduled_replan_serves_requests_landing_during_a_pass(tmp_path: Path) -> None:
    """A replan request raised by the pass's own inserts is served and then converges."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    queues = dummy.player_queues
    queues.on_items_updated = lambda: dummy._schedule_replan("queue-1")

    dummy._schedule_replan("queue-1")
    await asyncio.gather(*dummy.mass.tasks)

    assert len(queues.loads) == 2
    assert dummy._dj_queues["queue-1"].replan_pending is False


async def test_failing_pass_leaves_the_dj_schedulable(tmp_path: Path) -> None:
    """A raising pass clears its request, does not retry, and recovers on a later event."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    working_build_program = dummy._build_program
    attempts: list[str] = []

    def _failing_build_program(_station: dict[str, Any], host: dict[str, Any]) -> dict[str, Any]:
        attempts.append(host["id"])
        # an event landing mid-pass re-requests a replan behind the still running task
        dummy._schedule_replan("queue-1")
        raise MusicAssistantError("misconfigured host")

    dummy._build_program = _failing_build_program  # type: ignore[method-assign, assignment]
    dummy._schedule_replan("queue-1")
    await asyncio.gather(*dummy.mass.tasks)

    assert attempts == ["rick"]
    assert dummy.player_queues.loads == []
    assert dummy._dj_queues["queue-1"].replan_pending is False

    dummy._build_program = working_build_program  # type: ignore[method-assign]
    dummy._schedule_replan("queue-1")
    await asyncio.gather(*dummy.mass.tasks)

    assert len(dummy.player_queues.loads) == 2


async def test_failing_pass_clears_the_state_the_queue_was_rearmed_with(tmp_path: Path) -> None:
    """A re-arm during a failing pass must leave the new state schedulable, not the old one."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    old_state = dummy._dj_queues["queue-1"]
    working_build_program = dummy._build_program

    def _failing_build_program(_station: dict[str, Any], host: dict[str, Any]) -> dict[str, Any]:
        # set_queue_dj re-arms the queue mid-pass, marks the fresh state plannable once its
        # clip cleanup ran, and latches its request onto the still running task
        dummy._arm_dj_state("queue-1", host["id"]).ready = True
        dummy._schedule_replan("queue-1")
        raise MusicAssistantError("misconfigured host")

    dummy._build_program = _failing_build_program  # type: ignore[method-assign, assignment]
    dummy._schedule_replan("queue-1")
    await asyncio.gather(*dummy.mass.tasks)

    new_state = dummy._dj_queues["queue-1"]
    assert new_state is not old_state
    assert new_state.replan_pending is False

    dummy._build_program = working_build_program  # type: ignore[method-assign]
    dummy._schedule_replan("queue-1")
    await asyncio.gather(*dummy.mass.tasks)

    assert len(dummy.player_queues.loads) == 2


async def test_unloading_provider_schedules_no_replans(tmp_path: Path) -> None:
    """An unloading provider starts no new replan work."""
    dummy = _make_replan_dj(tmp_path, [_track(index) for index in range(4)])
    dummy._unloading = True

    dummy._schedule_replan("queue-1")

    assert dummy.mass.tasks == []
    state = dummy._dj_queues["queue-1"]
    assert state.replan_pending is False
    assert state.task is None


async def test_injection_rereads_a_guard_that_moved_during_the_pass(tmp_path: Path) -> None:
    """A guard that advanced while the pass awaited is honoured by the injections."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks), current_index=0, index_in_buffer=0)
    prepare_runtime_tokens = dummy._prepare_runtime_tokens

    async def _slow_prepare(program: dict[str, Any]) -> dict[str, str]:
        # stands in for a slow token source: the player buffers ahead while we wait
        dummy.player_queues._queue.index_in_buffer = 1
        return await prepare_runtime_tokens(program)

    dummy._prepare_runtime_tokens = _slow_prepare  # type: ignore[method-assign]
    await dummy._replan_queue("queue-1")

    queues = dummy.player_queues
    assert len(queues.loads) == 1
    assert queues.loads[0][0][0].extra_attributes[ATTR_GAP_NEXT_ID] == tracks[3].queue_item_id


async def test_stale_pass_inserts_nothing_after_a_mid_pass_dj_switch(tmp_path: Path) -> None:
    """A pass that resumes after set_queue_dj replaced its state must not insert clips."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    old_state = dummy._dj_queues["queue-1"]
    prepare_runtime_tokens = dummy._prepare_runtime_tokens

    daisy = _must_host()
    daisy["id"] = "daisy"
    dummy._hosts["daisy"] = daisy

    async def _switch_mid_pass(program: dict[str, Any]) -> dict[str, str]:
        # stands in for set_queue_dj swapping this queue to another host while the
        # in-flight pass is still awaiting a slow runtime token fetch
        dummy._arm_dj_state("queue-1", "daisy")
        return await prepare_runtime_tokens(program)

    dummy._prepare_runtime_tokens = _switch_mid_pass  # type: ignore[method-assign]
    await dummy._replan_queue("queue-1")

    assert dummy.player_queues.loads == []
    new_state = dummy._dj_queues["queue-1"]
    assert new_state is not old_state
    assert new_state.decided_gap_ids == set()


async def test_replan_holds_off_until_the_switch_cleanup_ran(tmp_path: Path) -> None:
    """A pass entering between arming a host and clearing the old clips must not plan."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    # stands in for set_queue_dj arming the new host while a drain task is already awake
    state = dummy._arm_dj_state("queue-1", "rick")
    state.replan_pending = True

    await dummy._replan_queue("queue-1")

    assert dummy.player_queues.loads == []
    assert state.decided_gap_ids == set()
    # the request is dropped, not kept, or the drain loop would spin on the bail
    assert state.replan_pending is False


async def test_switch_refills_the_gaps_its_own_cleanup_frees(tmp_path: Path) -> None:
    """A replan racing a host switch must not leave the gaps its cleanup frees unplanned."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    await dummy._replan_queue("queue-1")
    queues = dummy.player_queues
    old_clip_ids = {
        item.queue_item_id
        for item in queues.items("queue-1")
        if item.extra_attributes.get(ATTR_QUEUE_DJ)
    }
    assert len(old_clip_ids) == 2  # sanity: rick filled both gaps

    daisy = _must_host()
    daisy["id"] = "daisy"
    dummy._hosts["daisy"] = daisy
    write_queue_dj = dummy._write_queue_dj

    async def _write_and_replan() -> None:
        # a queue event wakes the drain task while the switch sits between arming the new
        # state and clearing the previous host's clips
        await write_queue_dj()
        await dummy._replan_queue("queue-1")

    dummy._write_queue_dj = _write_and_replan  # type: ignore[method-assign]
    await dummy.set_queue_dj("queue-1", "daisy")
    await asyncio.gather(*dummy.mass.tasks)

    assert old_clip_ids <= set(queues.deleted)
    remaining = [
        item for item in queues.items("queue-1") if item.extra_attributes.get(ATTR_QUEUE_DJ)
    ]
    assert len(remaining) == 2
    for clip in remaining:
        assert clip.extra_attributes[ATTR_HOST_ID] == "daisy"


async def test_a_failing_switch_cleanup_leaves_no_armed_state(tmp_path: Path) -> None:
    """A switch whose cleanup blows up must not leave behind a state that can never plan."""
    dummy = _make_replan_dj(tmp_path, [_track(index) for index in range(3)])
    daisy = _must_host()
    daisy["id"] = "daisy"
    dummy._hosts["daisy"] = daisy

    def _failing_cleanup(queue_id: str) -> None:  # noqa: ARG001
        raise MusicAssistantError("queue layer is unhappy")

    dummy._remove_pending_dj_clips = _failing_cleanup  # type: ignore[method-assign]

    with pytest.raises(MusicAssistantError):
        await dummy.set_queue_dj("queue-1", "daisy")

    assert dummy._dj_queues == {}
    assert dummy.mass.tasks == []


async def test_a_rejected_clip_keeps_no_guard_history(tmp_path: Path) -> None:
    """A merged clip the splice rejects must not block its own sections from airing later."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks), host=_hourly_host())
    for section in _hourly_sections():
        dummy._sections[section["id"]] = section
    prepare_runtime_tokens = dummy._prepare_runtime_tokens

    async def _slow_prepare(program: dict[str, Any]) -> dict[str, str]:
        # the player buffers ahead while the pass awaits, so the planned gap is spoken for
        dummy.player_queues._queue.index_in_buffer = 1
        return await prepare_runtime_tokens(program)

    dummy._prepare_runtime_tokens = _slow_prepare  # type: ignore[method-assign]
    await dummy._replan_queue("queue-1")

    state = dummy._dj_queues["queue-1"]
    assert dummy.player_queues.loads == []
    assert state.history["Weather"] == []
    assert state.history["News"] == []

    # the queue grows, so both sections get a fresh gap to air in
    dummy._prepare_runtime_tokens = prepare_runtime_tokens  # type: ignore[method-assign]
    extra = [_track(4), _track(5)]
    dummy.player_queues._items.extend(extra)
    await dummy._replan_queue("queue-1")

    assert len(dummy.player_queues.loads) == 1
    clip = dummy.player_queues.loads[0][0][0]
    assert clip.name == "Weather + News"
    assert clip.extra_attributes[ATTR_GAP_NEXT_ID] == extra[0].queue_item_id


async def test_a_pass_applies_all_its_clips_in_one_queue_update(tmp_path: Path) -> None:
    """A whole window of clips reaches the clients as one update, not one per clip."""
    tracks = [_track(index) for index in range(6)]
    dummy = _make_replan_dj(tmp_path, list(tracks))

    await dummy._replan_queue("queue-1")

    queues = dummy.player_queues
    assert len(queues.loads) == 4  # one clip per plannable gap
    assert queues.update_calls == 1
    for clip_items, _index in queues.loads:
        clip = clip_items[0]
        target = next(
            item
            for item in queues.items("queue-1")
            if item.queue_item_id == clip.extra_attributes[ATTR_GAP_NEXT_ID]
        )
        # sharing the target's sort index keeps the clip next to it when un-shuffling
        assert clip.sort_index == target.sort_index


async def test_a_switch_clears_and_refills_in_one_update_per_phase(tmp_path: Path) -> None:
    """The cleanup and the refill of a switch each replace the queue exactly once."""
    tracks = [_track(index) for index in range(6)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    await dummy._replan_queue("queue-1")
    queues = dummy.player_queues
    assert queues.update_calls == 1
    daisy = _must_host()
    daisy["id"] = "daisy"
    dummy._hosts["daisy"] = daisy

    await dummy.set_queue_dj("queue-1", "daisy")

    assert queues.update_calls == 2
    assert len(queues.deleted) == 4

    await asyncio.gather(*dummy.mass.tasks)

    assert queues.update_calls == 3
    assert len(queues.loads) == 8


async def test_a_vanished_target_leaves_its_gap_open(tmp_path: Path) -> None:
    """A gap whose target left the queue mid-pass stays open, so its return is planned."""
    tracks = [_track(index) for index in range(6)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    prepare_runtime_tokens = dummy._prepare_runtime_tokens

    async def _slow_prepare(program: dict[str, Any]) -> dict[str, str]:
        # stands in for a slow token source: the planned tracks are already fixed when the
        # user removes one of them from the queue
        dummy.player_queues.update_items(
            "queue-1",
            [item for item in dummy.player_queues.items("queue-1") if item is not tracks[3]],
        )
        return await prepare_runtime_tokens(program)

    dummy._prepare_runtime_tokens = _slow_prepare  # type: ignore[method-assign]
    await dummy._replan_queue("queue-1")

    queues = dummy.player_queues
    announced = {items[0].extra_attributes[ATTR_GAP_NEXT_ID] for items, _ in queues.loads}
    assert announced == {
        tracks[2].queue_item_id,
        tracks[4].queue_item_id,
        tracks[5].queue_item_id,
    }
    state = dummy._dj_queues["queue-1"]
    assert state.decided_gap_ids == announced

    # the track returns, as a reorder that rebuilds the queue does
    dummy._prepare_runtime_tokens = prepare_runtime_tokens  # type: ignore[method-assign]
    queues._items.append(tracks[3])
    await dummy._replan_queue("queue-1")

    assert queues.loads[-1][0][0].extra_attributes[ATTR_GAP_NEXT_ID] == tracks[3].queue_item_id
    assert tracks[3].queue_item_id in state.decided_gap_ids


async def test_decided_gaps_are_forgotten_when_their_track_leaves_the_queue(
    tmp_path: Path,
) -> None:
    """A track leaving the queue drops its decision, so a later return is planned afresh."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    queues = dummy.player_queues
    await dummy._replan_queue("queue-1")
    state = dummy._dj_queues["queue-1"]
    assert state.decided_gap_ids == {tracks[2].queue_item_id, tracks[3].queue_item_id}

    # the user removes the last track together with the clip that announced it
    queues._items = [
        item
        for item in queues.items("queue-1")
        if item is not tracks[3]
        and item.extra_attributes.get(ATTR_GAP_NEXT_ID) != tracks[3].queue_item_id
    ]
    await dummy._replan_queue("queue-1")

    assert state.decided_gap_ids == {tracks[2].queue_item_id}

    # the track comes back, as a reorder or a re-add rebuilds the queue
    queues._items.append(tracks[3])
    await dummy._replan_queue("queue-1")

    assert queues.loads[-1][0][0].extra_attributes[ATTR_GAP_NEXT_ID] == tracks[3].queue_item_id
    assert state.decided_gap_ids == {tracks[2].queue_item_id, tracks[3].queue_item_id}


async def test_reordering_the_queue_keeps_the_moved_gaps_plannable(tmp_path: Path) -> None:
    """Shuffling new tracks into the upcoming items must not hide the moved gaps for good."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    await dummy._replan_queue("queue-1")
    queues = dummy.player_queues
    assert len(queues.loads) == 2
    clips = {
        item.extra_attributes[ATTR_GAP_NEXT_ID]: item
        for item in queues.items("queue-1")
        if item.extra_attributes.get(ATTR_QUEUE_DJ)
    }

    # shuffle on add: the new tracks land in front of the old ones, moving the previously
    # last planned track to the very back. the clips travelled along with their own track
    extra = [_track(4), _track(5)]
    queues._items = [
        tracks[0],
        extra[0],
        tracks[1],
        extra[1],
        clips[tracks[2].queue_item_id],
        tracks[2],
        clips[tracks[3].queue_item_id],
        tracks[3],
    ]
    await dummy._replan_queue("queue-1")

    assert queues.deleted == []
    announced = {items[0].extra_attributes[ATTR_GAP_NEXT_ID] for items, _ in queues.loads[2:]}
    assert announced == {tracks[1].queue_item_id, extra[1].queue_item_id}


async def test_appending_a_batch_plans_only_the_new_gaps(tmp_path: Path) -> None:
    """A batch appended to a progressive queue is planned without re-rolling settled gaps."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    await dummy._replan_queue("queue-1")
    queues = dummy.player_queues
    state = dummy._dj_queues["queue-1"]

    extra = [_track(4), _track(5)]
    queues._items.extend(extra)
    await dummy._replan_queue("queue-1")

    assert len(queues.loads) == 4
    announced = {items[0].extra_attributes[ATTR_GAP_NEXT_ID] for items, _ in queues.loads[2:]}
    assert announced == {extra[0].queue_item_id, extra[1].queue_item_id}
    # the settled gaps never reached the planner, so their events are registered once
    assert [song for song, _minute in state.history["Song_Transition"]] == [2, 3, 4, 5]


async def test_a_repaired_clip_reopens_its_gap(tmp_path: Path) -> None:
    """A clip the repair drops leaves its gap open, so a later pass fills it again."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    await dummy._replan_queue("queue-1")
    queues = dummy.player_queues
    stale = next(
        item
        for item in queues.items("queue-1")
        if item.extra_attributes.get(ATTR_GAP_NEXT_ID) == tracks[3].queue_item_id
    )

    # a queue edit moved the clip away from the track it announces
    queues._items = [item for item in queues._items if item is not stale] + [stale]
    await dummy._replan_queue("queue-1")

    assert queues.deleted == [stale.queue_item_id]
    assert queues.loads[-1][0][0].extra_attributes[ATTR_GAP_NEXT_ID] == tracks[3].queue_item_id


async def test_replan_injects_nothing_while_the_dj_is_paused(tmp_path: Path) -> None:
    """A DJ in its render cool-down plans no breaks, and plans again once it is over."""
    dummy = _make_replan_dj(tmp_path, [_track(index) for index in range(4)])
    state = dummy._dj_queues["queue-1"]
    loop = asyncio.get_running_loop()
    state.render_health.tts.cooldown_until = loop.time() + 100

    await dummy._replan_queue("queue-1")
    assert dummy.player_queues.loads == []

    state.render_health.tts.cooldown_until = loop.time() - 1
    await dummy._replan_queue("queue-1")
    assert len(dummy.player_queues.loads) == 2


async def test_pausing_the_dj_drops_its_pending_clips_and_books_a_replan(tmp_path: Path) -> None:
    """A paused DJ takes its unaired breaks out, reopens their gaps and books the next pass."""
    tracks = [_track(index) for index in range(6)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    await dummy._replan_queue("queue-1")
    queues = dummy.player_queues
    loading, *pending = [item for item in queues._items if item.extra_attributes.get(ATTR_QUEUE_DJ)]
    assert len(pending) == 3
    # the player moves on to the track before the first break: that break sits right after
    # the guard, in the slot the queue is loading when its render fails
    queues._queue.current_index = 1
    queues._queue.index_in_buffer = 1
    await dummy._replan_queue("queue-1")
    state = dummy._dj_queues["queue-1"]

    dummy._pause_queue_dj("queue-1", loading.queue_item_id, TTS_COOLDOWN_SECONDS)

    assert queues.deleted == [clip.queue_item_id for clip in pending]
    assert queues._items[2] is loading
    assert state.decided_gap_ids == {loading.extra_attributes[ATTR_GAP_NEXT_ID]}
    # none of the breaks aired, so none keeps its guard weight
    assert state.history["Song_Transition"] == []
    assert dummy.mass.timers == [
        (
            TTS_COOLDOWN_SECONDS,
            dummy._schedule_replan,
            ("queue-1",),
            "ai_radio_dj_cooldown_queue-1",
        )
    ]


async def test_pausing_keeps_a_failed_clip_that_sits_past_the_guard_slot(tmp_path: Path) -> None:
    """A clip the queue reached behind an unplayable track stays, so the queue's walk holds."""
    tracks = [_track(index) for index in range(6)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    await dummy._replan_queue("queue-1")
    queues = dummy.player_queues
    airing, failed, *pending = [
        item for item in queues._items if item.extra_attributes.get(ATTR_QUEUE_DJ)
    ]
    # the first break is on air and the track behind it could not be loaded, so the queue
    # walked on to the second break, whose render is the failure that pauses the DJ
    queues._queue.current_index = queues._items.index(airing)
    queues._queue.index_in_buffer = queues._queue.current_index
    await dummy._replan_queue("queue-1")
    state = dummy._dj_queues["queue-1"]

    dummy._pause_queue_dj("queue-1", failed.queue_item_id, TTS_COOLDOWN_SECONDS)

    assert queues.deleted == [clip.queue_item_id for clip in pending]
    assert failed in queues._items
    assert state.decided_gap_ids == {
        airing.extra_attributes[ATTR_GAP_NEXT_ID],
        failed.extra_attributes[ATTR_GAP_NEXT_ID],
    }


async def test_pausing_forgets_the_failed_breaks_history_without_later_clips(
    tmp_path: Path,
) -> None:
    """A failed break with nothing pending behind it still loses its guard weight."""
    tracks = [_track(index) for index in range(5)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    await dummy._replan_queue("queue-1")
    queues = dummy.player_queues
    *_aired, failed = [item for item in queues._items if item.extra_attributes.get(ATTR_QUEUE_DJ)]
    state = dummy._dj_queues["queue-1"]
    failed_song = max(song for song, _minute in state.history["Song_Transition"])
    # the player is on the track right before the last break, which is the one that fails
    queues._queue.current_index = queues._items.index(failed) - 1
    queues._queue.index_in_buffer = queues._queue.current_index

    dummy._pause_queue_dj("queue-1", failed.queue_item_id, TTS_COOLDOWN_SECONDS)

    assert queues.deleted == []
    assert failed_song not in [song for song, _minute in state.history["Song_Transition"]]


async def test_a_pass_in_flight_when_the_pause_starts_injects_nothing(tmp_path: Path) -> None:
    """A replan suspended on its token fetch while the DJ pauses puts no breaks back."""
    dummy = _make_replan_dj(tmp_path, [_track(index) for index in range(4)])
    state = dummy._dj_queues["queue-1"]
    prepare_tokens = dummy._prepare_runtime_tokens

    async def _pause_mid_pass(program: dict[str, Any]) -> dict[str, str]:
        state.render_health.tts.cooldown_until = asyncio.get_running_loop().time() + 100
        return await prepare_tokens(program)

    dummy._prepare_runtime_tokens = _pause_mid_pass  # type: ignore[method-assign]

    await dummy._replan_queue("queue-1")

    assert dummy.player_queues.loads == []


async def test_replan_keeps_state_when_the_queue_is_not_registered_yet(tmp_path: Path) -> None:
    """A queue that has not registered yet is not treated as a vanished queue."""
    dummy = _make_replan_dj(tmp_path, [_track(index) for index in range(4)])
    # the boot-time replan runs before on_player_register created the queue
    dummy.player_queues._queue = FakeQueue("some-other-queue", None, None)
    dummy._dj_file.write_text("untouched")

    await dummy._replan_queue("queue-1")

    assert "queue-1" in dummy._dj_queues
    assert dummy._dj_file.read_text() == "untouched"


async def test_queue_added_event_replans_an_armed_queue(tmp_path: Path) -> None:
    """A queue registering after the provider loaded resumes injection on its own."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, list(tracks))
    dummy.player_queues._queue = FakeQueue("some-other-queue", None, None)
    await dummy._replan_queue("queue-1")
    assert dummy.player_queues.loads == []

    dummy.player_queues._queue = FakeQueue("queue-1", 0, 0)
    await dummy._on_dj_queue_event(
        cast("Any", SimpleNamespace(event=EventType.QUEUE_ADDED, object_id="queue-1"))
    )
    await asyncio.gather(*dummy.mass.tasks)

    assert len(dummy.player_queues.loads) == 2


async def test_dj_clip_is_sound_effect_media_type(tmp_path: Path) -> None:
    """A DJ clip's media item is a SoundEffect, the type the dynamic pool exclusions key on."""
    dummy = _make_replan_dj(tmp_path, [])
    section = PlannedSection(
        order=0,
        clip_id="dj_clip_1",
        section_id="Song_Transition",
        section_name="Song Transition",
        when="between_songs",
        insert_at_index=0,
        prompt="",
        max_chars=200,
        web_search_mode="disabled",
    )
    clip = dummy._section_to_clip_item("queue-1", "sess", {"id": "", "host_id": "rick"}, section)
    assert clip.media_item is not None
    assert clip.media_item.media_type == MediaType.SOUND_EFFECT


async def test_set_queue_dj_emits_a_queue_dj_hint_on_enable_and_disable(tmp_path: Path) -> None:
    """Arming and disarming a queue DJ each announce one refetch hint."""
    dummy = DummyQueueDJ(tmp_path)

    await dummy.set_queue_dj("queue-1", "rick")
    await dummy.set_queue_dj("queue-1", None)

    assert dummy.provider_events == [
        {"event": EVENT_QUEUE_DJ_UPDATED},
        {"event": EVENT_QUEUE_DJ_UPDATED},
    ]


async def test_set_queue_dj_announces_the_rollback(tmp_path: Path) -> None:
    """A failed arm rolls the state back, and clients are told to refetch that too."""

    class FailingCleanupQueueDJ(DummyQueueDJ):
        def _remove_pending_dj_clips(self, queue_id: str) -> None:
            """Fail the clip cleanup that follows the persisted arm."""
            raise RuntimeError("cleanup failed")

    dummy = FailingCleanupQueueDJ(tmp_path)

    with pytest.raises(RuntimeError):
        await dummy.set_queue_dj("queue-1", "rick")

    assert "queue-1" not in dummy._dj_queues
    # one hint for the persisted arm, one for the rollback that undid it in memory
    assert dummy.provider_events == [
        {"event": EVENT_QUEUE_DJ_UPDATED},
        {"event": EVENT_QUEUE_DJ_UPDATED},
    ]


async def test_player_removed_emits_a_queue_dj_hint(tmp_path: Path) -> None:
    """Dropping the DJ of a removed player announces a refetch hint."""
    dummy = DummyQueueDJ(tmp_path)
    await dummy.set_queue_dj("queue-1", "rick")
    dummy.provider_events.clear()

    await dummy._on_dj_queue_event(_player_removed("queue-1"))

    assert "queue-1" not in dummy._dj_queues
    # the queue's claim on mixed boundaries is released with its DJ state
    assert dummy.mass.boundary_mix_calls == [("queue-1", False)]
    assert dummy.provider_events == [{"event": EVENT_QUEUE_DJ_UPDATED}]

    fresh = DummyQueueDJ(tmp_path)
    await fresh._load_queue_dj()
    assert "queue-1" not in fresh._dj_queues


async def test_a_replan_that_finds_its_host_gone_releases_the_boundary_claim(
    tmp_path: Path,
) -> None:
    """A host deleted while its state is armed drops the state and its claim."""
    dj = _dj_harness(tmp_path)
    _fake_queue_with_sources(dj, "q1", [])
    dj.mass.player_queues._items = [_track(1), _track(2)]
    state = dj._arm_dj_state("q1", "amy")
    state.ready = True
    del dj._hosts["amy"]

    await dj._replan_queue("q1")

    assert "q1" not in dj._dj_queues
    assert dj.mass.boundary_mix_calls[-1] == ("q1", False)


def test_a_section_edit_rearms_the_boundary_claim(tmp_path: Path) -> None:
    """A section that opts into pre or post flips the claim of the queues its host armed."""
    dj = _dj_harness(tmp_path)
    dj._sections["talk"] = {"id": "talk"}
    dj._hosts["amy"]["section_ids"] = ["talk"]
    dj._arm_dj_state("q1", "amy")

    dj._refresh_stream_claims()
    assert dj.mass.boundary_mix_calls[-1] == ("q1", False)
    assert dj.mass.lead_hold_calls[-1] == ("q1", False)

    dj._sections["talk"]["allow_pre"] = True
    dj._refresh_stream_claims()
    assert dj.mass.boundary_mix_calls[-1] == ("q1", True)
    # only a pre lets a break start over the outgoing tail, so only it holds the tails
    assert dj.mass.lead_hold_calls[-1] == ("q1", True)

    dj._sections["talk"]["allow_pre"] = False
    dj._sections["talk"]["allow_post"] = True
    dj._refresh_stream_claims()
    assert dj.mass.boundary_mix_calls[-1] == ("q1", True)
    assert dj.mass.lead_hold_calls[-1] == ("q1", False)


def _terms(dummy: ReplanQueueDJ) -> list[ProgramTerm]:
    """Return the terms the queue's DJ holds for the boundaries of its last pass."""
    return list(dummy._term_store("queue-1").terms())


async def _due_term(
    dummy: ReplanQueueDJ, tracks: list[FakeQueueItem], index: int, seconds_left: float
) -> None:
    """Let the player run into the end of the given track, where its term is due."""
    queue = dummy.player_queues._queue
    queue.current_item = tracks[index]
    queue.corrected_elapsed_time = TRACK_SECONDS - seconds_left
    await dummy._pump_terms("queue-1")


async def _drain_tasks(mass: FakeMass, waves: int = 20) -> None:
    """Run every task the harness started, including the ones those tasks start."""
    for _ in range(waves):
        pending = [task for task in mass.tasks if not task.done()]
        if not pending:
            return
        await asyncio.gather(*pending)


def _intro_host(host_id: str = "rick") -> dict[str, Any]:
    """Return a host that speaks a section at the start of the playlist."""
    host = _must_host()
    host["id"] = host_id
    host["name"] = host_id.title()
    host["section_order"] = [
        {"when": "start_of_playlist", "flow": [{"MUST": "Song_Transition"}]},
        {"when": "between_songs", "flow": [{"MUST": "Song_Transition"}]},
    ]
    return host


def _intro_pending(dummy: ReplanQueueDJ) -> bool:
    """Return whether the DJ owes its queue an intro it could not speak yet."""
    state: DJQueueState = dummy._dj_queues["queue-1"]
    return state.intro_pending


def _armed_intro_dj(tmp_path: Path, host: dict[str, Any] | None = None) -> ReplanQueueDJ:
    """Build an armed DJ on a queue that mixes programs, with an intro-speaking host."""
    dummy = _make_replan_dj(
        tmp_path, [_track(index) for index in range(4)], host=host or _intro_host()
    )
    dummy.mass.activate_program_bus("queue-1")
    return dummy


def _overlay_dj(
    tmp_path: Path, host: dict[str, Any] | None = None
) -> tuple[ReplanQueueDJ, list[FakeQueueItem]]:
    """Build an armed DJ on a playing, program-mixing queue at a known position."""
    tracks = [_track(index) for index in range(4)]
    for track in tracks:
        track.duration = TRACK_SECONDS
    dummy = _make_replan_dj(tmp_path, tracks, host=host or _must_host())
    dummy.mass.activate_program_bus("queue-1")
    queue = dummy.player_queues._queue
    queue.current_item = tracks[0]
    queue.corrected_elapsed_time = POSITION_SECONDS
    return dummy, tracks


async def _replan_with_the_guard_moved(dummy: ReplanQueueDJ, guard_index: int) -> None:
    """Advance the player's buffer mid-pass, so planned gaps fall behind the guard."""
    prepare = dummy._prepare_runtime_tokens

    async def _slow_prepare(program: dict[str, Any]) -> dict[str, str]:
        # stands in for a slow token source: the player buffers ahead while we wait
        dummy.player_queues._queue.index_in_buffer = guard_index
        return await prepare(program)

    dummy._prepare_runtime_tokens = _slow_prepare  # type: ignore[method-assign]
    try:
        await dummy._replan_queue("queue-1")
    finally:
        dummy._prepare_runtime_tokens = prepare  # type: ignore[method-assign]


def _announced_gaps(dummy: ReplanQueueDJ) -> set[Any]:
    """Return the tracks the clips of the last pass were spliced in front of."""
    return {
        items[0].extra_attributes[ATTR_GAP_NEXT_ID] for items, _index in dummy.player_queues.loads
    }


def _outro_clips(dummy: ReplanQueueDJ) -> list[Any]:
    """Return the DJ clips that announce no following track, i.e. a show's outro."""
    return [
        item
        for item in dummy.player_queues.items("queue-1")
        if item.extra_attributes.get(ATTR_QUEUE_DJ)
        and item.extra_attributes.get(ATTR_GAP_NEXT_ID) is None
    ]


async def test_activating_the_dj_mixes_the_hosts_intro_over_the_stream(tmp_path: Path) -> None:
    """Switching a DJ on speaks its start-of-playlist section as a single overlay."""
    dummy = _armed_intro_dj(tmp_path)

    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)

    assert len(dummy.mass.overlays) == 1
    queue_id, pcm, label, start_in_seconds = dummy.mass.overlays[0]
    assert queue_id == "queue-1"
    assert label == OVERLAY_LABEL_INTRO
    # the intro is due right away, not at a boundary further ahead
    assert start_in_seconds == 0.0
    # it is an intro, so the stream's buffer depth never drops it as late
    assert dummy.mass.overlay_kinds == [ProgramKind.INTRO]
    # it is the intro section, voiced by the host that was switched on
    assert dummy.rendered == [("rick", "Song_Transition")]
    assert pcm


async def test_a_later_replan_keeps_the_intro_to_its_one_airing(tmp_path: Path) -> None:
    """A replan after the intro aired does not speak it a second time."""
    dummy = _armed_intro_dj(tmp_path)
    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)

    dummy._schedule_replan("queue-1")
    await _drain_tasks(dummy.mass)

    assert len(dummy.mass.overlays) == 1


async def test_the_intro_stays_out_when_the_setting_is_off(tmp_path: Path) -> None:
    """With the intro switched off nothing is rendered and nothing is mixed in."""
    dummy = _armed_intro_dj(tmp_path)
    dummy.config_values[CONF_PLAY_INTRO_ON_ACTIVATE] = False

    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)

    assert dummy.mass.overlays == []
    assert dummy.rendered == []


async def test_an_intro_without_a_live_flow_stream_is_left_out(tmp_path: Path) -> None:
    """A queue that is not streaming has no intro mixed into anything, and no error either."""
    dummy = _make_replan_dj(tmp_path, [_track(index) for index in range(4)], host=_intro_host())

    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)

    assert dummy.mass.overlays == []
    # the breaks are still planned as queue items, so nothing else was held up
    assert dummy.player_queues.loads


async def test_a_paused_queue_is_not_spoken_an_intro(tmp_path: Path) -> None:
    """An intro is not mixed into a queue that plays nothing right now."""
    dummy = _armed_intro_dj(tmp_path)
    dummy.player_queues._queue.state = PlaybackState.PAUSED

    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)

    assert dummy.mass.overlays == []


async def test_a_host_without_an_intro_section_is_not_spoken(tmp_path: Path) -> None:
    """A host that only speaks between songs gets no intro overlay."""
    dummy = _armed_intro_dj(tmp_path, host=_must_host())

    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)

    assert dummy.mass.overlays == []


async def test_a_failed_intro_render_is_logged_and_planning_resumes(tmp_path: Path) -> None:
    """An engine that fails the intro leaves no overlay behind and does not block the breaks."""
    dummy = _armed_intro_dj(tmp_path)
    dummy.fail_overlay_render = True

    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)

    assert dummy.mass.overlays == []
    assert dummy._dj_queues["queue-1"].intro_until is None
    # the boundaries are decided once the failed intro is out of the way, none as an item
    assert dummy.player_queues.loads == []
    assert _terms(dummy)


async def test_planning_holds_while_the_intro_renders(tmp_path: Path) -> None:
    """No break is planned while the intro is still being rendered."""
    dummy = _armed_intro_dj(tmp_path)
    state = dummy._dj_queues["queue-1"]
    state.intro_until = asyncio.get_running_loop().time() + INTRO_RENDER_TIMEOUT_SECONDS

    await dummy._replan_queue("queue-1")

    assert dummy.player_queues.loads == []
    assert state.decided_gap_ids == set()


async def test_a_hanging_intro_render_stops_holding_planning_at_the_deadline(
    tmp_path: Path,
) -> None:
    """An intro that never finishes holds the planner only until its cap, then lets it go."""
    dummy = _armed_intro_dj(tmp_path)
    state = dummy._dj_queues["queue-1"]
    state.intro_until = asyncio.get_running_loop().time() - 1

    await dummy._replan_queue("queue-1")

    assert state.intro_until is None
    assert dummy.player_queues.loads == []
    assert _terms(dummy)


async def test_reactivating_the_dj_speaks_the_intro_again(tmp_path: Path) -> None:
    """Switching the DJ off and on again speaks a fresh intro for the new session."""
    dummy = _armed_intro_dj(tmp_path)

    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)
    await dummy.set_queue_dj("queue-1", None)
    await _drain_tasks(dummy.mass)
    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)

    assert len(dummy.mass.overlays) == 2


async def test_a_show_start_speaks_the_hosts_intro_exactly_once(tmp_path: Path) -> None:
    """A show arming its host speaks one intro, not one per arming path."""
    dummy = ReplanQueueDJ(
        tmp_path,
        FakeQueue("queue-1", 0, 0),
        [_track(index) for index in range(4)],
        _intro_host("amy"),
    )
    dummy._stations["morning_show"] = {"id": "morning_show", "host_id": "amy"}
    _fake_queue_with_sources(dummy, "queue-1", [f"{dummy.instance_id}://radio/morning_show"])
    dummy.mass.activate_program_bus("queue-1")

    await dummy._ensure_show_dj("queue-1")
    await _drain_tasks(dummy.mass)

    assert len(dummy.mass.overlays) == 1
    assert dummy.mass.overlays[0][2] == OVERLAY_LABEL_INTRO


async def test_a_show_that_arms_while_its_queue_loads_speaks_the_intro_once_it_plays(
    tmp_path: Path,
) -> None:
    """A show arm precedes playback, so its intro is owed until the stream can air it."""
    dummy = ReplanQueueDJ(
        tmp_path,
        FakeQueue("queue-1", 0, 0),
        [_track(index) for index in range(4)],
        _intro_host("amy"),
    )
    dummy._stations["morning_show"] = {"id": "morning_show", "host_id": "amy"}
    _fake_queue_with_sources(dummy, "queue-1", [f"{dummy.instance_id}://radio/morning_show"])
    # the switch is on, but the queue is still loading and no flow session exists yet
    dummy.mass.streams.program_bus_enabled_flag = True
    dummy.player_queues._queue.state = PlaybackState.IDLE

    await dummy._ensure_show_dj("queue-1")
    await _drain_tasks(dummy.mass)

    assert dummy.mass.overlays == []
    assert _intro_pending(dummy)

    # the queue starts playing and its stream publishes a bus
    dummy.mass.activate_program_bus("queue-1")
    dummy.player_queues._queue.state = PlaybackState.PLAYING
    await dummy._replan_queue("queue-1")
    await _drain_tasks(dummy.mass)

    assert len(dummy.mass.overlays) == 1
    assert dummy.mass.overlays[0][2] == OVERLAY_LABEL_INTRO
    assert not _intro_pending(dummy)

    # a later pass owes nothing
    await dummy._replan_queue("queue-1")
    await _drain_tasks(dummy.mass)
    assert len(dummy.mass.overlays) == 1


async def test_a_dj_switched_on_before_its_queue_plays_speaks_the_intro_later(
    tmp_path: Path,
) -> None:
    """Switching a DJ on at an idle queue speaks its intro when playback starts."""
    dummy = _make_replan_dj(tmp_path, [_track(index) for index in range(4)], host=_intro_host())
    # the switch is on; the queue plays nothing yet and has no flow session of its own
    dummy.mass.streams.program_bus_enabled_flag = True
    dummy.player_queues._queue.state = PlaybackState.IDLE

    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)

    assert dummy.mass.overlays == []
    assert _intro_pending(dummy)

    dummy.mass.activate_program_bus("queue-1")
    dummy.player_queues._queue.state = PlaybackState.PLAYING
    await dummy._replan_queue("queue-1")
    await _drain_tasks(dummy.mass)

    assert len(dummy.mass.overlays) == 1
    assert dummy.mass.overlays[0][2] == OVERLAY_LABEL_INTRO


async def test_an_owed_intro_is_looked_for_again_when_the_stream_opens_late(
    tmp_path: Path,
) -> None:
    """A playing queue whose bus opens after the switch still hears its intro."""
    dummy = _make_replan_dj(tmp_path, [_track(index) for index in range(4)], host=_intro_host())
    # the switch is on, but the queue's flow session has not opened a bus yet
    dummy.mass.streams.program_bus_enabled_flag = True

    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)

    assert dummy.mass.overlays == []
    assert _intro_pending(dummy)
    retry = next(timer for timer in dummy.mass.timers if timer[3] == "ai_radio_intro_retry_queue-1")

    dummy.mass.activate_program_bus("queue-1")
    retry[1](*retry[2])
    await _drain_tasks(dummy.mass)

    assert len(dummy.mass.overlays) == 1
    assert dummy.mass.overlays[0][2] == OVERLAY_LABEL_INTRO


async def test_a_dj_that_is_switched_off_again_owes_no_intro(tmp_path: Path) -> None:
    """A DJ disabled before its queue plays speaks nothing once playback starts."""
    dummy = _make_replan_dj(tmp_path, [_track(index) for index in range(4)], host=_intro_host())
    dummy.player_queues._queue.state = PlaybackState.IDLE
    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)

    await dummy.set_queue_dj("queue-1", None)
    dummy.mass.activate_program_bus("queue-1")
    dummy.player_queues._queue.state = PlaybackState.PLAYING
    await dummy._replan_queue("queue-1")
    await _drain_tasks(dummy.mass)

    assert dummy.mass.overlays == []


async def test_a_restored_dj_owes_no_intro(tmp_path: Path) -> None:
    """A DJ restored from the cache sets no owed intro, so it stays silent once it plays."""
    first = _armed_intro_dj(tmp_path)
    await first.set_queue_dj("queue-1", "rick")
    await _drain_tasks(first.mass)

    restored = _armed_intro_dj(tmp_path)
    await restored._load_queue_dj()
    await restored._replan_queue("queue-1")
    await _drain_tasks(restored.mass)

    assert not _intro_pending(restored)
    assert restored.mass.overlays == []


async def test_a_dj_restored_from_the_cache_speaks_no_intro(tmp_path: Path) -> None:
    """A DJ that comes back from the cache after a restart stays silent until it is switched on."""
    first = _armed_intro_dj(tmp_path)
    await first.set_queue_dj("queue-1", "rick")
    await _drain_tasks(first.mass)
    assert len(first.mass.overlays) == 1

    restored = _armed_intro_dj(tmp_path)
    await restored._load_queue_dj()

    assert restored._dj_queues["queue-1"].host_id == "rick"
    assert restored.mass.overlays == []
    assert restored.rendered == []


async def test_a_host_switch_speaks_the_new_hosts_intro_once(tmp_path: Path) -> None:
    """Switching the host mid-queue speaks the new host's intro, and only that one."""
    dummy = _armed_intro_dj(tmp_path)
    dummy._hosts["amy"] = _intro_host("amy")
    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)

    await dummy.set_queue_dj("queue-1", "amy")
    await _drain_tasks(dummy.mass)

    assert len(dummy.mass.overlays) == 2
    assert [host_id for host_id, _section in dummy.rendered] == ["rick", "amy"]


async def test_a_bus_that_disappeared_before_the_intro_renders_is_survived(
    tmp_path: Path,
) -> None:
    """A flow session that ended while the intro rendered is not an error."""
    dummy = _armed_intro_dj(tmp_path)
    await dummy.set_queue_dj("queue-1", "rick")
    dummy.mass.drop_program_bus("queue-1")

    await _drain_tasks(dummy.mass)

    assert dummy.mass.overlays == []
    assert dummy._dj_queues["queue-1"].intro_until is None


async def test_the_intro_stays_out_with_the_program_bus_switched_off(tmp_path: Path) -> None:
    """The experimental program bus switch is the kill switch for the intro too."""
    dummy = _armed_intro_dj(tmp_path)
    dummy.mass.streams.program_bus_enabled_flag = False

    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)

    assert dummy.mass.overlays == []


async def test_a_host_that_keeps_its_intro_on_speaks_it(tmp_path: Path) -> None:
    """A host that leaves its switch on speaks its intro exactly as before."""
    host = _intro_host()
    host["play_intro"] = True
    dummy = _armed_intro_dj(tmp_path, host=host)

    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)

    assert [label for _qid, _pcm, label, _start in dummy.mass.overlays] == [OVERLAY_LABEL_INTRO]
    assert dummy.rendered == [("rick", "Song_Transition")]


async def test_a_host_that_switches_its_intro_off_speaks_none(tmp_path: Path) -> None:
    """The host's own switch decides, even for a host that plans a start-of-playlist section."""
    host = _intro_host()
    host["play_intro"] = False
    dummy = _armed_intro_dj(tmp_path, host=host)

    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)

    assert dummy.mass.overlays == []
    assert dummy.rendered == []
    # nothing is owed either, so no lookup is booked for a later stream
    assert not _intro_pending(dummy)


async def test_a_host_that_switches_its_intro_off_books_no_lookups(tmp_path: Path) -> None:
    """A switched-off intro is not looked for again when the stream opens late."""
    host = _intro_host()
    host["play_intro"] = False
    dummy = _armed_intro_dj(tmp_path, host=host)
    dummy.mass.drop_program_bus("queue-1")

    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)

    assert not _intro_pending(dummy)
    # no intro lookup is booked for a later stream, whatever else the DJ scheduled
    assert not [
        timer for timer in dummy.mass.timers if (timer[3] or "").startswith("ai_radio_intro")
    ]


async def test_the_provider_switch_off_wins_over_a_host_that_wants_its_intro(
    tmp_path: Path,
) -> None:
    """The global switch is the stricter side, so it keeps a host's intro switched off."""
    dummy = _armed_intro_dj(tmp_path)  # the host leaves its switch on
    dummy.config_values[CONF_PLAY_INTRO_ON_ACTIVATE] = False

    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)

    assert dummy.mass.overlays == []
    assert dummy.rendered == []


async def test_a_boundary_the_player_already_holds_is_a_term_not_an_item(
    tmp_path: Path,
) -> None:
    """Every boundary is spoken over the stream; the queue never gains an item for it."""
    dummy, tracks = _overlay_dj(tmp_path)
    dummy._sections["Song_Transition"]["allow_post"] = True

    await _replan_with_the_guard_moved(dummy, 1)

    assert dummy.player_queues.loads == []
    # the pass serves the boundary the running item ends at along with the ones ahead
    assert len(_terms(dummy)) == 3

    # once that item plays and its end comes close, its term is placed on the boundary
    await _due_term(dummy, tracks, 1, 8.0)

    assert len(dummy.mass.overlays) == 1
    queue_id, pcm, label, delay = dummy.mass.overlays[0]
    assert queue_id == "queue-1"
    assert label == OVERLAY_LABEL_TRANSITION
    assert pcm
    # the running item has this much left, so the break starts on its boundary; the bus
    # measures the delay from what the listener hears
    assert delay == pytest.approx(8.0 - TERM_START_BIAS_SECONDS)
    # the boundary the placed term speaks is decided in the store, not in the item path's set
    store = dummy._term_store("queue-1")
    assert store.state((tracks[1].queue_item_id, tracks[2].queue_item_id)) is (
        BoundaryState.DECIDED
    )


async def test_a_term_without_a_playhead_is_dropped(tmp_path: Path) -> None:
    """Without a position there is no boundary to aim at, so nothing is mixed in."""
    dummy, _tracks = _overlay_dj(tmp_path)
    dummy.player_queues._queue.current_item = None

    await _replan_with_the_guard_moved(dummy, 1)
    await _drain_tasks(dummy.mass)

    assert dummy.mass.overlays == []
    assert dummy.player_queues.loads == []
    assert all(term.is_terminal for term in _terms(dummy))


async def test_a_cut_short_head_leaves_no_term_to_place(tmp_path: Path) -> None:
    """A break whose head no longer fits has no start left, so it is dropped, never placed."""
    host = _must_host()
    host["term_head_seconds"] = 12.0
    dummy, tracks = _overlay_dj(tmp_path, host=host)
    dummy._sections["Song_Transition"]["allow_pre"] = True

    async def _long_pcm(
        _queue_id: str,
        _program: dict[str, Any],
        _section: PlannedSection,
        pcm_format: AudioFormat,
    ) -> bytes:
        # a break long enough to want a full head, against a boundary that is closer
        return b"\x00\x00" * int(20 * pcm_format.sample_rate) * pcm_format.channels

    dummy._render_section_pcm = _long_pcm  # type: ignore[method-assign]
    await dummy._replan_queue("queue-1")
    await _due_term(dummy, tracks, 1, 10.0)

    assert dummy.mass.overlays == []
    cut = next(term for term in _terms(dummy) if term.successor_id == tracks[2].queue_item_id)
    assert cut.dropped_reason == "too_late"


async def test_a_too_close_gap_without_a_bus_inserts_nothing(tmp_path: Path) -> None:
    """Without a flow session a too-close gap stays empty, exactly as before."""
    tracks = [_track(index) for index in range(4)]
    for track in tracks:
        track.duration = TRACK_SECONDS
    dummy = _make_replan_dj(tmp_path, tracks)
    dummy.player_queues._queue.current_item = tracks[0]
    dummy.player_queues._queue.corrected_elapsed_time = POSITION_SECONDS

    await _replan_with_the_guard_moved(dummy, 1)

    assert dummy.mass.overlays == []
    assert dummy.rendered == []
    assert _announced_gaps(dummy) == {tracks[3].queue_item_id}


async def test_a_too_close_gap_with_the_switch_on_but_no_live_stream_inserts_nothing(
    tmp_path: Path,
) -> None:
    """The switch alone is not enough: without a live bus the gap stays an empty item gap."""
    tracks = [_track(index) for index in range(4)]
    for track in tracks:
        track.duration = TRACK_SECONDS
    dummy = _make_replan_dj(tmp_path, tracks)
    # the experimental switch is on, but this queue's stream carries no program bus
    dummy.mass.streams.program_bus_enabled_flag = True
    dummy.player_queues._queue.current_item = tracks[0]
    dummy.player_queues._queue.corrected_elapsed_time = POSITION_SECONDS

    await _replan_with_the_guard_moved(dummy, 1)
    await _drain_tasks(dummy.mass)
    # the play's flow session may still arrive, so the terms first wait out their grace
    for term in _terms(dummy):
        term.bus_missing_since = asyncio.get_running_loop().time() - BUS_GRACE_SECONDS - 1
    await dummy._pump_terms("queue-1")

    assert dummy.mass.overlays == []
    assert dummy.rendered == []
    assert dummy.player_queues.loads == []
    assert [term.dropped_reason for term in _terms(dummy)] == ["no_bus", "no_bus", "no_bus"]
    # a term that never aired holds nothing back: the gap may be planned again
    store = dummy._term_store("queue-1")
    assert store.state((tracks[1].queue_item_id, tracks[2].queue_item_id)) is (BoundaryState.OPEN)


async def test_a_plannable_gap_becomes_a_term_not_a_clip(tmp_path: Path) -> None:
    """A planned boundary is decided as a term, and the queue keeps its items."""
    dummy, _tracks = _overlay_dj(tmp_path)

    await dummy._replan_queue("queue-1")

    assert dummy.mass.overlays == []
    assert dummy.player_queues.loads == []
    assert len(_terms(dummy)) == 3


async def test_a_clip_the_queue_still_holds_is_taken_out_before_its_term(
    tmp_path: Path,
) -> None:
    """A clip the queue still holds goes once the bus plans its boundary as a term."""
    dummy, tracks = _overlay_dj(tmp_path)
    state = dummy._dj_queues["queue-1"]
    existing = _dj_clip(tracks[2].queue_item_id, state.dj_session_id)
    dummy.player_queues._items = [tracks[0], tracks[1], existing, tracks[2], tracks[3]]

    await dummy._replan_queue("queue-1")

    assert dummy.mass.overlays == []
    assert dummy.player_queues.deleted == [existing.queue_item_id]
    assert dummy.player_queues.loads == []
    # the boundary the clip served is decided as a term for this session
    assert len(_terms(dummy)) == 3
    store = dummy._term_store("queue-1")
    assert store.state((tracks[1].queue_item_id, tracks[2].queue_item_id)) is (
        BoundaryState.DECIDED
    )
    assert state.decided_gap_ids == set()


async def test_switching_the_bus_on_takes_the_item_clips_out_before_a_term(
    tmp_path: Path,
) -> None:
    """A queue switched to terms never airs an item clip and a term at the same boundary."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, tracks)
    queue = dummy.player_queues._queue
    queue.current_item = tracks[0]
    queue.corrected_elapsed_time = POSITION_SECONDS

    # the flag-off path speaks both boundaries as queue items
    await dummy._replan_queue("queue-1")
    assert (
        len(
            [
                item
                for item in dummy.player_queues.items("queue-1")
                if item.extra_attributes.get(ATTR_QUEUE_DJ)
            ]
        )
        == 2
    )
    loads_before_switch = list(dummy.player_queues.loads)

    # switching the program bus on takes them out before a term is planned for the same gap
    dummy.mass.activate_program_bus("queue-1")
    await dummy._replan_queue("queue-1")

    assert [
        item
        for item in dummy.player_queues.items("queue-1")
        if item.extra_attributes.get(ATTR_QUEUE_DJ)
    ] == []
    # the released gaps are planned as terms, never spliced as items again
    assert dummy.player_queues.loads == loads_before_switch
    assert len(_terms(dummy)) == 3
    # and the boundary is spoken over the stream instead of by a clip item
    await _due_term(dummy, tracks, 1, 8.0)
    assert len(dummy.mass.overlays) == 1


async def test_an_event_after_the_switch_takes_the_item_clips_out(tmp_path: Path) -> None:
    """A queue event after the switch purges the clips before the term pass can air them."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, tracks)
    queue = dummy.player_queues._queue
    queue.current_item = tracks[0]
    queue.corrected_elapsed_time = POSITION_SECONDS
    await dummy._replan_queue("queue-1")
    assert any(
        item.extra_attributes.get(ATTR_QUEUE_DJ) for item in dummy.player_queues.items("queue-1")
    )

    dummy.mass.activate_program_bus("queue-1")
    await dummy._on_dj_queue_event(
        SimpleNamespace(event=EventType.QUEUE_UPDATED, object_id="queue-1")
    )
    await _drain_tasks(dummy.mass)

    # the event's own pass planned the boundaries again, and no clip item is left
    assert not any(
        item.extra_attributes.get(ATTR_QUEUE_DJ) for item in dummy.player_queues.items("queue-1")
    )
    assert len(_terms(dummy)) == 3


async def test_a_gap_that_vanished_gets_neither_a_clip_nor_an_overlay(tmp_path: Path) -> None:
    """A target that left the queue mid-pass is not spoken over either."""
    dummy, tracks = _overlay_dj(tmp_path)
    prepare = dummy._prepare_runtime_tokens

    async def _remove_target(program: dict[str, Any]) -> dict[str, str]:
        dummy.player_queues.update_items(
            "queue-1",
            [item for item in dummy.player_queues.items("queue-1") if item is not tracks[2]],
        )
        return await prepare(program)

    dummy._prepare_runtime_tokens = _remove_target  # type: ignore[method-assign]
    await dummy._replan_queue("queue-1")
    await _drain_tasks(dummy.mass)

    assert dummy.mass.overlays == []
    assert dummy.player_queues.loads == []
    # the boundary whose target left the queue is given up, never spoken over a new one
    await _due_term(dummy, tracks, 1, 8.0)
    assert dummy.mass.overlays == []
    assert (
        next(
            term.dropped_reason
            for term in _terms(dummy)
            if term.queue_item_id == tracks[1].queue_item_id
        )
        == "no_successor"
    )


async def test_an_unplayable_neighbour_gets_no_break_and_no_overlay(tmp_path: Path) -> None:
    """A boundary with a track that cannot play is not spoken over at all."""
    tracks = [_track(index) for index in range(5)]
    for track in tracks:
        track.duration = TRACK_SECONDS
    tracks[2].available = False
    dummy = _make_replan_dj(tmp_path, tracks)
    dummy.mass.activate_program_bus("queue-1")
    dummy.player_queues._queue.current_item = tracks[0]
    dummy.player_queues._queue.corrected_elapsed_time = POSITION_SECONDS

    await dummy._replan_queue("queue-1")

    assert dummy.mass.overlays == []
    assert dummy.rendered == []
    announced = _announced_gaps(dummy)
    assert tracks[2].queue_item_id not in announced
    assert tracks[3].queue_item_id not in announced


async def test_a_paused_queue_gets_no_transition_overlay(tmp_path: Path) -> None:
    """A queue that is not playing is not spoken over, however close the boundary is."""
    dummy, _tracks = _overlay_dj(tmp_path)
    dummy.player_queues._queue.state = PlaybackState.PAUSED

    await _replan_with_the_guard_moved(dummy, 1)

    assert dummy.mass.overlays == []


async def test_each_boundary_gets_one_term_and_never_two(tmp_path: Path) -> None:
    """Two boundaries behind each other are decided once each, however often a pass runs."""
    dummy, _tracks = _overlay_dj(tmp_path)

    await _replan_with_the_guard_moved(dummy, 2)
    assert len(_terms(dummy)) == 3
    assert dummy.player_queues.loads == []

    await dummy._replan_queue("queue-1")

    assert len(_terms(dummy)) == 3
    assert dummy.player_queues.loads == []


async def test_a_posting_break_keeps_its_delay(tmp_path: Path) -> None:
    """A break that plays on over the next intro still starts on its boundary."""
    dummy, tracks = _overlay_dj(tmp_path)
    dummy._sections["Song_Transition"]["allow_post"] = True

    await dummy._replan_queue("queue-1")
    await _due_term(dummy, tracks, 1, 8.0)

    assert len(dummy.mass.overlays) == 1
    assert dummy.mass.overlays[0][3] == pytest.approx(8.0 - TERM_START_BIAS_SECONDS)


async def test_a_break_is_dropped_when_the_track_moves_on_while_it_renders(
    tmp_path: Path,
) -> None:
    """A break timed to a boundary the player left while rendering is not aired elsewhere."""
    dummy, tracks = _overlay_dj(tmp_path)
    render = dummy._render_section_pcm
    queue = dummy.player_queues._queue

    async def _render_slowly(
        queue_id: str, program: dict[str, Any], section: PlannedSection, pcm_format: AudioFormat
    ) -> bytes:
        # the track ends while the engine renders, so the boundary the break was written
        # for is behind the player by the time the audio comes back
        queue.current_item = tracks[2]
        queue.corrected_elapsed_time = 1.0
        return await render(queue_id, program, section, pcm_format)

    dummy._render_section_pcm = _render_slowly  # type: ignore[method-assign]
    await dummy._replan_queue("queue-1")

    await _due_term(dummy, tracks, 1, 8.0)

    assert dummy.mass.overlays == []
    rendered = next(term for term in _terms(dummy) if term.successor_id == tracks[2].queue_item_id)
    assert rendered.dropped_reason == "skipped"


async def test_a_break_is_dropped_when_the_queue_stops_while_it_renders(
    tmp_path: Path,
) -> None:
    """A break rendered while the queue paused is not mixed into a stream nobody hears."""
    dummy, tracks = _overlay_dj(tmp_path)
    render = dummy._render_section_pcm
    queue = dummy.player_queues._queue

    async def _render_slowly(
        queue_id: str, program: dict[str, Any], section: PlannedSection, pcm_format: AudioFormat
    ) -> bytes:
        queue.state = PlaybackState.PAUSED
        return await render(queue_id, program, section, pcm_format)

    dummy._render_section_pcm = _render_slowly  # type: ignore[method-assign]
    await dummy._replan_queue("queue-1")

    await _due_term(dummy, tracks, 1, 8.0)

    # a render that landed in a pause is not placed and not lost: the term waits again
    assert dummy.mass.overlays == []
    waiting = next(term for term in _terms(dummy) if term.successor_id == tracks[2].queue_item_id)
    assert waiting.state is TermState.DECIDED


def _gap_section() -> PlannedSection:
    """Return the plan of the break in front of the third track."""
    return PlannedSection(
        order=0,
        clip_id="",
        section_id="Song_Transition",
        section_name="Song Transition",
        when="between_songs",
        insert_at_index=2,
        prompt="",
        max_chars=200,
        web_search_mode="disabled",
    )


def _dj_program(dummy: ReplanQueueDJ) -> dict[str, Any]:
    """Return the program a break on the harness's DJ belongs to."""
    return dummy._build_program({"id": "morning_show", "name": "AI DJ Rick"}, dummy._hosts["rick"])


async def test_an_intro_that_trips_the_cooldown_books_the_same_wake_up(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The intro path pauses the DJ through the same place as a break does."""
    dummy = _armed_intro_dj(tmp_path)
    state = dummy._dj_queues["queue-1"]
    state.render_health.tts.failures = RENDER_FAILURES_BEFORE_COOLDOWN - 1
    dummy.fail_overlay_render = True

    with caplog.at_level(logging.WARNING):
        await dummy._place_intro_overlay("queue-1", state)
    await _drain_tasks(dummy.mass)

    assert dummy.mass.overlays == []
    assert state.render_health.cooling_down(asyncio.get_running_loop().time())
    assert (
        TTS_COOLDOWN_SECONDS,
        dummy._schedule_replan,
        ("queue-1",),
        "ai_radio_dj_cooldown_queue-1",
    ) in dummy.mass.timers
    assert "pauses the breaks on queue queue-1" in caplog.text


async def test_the_intro_is_not_rendered_while_the_breaks_are_paused(tmp_path: Path) -> None:
    """An engine that is paused for failing is not paid for the intro either."""
    dummy = _armed_intro_dj(tmp_path)
    state = dummy._dj_queues["queue-1"]
    state.render_health.tts.cooldown_until = asyncio.get_running_loop().time() + 100

    await dummy._place_intro_overlay("queue-1", state)

    assert dummy.rendered == []
    assert dummy.mass.overlays == []


async def test_a_failed_intro_counts_against_the_dj(tmp_path: Path) -> None:
    """A failed intro render counts towards the pause of the queue's breaks."""
    dummy = _armed_intro_dj(tmp_path)
    state = dummy._dj_queues["queue-1"]
    state.render_health.tts.failures = RENDER_FAILURES_BEFORE_COOLDOWN - 1
    dummy.fail_overlay_render = True

    await dummy._place_intro_overlay("queue-1", state)

    assert state.render_health.cooling_down(asyncio.get_running_loop().time())


async def test_a_successful_intro_clears_the_failures(tmp_path: Path) -> None:
    """An intro that comes back clears the failure count the DJ was holding."""
    dummy = _armed_intro_dj(tmp_path)
    state = dummy._dj_queues["queue-1"]
    state.render_health.tts.failures = 2

    await dummy._place_intro_overlay("queue-1", state)

    assert state.render_health.tts.failures == 0
    assert len(dummy.mass.overlays) == 1


async def test_the_intro_hold_ends_at_its_deadline_without_another_queue_event(
    tmp_path: Path,
) -> None:
    """A hanging intro render stops holding the breaks back at the deadline it was given."""
    dummy = _armed_intro_dj(tmp_path)

    async def _hang(*_args: Any) -> bytes:
        await asyncio.Event().wait()
        raise AssertionError("the hanging render never returns")

    dummy._render_section_pcm = _hang  # type: ignore[assignment]
    state = dummy._dj_queues["queue-1"]
    dummy._start_intro_overlay("queue-1", state)
    hold = next(timer for timer in dummy.mass.timers if timer[3] == "ai_radio_intro_hold_queue-1")

    assert dummy._intro_holds_breaks("queue-1", state) is True

    hold[1](*hold[2])

    assert state.intro_until is None
    assert dummy._intro_holds_breaks("queue-1", state) is False
    # the pass that would plan the breaks is booked without waiting for a queue event
    assert state.replan_pending is True
    for task in dummy.mass.tasks:
        task.cancel()
    await asyncio.sleep(0)


async def test_a_skip_while_the_intro_renders_drops_it(tmp_path: Path) -> None:
    """An intro rendered for a track the listener skipped past is not spoken over the next."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, tracks, host=_intro_host())
    dummy.mass.activate_program_bus("queue-1")
    queue = dummy.player_queues._queue
    queue.current_item = tracks[0]
    render = dummy._render_section_pcm

    async def _render_slowly(
        queue_id: str, program: dict[str, Any], section: PlannedSection, pcm_format: AudioFormat
    ) -> bytes:
        # the listener skips to the next track while the engine works
        queue.current_item = tracks[1]
        return await render(queue_id, program, section, pcm_format)

    dummy._render_section_pcm = _render_slowly  # type: ignore[method-assign]

    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)

    assert dummy.mass.overlays == []
    assert not _intro_pending(dummy)


async def test_loading_another_album_before_the_play_drops_the_owed_intro(
    tmp_path: Path,
) -> None:
    """An intro owed to a paused album is not spoken over the album loaded after it."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, tracks, host=_intro_host())
    # the switch is on; the queue is paused and its flow session does not exist yet
    dummy.mass.streams.program_bus_enabled_flag = True
    queue = dummy.player_queues._queue
    queue.state = PlaybackState.PAUSED
    queue.current_item = tracks[0]

    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)
    assert _intro_pending(dummy)

    # another album is loaded and playback starts on it
    other = [_track(index) for index in range(10, 14)]
    dummy.player_queues._items = other
    queue.current_item = other[0]
    queue.state = PlaybackState.PLAYING
    dummy.mass.activate_program_bus("queue-1")
    await dummy._replan_queue("queue-1")
    await _drain_tasks(dummy.mass)

    assert dummy.mass.overlays == []
    assert not _intro_pending(dummy)


async def test_playback_ending_while_the_intro_renders_drops_it(tmp_path: Path) -> None:
    """An intro that comes back after its playback ended is not mixed into the next one."""
    tracks = [_track(index) for index in range(4)]
    dummy = _make_replan_dj(tmp_path, tracks, host=_intro_host())
    dummy.mass.activate_program_bus("queue-1")
    queue = dummy.player_queues._queue
    queue.current_item = tracks[0]
    render = dummy._render_section_pcm

    async def _render_slowly(
        program: dict[str, Any], section: PlannedSection, pcm_format: AudioFormat
    ) -> bytes:
        # the queue runs out of items while the engine works
        queue.ended = True
        return await render(program, section, pcm_format)

    dummy._render_section_pcm = _render_slowly  # type: ignore[method-assign]

    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)

    assert dummy.mass.overlays == []


async def test_the_program_bus_switch_off_books_no_intro_lookups(tmp_path: Path) -> None:
    """With the switch off no look for a stream that can never carry an intro is booked."""
    dummy = _make_replan_dj(tmp_path, [_track(index) for index in range(4)], host=_intro_host())
    assert dummy.mass.streams.program_bus_enabled_flag is False

    await dummy.set_queue_dj("queue-1", "rick")
    await _drain_tasks(dummy.mass)

    assert [
        timer for timer in dummy.mass.timers if timer[3] == "ai_radio_intro_retry_queue-1"
    ] == []
    assert not _intro_pending(dummy)


async def test_a_failed_transition_render_settles_the_gap_without_an_item(
    tmp_path: Path,
) -> None:
    """A break that fails to render is logged, leaves no item and is not retried forever."""
    dummy, tracks = _overlay_dj(tmp_path)
    dummy.fail_overlay_render = True

    await dummy._replan_queue("queue-1")
    await _drain_tasks(dummy.mass)
    await _due_term(dummy, tracks, 1, 8.0)

    assert dummy.mass.overlays == []
    assert len(dummy.rendered) == 1
    failed = next(term for term in _terms(dummy) if term.successor_id == tracks[2].queue_item_id)
    assert failed.dropped_reason == "render_failed"
    # the break never aired, so it holds neither its guard nor its gap back
    assert tracks[2].queue_item_id not in dummy._dj_queues["queue-1"].decided_gap_ids

    await dummy._replan_queue("queue-1")
    await _drain_tasks(dummy.mass)
    # the pass books itself for the moment its budget window opens, as it does in production;
    # it lands a moment inside the window, never exactly on its edge
    window = (
        dummy._term_estimate("queue-1").seconds
        + LEAD_MIN_LEAD_SECONDS
        + TERM_SCHEDULE_TOLERANCE_SECONDS
    )
    await _due_term(dummy, tracks, 1, window - 0.5)

    # a later pass tries the open gap again, and still makes no item for it
    assert len(dummy.rendered) == 2
    assert dummy.mass.overlays == []
    assert dummy.player_queues.loads == []


async def test_the_program_bus_switch_off_keeps_the_item_behaviour(tmp_path: Path) -> None:
    """With the experimental switch off a too-close gap behaves exactly as on dev."""
    dummy, tracks = _overlay_dj(tmp_path)
    dummy.mass.streams.program_bus_enabled_flag = False

    await _replan_with_the_guard_moved(dummy, 1)

    assert dummy.mass.overlays == []
    assert dummy.rendered == []
    assert _announced_gaps(dummy) == {tracks[3].queue_item_id}

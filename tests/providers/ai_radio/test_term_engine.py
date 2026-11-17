"""Unit tests for the life of a queue DJ's program terms."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from music_assistant_models.enums import (
    ContentType,
    CrossfadeMode,
    MediaType,
    PlaybackState,
)
from music_assistant_models.media_items import AudioFormat

from music_assistant.controllers.streams.constants import LEAD_MIN_LEAD_SECONDS
from music_assistant.controllers.streams.program_ledger import OverlayProgramState
from music_assistant.controllers.streams.program_mixer import FlowProgramBus
from music_assistant.controllers.streams.program_schedule import (
    ProgramKind,
    ScheduleReason,
    ScheduleResult,
)
from music_assistant.providers.ai_radio.constants import (
    ATTR_QUEUE_DJ,
    OVERLAY_LABEL_TRANSITION,
    TALK_OVER_MAX_PRE,
    TERM_HEAD_DEFAULT_SECONDS,
)
from music_assistant.providers.ai_radio.models import DJQueueState, PlannedSection
from music_assistant.providers.ai_radio.term_engine import (
    BUS_GRACE_SECONDS,
    PAUSED_TERM_PUMP_SECONDS,
    TERM_PUMP_MAX_SECONDS,
    AIRadioTermEngineMixin,
    TermBoundaryPlan,
    running_boundary,
    term_boundary_plans,
    term_plans_in_horizon,
)
from music_assistant.providers.ai_radio.term_timing import (
    RENDER_ESTIMATE_SECONDS,
    TERM_RENDER_MARGIN_SECONDS,
    TERM_SCHEDULE_TOLERANCE_SECONDS,
    TERM_START_BIAS_SECONDS,
)
from music_assistant.providers.ai_radio.terms import (
    TERMS_PER_QUEUE,
    BoundaryState,
    ProgramTerm,
    TermDropReason,
    TermState,
    TermStore,
    term_key,
)

PCM_FORMAT = AudioFormat(
    content_type=ContentType.PCM_S16LE,
    sample_rate=48000,
    bit_depth=16,
    channels=2,
)
# every stand-in render is this long
RENDER_SECONDS = 3.0
# the lead a live queue reported while its producer ran ahead of the listener
PRODUCER_LEAD_SECONDS = 12.34


def _pcm(seconds: float = RENDER_SECONDS) -> bytes:
    """Return silent PCM of the given length in the flow's format."""
    return b"\x00\x00" * int(seconds * PCM_FORMAT.sample_rate) * PCM_FORMAT.channels


@dataclass
class FakeQueueItem:
    """Minimal queue item stand-in."""

    queue_item_id: str
    duration: int | None = 200
    available: bool = True
    media_type: MediaType = MediaType.TRACK
    extra_attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Give the item the fields a queue item always has."""
        self.name = self.queue_item_id
        self.media_item = None
        self.streamdetails = None


@dataclass
class FakeQueue:
    """Minimal player queue stand-in."""

    queue_id: str
    items: list[FakeQueueItem]
    current: str | None = "a"
    elapsed: float = 0.0
    state: PlaybackState = PlaybackState.PLAYING
    ended: bool = False
    playback_speed: float = 1.0

    @property
    def current_item(self) -> FakeQueueItem | None:
        """Return the item the queue is playing."""
        return next((item for item in self.items if item.queue_item_id == self.current), None)

    @property
    def corrected_elapsed_time(self) -> float:
        """Return the media-time position in the running item."""
        return self.elapsed


class FakePlayerQueues:
    """Minimal player queues controller stand-in."""

    def __init__(self, queue: FakeQueue) -> None:
        """Keep one queue with its items."""
        self.queue = queue
        self.update_calls = 0

    def get(self, queue_id: str) -> FakeQueue | None:
        """Return the queue when the id matches."""
        return self.queue if queue_id == self.queue.queue_id else None

    def items(self, queue_id: str, limit: int = 500, offset: int = 0) -> list[FakeQueueItem]:
        """Return one page of the queue's items."""
        return self.queue.items[offset : offset + limit]

    def update_items(self, queue_id: str, items: list[FakeQueueItem]) -> None:
        """Record that a caller rewrote the queue."""
        self.update_calls += 1
        self.queue.items = items


class FakeConfig:
    """Minimal core config stand-in."""

    def __init__(self) -> None:
        """Start with the default crossfade length."""
        self.crossfade_duration = 8

    def get_raw_core_config_value(self, domain: str, key: str, default: Any = None) -> Any:
        """Return the configured crossfade length, or the caller's default when unset."""
        if key == "crossfade_duration" and self.crossfade_duration is not None:
            return self.crossfade_duration
        return default


class FakeStreams:
    """Minimal streams controller stand-in for the program bus."""

    def __init__(self) -> None:
        """Start with a live bus and no programs."""
        self.enabled = True
        self.active: set[str] = set()
        self.overlays: list[tuple[str, bytes, str, float, ProgramKind]] = []
        self.heads: list[float] = []
        self.cancelled: list[tuple[str, str, str, int | None]] = []
        self.ledger: list[SimpleNamespace] = []
        self.crossfade_mode = CrossfadeMode.DISABLED
        self.format = PCM_FORMAT
        self.result: ScheduleResult | None = None
        self.on_schedule: Callable[[], ScheduleResult | None] | None = None
        # seconds the fake producer runs ahead of the listener, or None without a reading
        self.lead_seconds: float | None = None
        # how often the pump asked for the queue's ledger
        self.ledger_reads = 0

    def program_bus_enabled(self) -> bool:
        """Return whether the experimental bus is switched on."""
        return self.enabled

    def program_bus_lead_seconds(self, queue_id: str) -> float | None:
        """Return how far the queue's producer runs ahead of what the listener heard."""
        return self.lead_seconds

    def program_bus_active(self, queue_id: str) -> bool:
        """Return whether the queue has a live flow session."""
        return queue_id in self.active

    def program_bus_format(self, queue_id: str) -> AudioFormat | None:
        """Return the format of the queue's live flow session."""
        return self.format if queue_id in self.active else None

    def get_crossfade_mode(self, queue: FakeQueue) -> CrossfadeMode:
        """Return the queue's effective crossfade mode."""
        return self.crossfade_mode

    def schedule_overlay(
        self,
        queue_id: str,
        pcm: bytes,
        *,
        label: str,
        start_in_seconds: float = 0.0,
        priority: Any = None,
        kind: ProgramKind = ProgramKind.TERM,
        head_seconds: float = 0.0,
    ) -> ScheduleResult:
        """Record a placement and answer with the scripted result."""
        if queue_id not in self.active:
            return ScheduleResult(
                scheduled=False,
                anchor_frame=0,
                frames=0,
                late_by_seconds=0.0,
                lead_shortfall_seconds=0.0,
                reason=ScheduleReason.NO_BUS,
            )
        self.overlays.append((queue_id, pcm, label, start_in_seconds, kind))
        self.heads.append(head_seconds)
        answer = self.on_schedule() if self.on_schedule is not None else self.result
        if answer is not None:
            return answer
        return ScheduleResult(
            scheduled=True,
            anchor_frame=round(start_in_seconds * PCM_FORMAT.sample_rate),
            frames=len(pcm) // 4,
            late_by_seconds=0.0,
            lead_shortfall_seconds=0.0,
            reason=None,
        )

    def overlay_state(self, queue_id: str) -> list[SimpleNamespace]:
        """Return the queue's ledger rows."""
        self.ledger_reads += 1
        return list(self.ledger)

    def cancel_overlay(
        self, queue_id: str, label: str, reason: str, *, start_frame: int | None = None
    ) -> bool:
        """Record a cancellation and report it as withdrawn."""
        self.cancelled.append((queue_id, label, reason, start_frame))
        return True


class FakeMass:
    """Minimal MusicAssistant stand-in for the term engine."""

    def __init__(self, queue: FakeQueue) -> None:
        """Wire the fakes the engine talks to."""
        self.player_queues = FakePlayerQueues(queue)
        self.streams = FakeStreams()
        self.config = FakeConfig()
        self.timers: list[tuple[float, Callable[..., Any], tuple[Any, ...], str | None]] = []
        self.tasks: list[asyncio.Task[Any]] = []
        self._timers_by_id: dict[str, tuple[Any, ...]] = {}
        self._tasks_by_id: dict[str, asyncio.Task[Any]] = {}

    def call_later(
        self, delay: float, target: Callable[..., Any], *args: Any, task_id: str | None = None
    ) -> None:
        """Record a delayed call, cancelling an earlier one with the same id like mass does."""
        if task_id and (existing := self._timers_by_id.get(task_id)) is not None:
            if existing in self.timers:
                self.timers.remove(existing)
        entry = (delay, target, args, task_id)
        self.timers.append(entry)
        if task_id:
            self._timers_by_id[task_id] = entry

    def create_task(
        self, target: Any, task_id: str | None = None, abort_existing: bool = False
    ) -> asyncio.Task[Any]:
        """Run a coroutine as a task, aborting a live one with the same id when asked."""
        if task_id and (existing := self._tasks_by_id.get(task_id)) and not existing.done():
            if abort_existing:
                existing.cancel()
            else:
                target.close()
                return existing
        task = asyncio.ensure_future(target)
        self.tasks.append(task)
        if task_id:
            self._tasks_by_id[task_id] = task
        return task

    def fire(self, task_id: str) -> asyncio.Task[Any]:
        """Fire the timer with the given id the way mass does."""
        entry = self._timers_by_id.pop(task_id)
        if entry in self.timers:
            self.timers.remove(entry)
        _delay, target, args, _task_id = entry
        return self.create_task(target(*args), task_id=task_id, abort_existing=True)


class TermEngineHarness(AIRadioTermEngineMixin):
    """Minimal harness around the term engine mixin."""

    instance_id = "ai_radio_test"

    def __init__(self, queue: FakeQueue) -> None:
        """Initialize the engine with one playing queue."""
        self.logger = logging.getLogger("test_term_engine")
        self.mass = FakeMass(queue)
        self._dj_queues = {
            queue.queue_id: DJQueueState(
                queue_id=queue.queue_id, host_id="rick", dj_session_id="dj1", ready=True
            )
        }
        self.rendered: list[tuple[str, str]] = []
        # every replan pass the engine asked for
        self.replanned: list[str] = []
        self.fail_render = False
        self.during_render: Callable[[], None] | None = None
        self.rendered_pcm = _pcm()
        # a render that waits here lets a test inspect an in-flight pump
        self.render_gate: asyncio.Event | None = None
        self.render_started = asyncio.Event()
        self.mass.streams.active.add(queue.queue_id)

    async def _render_mixed_program(
        self,
        queue_id: str,
        program: dict[str, Any],
        sections: list[PlannedSection],
        pcm_format: AudioFormat,
    ) -> bytes:
        """Record the render and answer with stand-in audio, or fail as an engine does."""
        self.rendered.extend(
            (str(program.get("host_id") or ""), section.section_id) for section in sections
        )
        if self.during_render is not None:
            self.during_render()
        self.render_started.set()
        if self.render_gate is not None:
            await self.render_gate.wait()
        if self.fail_render:
            raise RuntimeError("engine down")
        return self.rendered_pcm

    def _dj_queue_items(self, queue_id: str) -> list[FakeQueueItem]:
        """Return the queue's items."""
        return self.mass.player_queues.items(queue_id)

    @staticmethod
    def _queue_plays(queue: FakeQueue) -> bool:
        """Return whether the queue is playing."""
        return not queue.ended and queue.state == PlaybackState.PLAYING

    def _schedule_replan(self, queue_id: str) -> None:
        """Record a replan request instead of running the pass behind it."""
        self.replanned.append(queue_id)


def _window(count: int = 4) -> list[dict[str, Any]]:
    """Return the window tracks a planning pass plans against."""
    return [{"item_id": f"i{index}", "available": True, "duration": 200} for index in range(count)]


def test_a_planned_gap_names_its_track_and_its_successor() -> None:
    """The term speaks over the end of the track before the gap it was planned for."""
    plans = term_boundary_plans([_section_at(2)], _window(), None, {})

    assert len(plans) == 1
    assert plans[0].queue_item_id == "i1"
    assert plans[0].successor_id == "i2"


def test_an_outro_at_the_queue_end_has_no_successor() -> None:
    """The end of the window is a boundary without a track after it."""
    plans = term_boundary_plans([_section_at(4)], _window(4), None, {})

    assert len(plans) == 1
    assert plans[0].queue_item_id == "i3"
    assert plans[0].successor_id is None


def test_a_boundary_that_cannot_play_gets_no_plan_and_no_guard_event() -> None:
    """A track that cannot play is nothing to speak over, and its section keeps no history."""
    window = _window()
    window[2]["available"] = False
    section = _section_at(2)
    section.history_events = [("transition", (3, 12.0))]
    history = {"transition": [(3, 12.0)]}

    plans = term_boundary_plans([section], window, None, history)

    assert plans == []
    assert history["transition"] == []


def test_a_section_before_the_window_start_gets_no_plan() -> None:
    """A break in front of the first window track has no item to speak over."""
    section = _section_at(0)
    section.history_events = [("transition", (0, 0.0))]
    history = {"transition": [(0, 0.0)]}

    assert term_boundary_plans([section], _window(), None, history) == []
    assert history["transition"] == []


def test_a_boundary_past_the_horizon_keeps_no_plan_and_no_guard_event() -> None:
    """A gap the bounded store cannot hold waits, and its section spends nothing meanwhile."""
    window = _window(6)
    near = _section_at(1)
    far = _section_at(5)
    near.history_events = [("transition", (2, 0.0))]
    far.history_events = [("transition", (6, 0.0))]
    history = {"transition": [(2, 0.0), (6, 0.0)]}

    plans = term_boundary_plans([near, far], window, None, history)
    kept = term_plans_in_horizon(plans, {("i0", "i1")}, history)

    assert [plan.successor_id for plan in kept] == ["i1"]
    # the deferred boundary leaves no guard weight behind for the pass that decides it
    assert history["transition"] == [(2, 0.0)]


def test_an_outro_is_never_past_the_horizon() -> None:
    """The end of the show belongs after the last song, so no horizon can hold it back."""
    window = _window(4)
    plans = term_boundary_plans([_section_at(1), _section_at(4)], window, None, {})

    kept = term_plans_in_horizon(plans, set(), {})

    assert [plan.successor_id for plan in kept] == [None]


def test_the_running_boundary_names_the_item_and_what_follows_it() -> None:
    """The boundary the pump looks for is the end of the item that plays now."""
    items: Any = [FakeQueueItem(queue_item_id="a"), FakeQueueItem(queue_item_id="b")]

    boundary = running_boundary(items, "a")

    assert boundary is not None
    assert (boundary.index, boundary.item_id, boundary.successor_id) == (0, "a", "b")


def test_the_running_boundary_skips_the_dj_clips_behind_the_running_item() -> None:
    """A clip spliced at a boundary is not the item the boundary hands over to."""
    items: Any = [
        FakeQueueItem(queue_item_id="a"),
        FakeQueueItem(queue_item_id="clip", extra_attributes={ATTR_QUEUE_DJ: True}),
        FakeQueueItem(queue_item_id="b"),
    ]

    boundary = running_boundary(items, "a")

    assert boundary is not None
    assert boundary.successor_id == "b"


def test_a_running_item_outside_the_queue_has_no_boundary() -> None:
    """An item the queue does not list is no boundary to plan or time."""
    items: Any = [FakeQueueItem(queue_item_id="a")]

    assert running_boundary(items, "gone") is None
    assert running_boundary(items, None) is None


def _section_at(index: int, section_id: str = "transition") -> PlannedSection:
    """Return a between-songs section planned in front of the given window index."""
    section = _section(section_id)
    section.insert_at_index = index
    return section


def _section(
    section_id: str = "transition", *, allow_pre: bool = False, allow_post: bool = False
) -> PlannedSection:
    """Return a planned section of the given kind."""
    return PlannedSection(
        order=0,
        clip_id="",
        section_id=section_id,
        section_name=section_id,
        when="between_songs",
        insert_at_index=0,
        prompt="say something",
        max_chars=200,
        web_search_mode="disabled",
        allow_pre=allow_pre,
        allow_post=allow_post,
    )


def _plan(
    queue_item_id: str = "a",
    successor_id: str | None = "b",
    sections: tuple[PlannedSection, ...] | None = None,
) -> TermBoundaryPlan:
    """Return a boundary plan for the given term."""
    return TermBoundaryPlan(
        queue_item_id=queue_item_id,
        successor_id=successor_id,
        sections=sections or (_section(),),
    )


def _harness(
    *,
    durations: tuple[int | None, ...] = (200, 200, 200),
    current: str = "a",
    elapsed: float = 0.0,
    state: PlaybackState = PlaybackState.PLAYING,
) -> TermEngineHarness:
    """Return a playing queue whose first item is `remaining` seconds from its end."""
    ids = [chr(ord("a") + index) for index in range(len(durations))]
    items = [
        FakeQueueItem(queue_item_id=item_id, duration=duration)
        for item_id, duration in zip(ids, durations, strict=True)
    ]
    return TermEngineHarness(
        FakeQueue(queue_id="queue-1", items=items, current=current, elapsed=elapsed, state=state)
    )


def _terms(harness: TermEngineHarness) -> list[Any]:
    """Return the terms the harness's queue holds."""
    return list(harness._term_store("queue-1").terms())


def _decide(
    harness: TermEngineHarness, *plans: TermBoundaryPlan, head_seconds: float | None = None
) -> None:
    """Decide the given boundaries for the harness's queue, with the host's lead if given."""
    program: dict[str, Any] = {"host_id": "rick"}
    if head_seconds is not None:
        program["term_head_seconds"] = head_seconds
    harness._decide_terms("queue-1", program, list(plans) or [_plan()])


def _fast_renders(harness: TermEngineHarness) -> None:
    """Settle the render estimate on its floor, so a nearby boundary is still reachable."""
    for _ in range(10):
        harness._term_estimate("queue-1").record(0.5)


async def test_a_term_beyond_its_budget_is_not_rendered_yet() -> None:
    """A term is only rendered once its boundary is within the render budget."""
    harness = _harness(elapsed=120.0)
    _decide(harness)

    await harness._pump_terms("queue-1")

    assert harness.rendered == []
    assert harness.mass.streams.overlays == []
    assert harness._term_store("queue-1").terms()[0].state is TermState.DECIDED
    # the pass that renders it is booked for the moment the budget window opens: the
    # bus's minimum lead and the tolerance are part of that window too
    assert harness.mass.timers
    delay, _target, _args, task_id = harness.mass.timers[-1]
    # the budget window is 71s out, but a pass that far ahead would serve a lead the bus
    # may have outgrown since, so the wait is capped and the window read again then
    window = 0.0 + LEAD_MIN_LEAD_SECONDS + 6.0 + 1.0
    assert 80.0 - window > TERM_PUMP_MAX_SECONDS
    assert delay == pytest.approx(TERM_PUMP_MAX_SECONDS)
    # every booking has its own id, so it can never abort another pass
    assert task_id.startswith("ai_radio_term_pump_queue-1")


async def test_a_decided_term_logs_its_state_and_its_timing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A pass that decides a term leaves its state, its position and its timing in the log."""
    caplog.set_level(logging.DEBUG, logger="test_term_engine")
    harness = _harness(elapsed=120.0)
    _decide(harness)

    await harness._pump_terms("queue-1")

    line = next(
        record.getMessage() for record in caplog.records if record.getMessage().startswith("TERM ")
    )
    assert "queue=queue-1" in line
    assert "item=a" in line
    assert "state=decided" in line
    assert "pos=current" in line
    assert "remaining=80.00" in line
    assert "head=0.00" in line
    assert "estimate=6.00" in line
    assert "lead=2.00" in line
    assert "due=False" in line
    assert "late=False" in line


async def test_a_due_term_renders_and_is_placed_at_its_boundary() -> None:
    """A term close enough is rendered and mixed in remaining minus head seconds from now."""
    harness = _harness(elapsed=192.0)
    _decide(harness, _plan(sections=(_section(allow_post=True),)))

    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    assert term.state is TermState.SCHEDULED
    assert harness.rendered == [("rick", "transition")]
    queue_id, pcm, _label, start_in_seconds, kind = harness.mass.streams.overlays[0]
    assert queue_id == "queue-1"
    assert pcm
    assert start_in_seconds == pytest.approx(8.0 - TERM_START_BIAS_SECONDS)
    assert kind is ProgramKind.TERM
    assert term.pcm_len == len(pcm)
    assert term.scheduled_frame == round(7.75 * PCM_FORMAT.sample_rate)


async def test_a_term_tells_the_bus_the_head_it_was_planned_with() -> None:
    """The audio path shows the head a term was planned with, so the bus has to be told it."""
    harness = _harness(elapsed=190.0)
    harness.rendered_pcm = _pcm(3.0)
    _decide(harness, _plan(sections=(_section(allow_pre=True, allow_post=True),)))

    await harness._pump_terms("queue-1")

    assert harness.mass.streams.heads == [pytest.approx(TERM_HEAD_DEFAULT_SECONDS)]


async def test_a_clip_shorter_than_the_lead_starts_as_early_as_it_lasts() -> None:
    """A break that does not last a whole lead still starts where the host wants it."""
    harness = _harness(elapsed=190.0)
    harness.rendered_pcm = _pcm(1.5)
    _decide(harness, _plan(sections=(_section(allow_pre=True, allow_post=True),)))

    await harness._pump_terms("queue-1")

    assert harness.mass.streams.heads == [pytest.approx(1.5)]
    assert harness.mass.streams.overlays[0][3] == pytest.approx(
        10.0 - 1.5 - TERM_START_BIAS_SECONDS
    )


async def test_a_term_is_placed_the_lead_its_host_set_before_the_boundary() -> None:
    """The lead the host was configured with is both the decision and the placement."""
    harness = _harness(elapsed=186.0)
    harness.rendered_pcm = _pcm(8.0)
    _decide(
        harness,
        _plan(sections=(_section(allow_pre=True, allow_post=True),)),
        head_seconds=5.0,
    )

    term = _terms(harness)[0]
    assert term.head_seconds == pytest.approx(5.0)

    await harness._pump_terms("queue-1")

    assert harness.mass.streams.heads == [pytest.approx(5.0)]
    assert harness.mass.streams.overlays[0][3] == pytest.approx(
        14.0 - 5.0 - TERM_START_BIAS_SECONDS
    )


async def test_a_lead_of_zero_places_the_term_on_the_boundary() -> None:
    """A host that wants no lead lets the break start with the crossfade."""
    harness = _harness(elapsed=192.0)
    harness.rendered_pcm = _pcm(8.0)
    _decide(
        harness,
        _plan(sections=(_section(allow_pre=True, allow_post=True),)),
        head_seconds=0.0,
    )

    await harness._pump_terms("queue-1")

    assert harness.mass.streams.heads == [pytest.approx(0.0)]
    assert harness.mass.streams.overlays[0][3] == pytest.approx(8.0 - TERM_START_BIAS_SECONDS)


async def test_a_section_that_may_lead_speaks_before_the_boundary() -> None:
    """A break whose section allows a pre is decided with that pre."""
    harness = _harness()
    _decide(harness, _plan(sections=(_section(allow_pre=True),)))

    assert _terms(harness)[0].allow_pre is True
    assert _terms(harness)[0].head_seconds == pytest.approx(TERM_HEAD_DEFAULT_SECONDS)


async def test_a_section_without_pre_has_no_head_whatever_the_host_set() -> None:
    """A break that may not start over the outgoing tail is placed on the boundary."""
    harness = _harness(elapsed=192.0)
    harness.rendered_pcm = _pcm(8.0)
    _decide(
        harness,
        _plan(sections=(_section(allow_post=True),)),
        head_seconds=12.0,
    )

    assert _terms(harness)[0].allow_pre is False
    assert _terms(harness)[0].head_seconds == 0.0

    await harness._pump_terms("queue-1")

    assert harness.mass.streams.heads == [pytest.approx(0.0)]
    assert harness.mass.streams.overlays[0][3] == pytest.approx(8.0 - TERM_START_BIAS_SECONDS)


async def test_a_posting_term_starts_its_lead_before_the_boundary_and_runs_on() -> None:
    """A break that may carry over the next intro keeps the whole lead its host set."""
    harness = _harness(elapsed=190.0)
    harness.rendered_pcm = _pcm(8.62)
    _decide(harness, _plan(sections=(_section(allow_pre=True, allow_post=True),)))

    await harness._pump_terms("queue-1")

    term = _terms(harness)[0]
    assert term.state is TermState.SCHEDULED
    start_in_seconds = harness.mass.streams.overlays[0][3]
    # frame-quantised clip length, so the placement is exact to the sample rate
    assert start_in_seconds == pytest.approx(
        10.0 - TERM_HEAD_DEFAULT_SECONDS - TERM_START_BIAS_SECONDS, abs=1e-3
    )
    # the rest of the clip plays on over the next record, which the post allows
    assert start_in_seconds + 8.62 > 10.0 + TERM_HEAD_DEFAULT_SECONDS
    assert harness.mass.streams.heads == [pytest.approx(TERM_HEAD_DEFAULT_SECONDS, abs=1e-3)]


async def test_a_term_the_producer_runs_ahead_of_is_rendered_and_placed() -> None:
    """The producer's lead is part of the window: without it the term lands past its boundary."""
    harness = _harness(elapsed=170.0)
    bus: Any = harness.mass.streams
    bus.lead_seconds = PRODUCER_LEAD_SECONDS
    harness.rendered_pcm = _pcm(12.0)
    _decide(
        harness,
        _plan(sections=(_section(allow_pre=True),)),
        head_seconds=TALK_OVER_MAX_PRE,
    )

    await harness._pump_terms("queue-1")

    term = _terms(harness)[0]
    assert term.state is TermState.SCHEDULED
    assert term.dropped_reason is None
    # 30s to the boundary, a 12s head and the bias: the delay stays above the lead
    start_in_seconds = bus.overlays[0][3]
    assert start_in_seconds == pytest.approx(30.0 - TALK_OVER_MAX_PRE)
    assert start_in_seconds >= PRODUCER_LEAD_SECONDS


async def test_a_clip_that_took_its_render_still_lands_after_the_lead_grew() -> None:
    """The live drop: the render is paid, so only the start the clip can still reach decides."""
    harness = _harness(elapsed=154.39)
    bus: Any = harness.mass.streams
    bus.lead_seconds = 37.01
    harness.rendered_pcm = _pcm(12.0)
    _decide(harness, _plan(sections=(_section(allow_pre=True, allow_post=True),)))

    def _render_elapsed() -> None:
        # 5.81s of playback pass while the engine speaks, and the producer's buffer grew
        # by half a second with them
        harness.mass.player_queues.queue.items[0].duration -= 5.81
        bus.lead_seconds = 37.51

    harness.during_render = _render_elapsed
    await harness._pump_terms("queue-1")

    term = _terms(harness)[0]
    assert term.state is TermState.SCHEDULED
    assert term.dropped_reason is None
    # 45.61s to the boundary at the decision, 39.8s when the clip is placed: the head and
    # the bias come off that, and the start is still a hair ahead of the produced frames
    start_in_seconds = bus.overlays[0][3]
    assert start_in_seconds == pytest.approx(
        39.8 - TERM_HEAD_DEFAULT_SECONDS - TERM_START_BIAS_SECONDS
    )
    assert start_in_seconds >= bus.lead_seconds


async def test_a_boundary_inside_the_producer_lead_is_dropped_without_a_render() -> None:
    """A boundary the producer already reached is not worth an engine call."""
    harness = _harness(elapsed=183.0)
    bus: Any = harness.mass.streams
    bus.lead_seconds = PRODUCER_LEAD_SECONDS
    _decide(harness, _plan(sections=(_section(allow_pre=True),)))

    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    assert term.state is TermState.DROPPED
    assert term.dropped_reason == "too_late"
    assert harness.rendered == []


async def test_the_pump_waits_the_producer_lead_out_before_it_renders() -> None:
    """A pass is booked for the moment the window opens, the lead included."""
    harness = _harness(elapsed=140.0)
    bus: Any = harness.mass.streams
    bus.lead_seconds = PRODUCER_LEAD_SECONDS
    _decide(
        harness,
        _plan(sections=(_section(allow_pre=True),)),
        head_seconds=TALK_OVER_MAX_PRE,
    )

    await harness._pump_terms("queue-1")

    assert harness.rendered == []
    delay, _target, _args, _task_id = harness.mass.timers[-1]
    assert delay == pytest.approx(
        60.0
        - (TALK_OVER_MAX_PRE + PRODUCER_LEAD_SECONDS + RENDER_ESTIMATE_SECONDS)
        - TERM_SCHEDULE_TOLERANCE_SECONDS
        - TERM_RENDER_MARGIN_SECONDS
    )


@pytest.mark.parametrize(
    "mode",
    [
        CrossfadeMode.DISABLED,
        CrossfadeMode.STANDARD_CROSSFADE,
        CrossfadeMode.SMART_CROSSFADE,
        CrossfadeMode.SOURCE,
    ],
)
async def test_a_track_boundary_is_its_end_in_every_crossfade_mode(mode: CrossfadeMode) -> None:
    """Whatever the player is set to fade, the term is timed to the track's own end."""
    harness = _harness(elapsed=190.0)
    harness.mass.streams.crossfade_mode = mode
    harness.mass.config.crossfade_duration = 8
    item = harness.mass.player_queues.queue.current_item

    assert harness._term_fade_seconds(item) == 0.0


async def test_a_standard_crossfade_does_not_move_the_boundary_earlier() -> None:
    """A player that will fade still ends the item where the item's duration says it ends."""
    harness = _harness(elapsed=184.0)
    harness.mass.streams.crossfade_mode = CrossfadeMode.STANDARD_CROSSFADE
    harness.mass.config.crossfade_duration = 8
    _decide(harness, _plan(sections=(_section(allow_post=True),)))

    await harness._pump_terms("queue-1")

    # 16 s of track left: the configured 8 s of crossfade are not taken off them
    assert harness.mass.streams.overlays[0][3] == pytest.approx(16.0 - TERM_START_BIAS_SECONDS)


async def test_a_smart_fade_boundary_is_the_end_of_the_track(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A fade the player only plans at the handover is no boundary a term can lean on."""
    caplog.set_level(logging.DEBUG, logger="test_term_engine")
    harness = _harness(durations=(12, 200, 200), elapsed=0.0)
    harness.mass.streams.crossfade_mode = CrossfadeMode.SMART_CROSSFADE
    harness.mass.config.crossfade_duration = 8
    _decide(harness, _plan(sections=(_section(allow_post=True),)))

    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    assert term.state is TermState.SCHEDULED
    # the 12 s track ends 12 s after its start, so the break sits on that end
    assert "remaining=12.00" in caplog.text
    assert harness.mass.streams.overlays[0][3] == pytest.approx(12.0 - TERM_START_BIAS_SECONDS)
    # and no configured duration stands in for the fade the player never promised
    assert "smart fade" not in caplog.text


async def test_a_missing_crossfade_length_is_no_boundary_reading() -> None:
    """An install that stored no crossfade length has nothing to subtract either."""
    harness = _harness(durations=(12, 200, 200), elapsed=0.0)
    harness.mass.streams.crossfade_mode = CrossfadeMode.SMART_CROSSFADE
    harness.mass.config.crossfade_duration = None
    item = harness.mass.player_queues.queue.current_item

    assert harness._term_fade_seconds(item) == 0.0
    _decide(harness, _plan(sections=(_section(allow_post=True),)))

    await harness._pump_terms("queue-1")

    assert harness._term_store("queue-1").terms()[0].state is TermState.SCHEDULED
    assert harness.mass.streams.overlays[0][3] == pytest.approx(12.0 - TERM_START_BIAS_SECONDS)


async def test_the_live_boundary_is_the_track_end_not_the_configured_crossfade(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The heard report: an 8 s crossfade setting must not move the break 8 s early."""
    caplog.set_level(logging.DEBUG, logger="test_term_engine")
    harness = _harness(durations=(268, 200, 200), elapsed=200.0)
    harness.mass.streams.crossfade_mode = CrossfadeMode.SMART_CROSSFADE
    harness.mass.config.crossfade_duration = 8
    _decide(harness, _plan(sections=(_section(allow_post=True),)))

    await harness._pump_terms("queue-1")

    # 268 s track at 200 s: 68 s left, not the 60 a fade estimate leaves
    assert "remaining=68.00" in caplog.text


async def test_a_break_starts_its_lead_before_the_end_and_runs_past_it() -> None:
    """The host's lead alone decides how far a break reaches back over the track."""
    harness = _harness(durations=(268, 200, 200), elapsed=262.0)
    harness.mass.streams.crossfade_mode = CrossfadeMode.SMART_CROSSFADE
    harness.mass.config.crossfade_duration = 8
    harness.rendered_pcm = _pcm(8.0)
    _fast_renders(harness)
    _decide(
        harness,
        _plan(sections=(_section(allow_pre=True, allow_post=True),)),
        head_seconds=2.0,
    )

    await harness._pump_terms("queue-1")

    # 6 s of track left, 2 s of them played by the break's head
    start_in_seconds = harness.mass.streams.overlays[0][3]
    assert start_in_seconds == pytest.approx(6.0 - 2.0 - TERM_START_BIAS_SECONDS)
    # the rest of the 8 s clip plays on over the next record
    assert start_in_seconds + 8.0 > 6.0


async def test_a_boundary_without_a_media_type_is_unknown() -> None:
    """A queue item that knows no media type says nothing about where its boundary is."""
    harness = _harness(elapsed=190.0)
    harness.mass.streams.crossfade_mode = CrossfadeMode.SMART_CROSSFADE
    harness.mass.player_queues.queue.items[0].media_type = MediaType.UNKNOWN
    assert harness._term_fade_seconds(harness.mass.player_queues.queue.current_item) is None
    _decide(harness)

    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    assert term.state is TermState.DROPPED
    assert term.dropped_reason == "unknown_boundary"
    assert harness.rendered == []


async def test_a_non_track_item_keeps_its_own_end() -> None:
    """A podcast or audiobook ends where its duration says, with no fade taken off it."""
    harness = _harness(elapsed=190.0)
    harness.mass.streams.crossfade_mode = CrossfadeMode.STANDARD_CROSSFADE
    harness.mass.config.crossfade_duration = 8
    queue = harness.mass.player_queues.queue
    queue.items[0].media_type = MediaType.PODCAST

    assert harness._term_fade_seconds(queue.current_item) == 0.0


async def test_an_item_without_a_duration_has_no_term() -> None:
    """A dynamic source that reports no duration is never timed."""
    harness = _harness(durations=(None, 200, 200))
    _decide(harness)

    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    assert term.dropped_reason == "no_duration"
    assert harness.rendered == []


async def test_a_boundary_too_close_to_render_is_dropped_without_a_render() -> None:
    """The render budget keeps an engine call from being spent on a lost boundary."""
    harness = _harness(elapsed=198.0)
    _decide(harness)

    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    assert term.state is TermState.DROPPED
    assert term.dropped_reason == "too_late"
    assert harness.rendered == []


async def test_a_placement_that_missed_the_boundary_is_dropped_and_cancelled() -> None:
    """A term the bus reports as late is taken back instead of airing somewhere else."""
    harness = _harness(elapsed=192.0)
    harness.mass.streams.result = ScheduleResult(
        scheduled=True,
        anchor_frame=48000,
        frames=48000,
        late_by_seconds=TERM_SCHEDULE_TOLERANCE_SECONDS + 0.5,
        lead_shortfall_seconds=0.0,
        reason=None,
    )
    _decide(harness)

    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    assert term.dropped_reason == "too_late"
    assert harness.mass.streams.cancelled == [("queue-1", "AI DJ transition", "too_late", 48000)]


async def test_a_refused_placement_keeps_the_reason_the_api_gave() -> None:
    """A window the bus already gave away is reported as such."""
    harness = _harness(elapsed=192.0)
    harness.mass.streams.result = ScheduleResult(
        scheduled=False,
        anchor_frame=0,
        frames=0,
        late_by_seconds=0.0,
        lead_shortfall_seconds=0.0,
        reason=ScheduleReason.WINDOW_TAKEN,
    )
    _decide(harness)

    await harness._pump_terms("queue-1")

    assert harness._term_store("queue-1").terms()[0].dropped_reason == "window_taken"


async def test_a_placement_that_took_another_window_is_dropped() -> None:
    """A term the bus had to move past another program does not air on a false boundary."""
    harness = _harness(elapsed=192.0)
    harness.mass.streams.result = ScheduleResult(
        scheduled=True,
        anchor_frame=96000,
        frames=144000,
        late_by_seconds=0.0,
        lead_shortfall_seconds=0.0,
        reason=ScheduleReason.WINDOW_TAKEN,
    )
    _decide(harness)

    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    assert term.dropped_reason == "window_taken"
    assert harness.mass.streams.cancelled == [
        ("queue-1", "AI DJ transition", "window_taken", 96000)
    ]


async def test_a_placement_with_another_reason_stays_placed() -> None:
    """A reason that does not mean the window was moved is not a reason to take the term back."""
    harness = _harness(elapsed=192.0)
    harness.mass.streams.result = ScheduleResult(
        scheduled=True,
        anchor_frame=96000,
        frames=144000,
        late_by_seconds=0.0,
        lead_shortfall_seconds=0.0,
        reason=ScheduleReason.HEAD_TRUNCATED,
    )
    _decide(harness)

    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    assert term.state is TermState.SCHEDULED
    assert harness.mass.streams.cancelled == []


async def test_the_deadline_ends_where_the_head_the_bus_cut_ends() -> None:
    """
    A head the bus had to cut is not counted twice: the lateness already covers it.

    The placement is built by hand: ``place_program`` raises the anchor to the produced frame
    and folds the lateness into ``late_by_seconds``, so the bus never has to cut the front of
    a program placed through it. What is pinned here is the term's arithmetic on such a
    placement, not the producer that would make one.
    """
    harness = _harness(elapsed=192.0)
    # the real bus, with half a second of the flow already produced past the frame the
    # term was planned at, so the placement drops that much of the front of the term
    program_bus = FlowProgramBus(PCM_FORMAT, logging.getLogger("test_term_engine"))
    await program_bus.feed(_pcm(0.5))
    placement = program_bus.place(_pcm(RENDER_SECONDS), 0, label=OVERLAY_LABEL_TRANSITION)
    assert placement is not None
    # three seconds were offered, half a second of them was already produced and is cut
    assert placement.frames == int(2.5 * PCM_FORMAT.sample_rate)
    harness.mass.streams.result = ScheduleResult(
        scheduled=True,
        anchor_frame=placement.start_frame,
        frames=placement.frames,
        # the head is cut because the term starts half a second past its boundary
        late_by_seconds=0.5,
        lead_shortfall_seconds=0.0,
        reason=None,
    )
    _decide(harness, _plan(sections=(_section(allow_post=True),)))

    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    assert term.state is TermState.SCHEDULED
    # it starts half a second late and plays the two and a half the bus kept, so it ends
    # where a placement that was not late would: the planned start plus the whole render
    assert term.seconds_left == pytest.approx(
        7.75 + RENDER_SECONDS + TERM_SCHEDULE_TOLERANCE_SECONDS
    )


async def test_the_deadline_includes_the_lead_shortfall() -> None:
    """A start the bus pushed back by its lead ends that much later."""
    harness = _harness(elapsed=192.0)
    harness.mass.streams.result = ScheduleResult(
        scheduled=True,
        anchor_frame=0,
        frames=10 * PCM_FORMAT.sample_rate,
        late_by_seconds=0.0,
        lead_shortfall_seconds=1.0,
        reason=None,
    )
    _decide(harness, _plan(sections=(_section(allow_post=True),)))

    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    assert term.seconds_left == pytest.approx(7.75 + 1.0 + 10.0 + TERM_SCHEDULE_TOLERANCE_SECONDS)


async def test_the_deadline_counts_stream_time_not_wall_time() -> None:
    """The deadline counts what was heard: a pause in between is not charged to it."""
    harness = _harness(elapsed=192.0)
    _decide(harness)
    await harness._pump_terms("queue-1")
    term = harness._term_store("queue-1").terms()[0]
    before = term.seconds_left
    # two wall seconds passed, but only one of them played: the media moved 1.25s at 1.25x
    queue = harness.mass.player_queues.queue
    queue.playback_speed = 1.25
    queue.elapsed += 1.25
    term.last_seen_at = asyncio.get_running_loop().time() - 2.0

    await harness._pump_terms("queue-1")

    assert term.seconds_left is not None
    assert before is not None
    assert term.seconds_left == pytest.approx(before - 1.0)


async def test_a_failed_render_does_not_pull_the_render_estimate() -> None:
    """A render that fails returns at once; that says nothing about how long one takes."""
    harness = _harness(elapsed=192.0)
    harness.fail_render = True
    _decide(harness)
    before = harness._term_estimate("queue-1").seconds

    await harness._pump_terms("queue-1")

    estimate = harness._term_estimate("queue-1")
    assert estimate.seconds == before
    assert estimate.samples == 0


async def test_a_render_that_lands_measures_the_estimate() -> None:
    """A render that produced audio is what the estimate is made of."""
    harness = _harness(elapsed=192.0)
    _decide(harness)

    await harness._pump_terms("queue-1")

    assert harness._term_estimate("queue-1").samples == 1


async def test_a_render_failure_drops_the_term() -> None:
    """An engine that fails loses its term, never the queue."""
    harness = _harness(elapsed=192.0)
    harness.fail_render = True
    _decide(harness)

    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    assert term.state is TermState.DROPPED
    assert term.dropped_reason == "render_failed"
    assert harness.mass.streams.overlays == []


async def test_a_paused_queue_without_a_bus_keeps_its_term() -> None:
    """A DJ switched on into a pause waits for the play, bus or no bus."""
    harness = _harness(elapsed=192.0, state=PlaybackState.PAUSED)
    harness.mass.streams.active.clear()
    _decide(harness)

    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    assert term.state is TermState.DECIDED
    assert term.dropped_reason is None
    assert harness.rendered == []


async def test_a_paused_queue_renders_nothing_and_keeps_its_term() -> None:
    """A pause is not a lost boundary: the term waits for the resume."""
    harness = _harness(elapsed=192.0, state=PlaybackState.PAUSED)
    _decide(harness)

    await harness._pump_terms("queue-1")

    assert harness.rendered == []
    assert harness._term_store("queue-1").terms()[0].state is TermState.DECIDED


async def test_a_pause_during_the_render_returns_the_term_to_decided() -> None:
    """Audio that was rendered into a pause was never heard, so it is rendered again later."""
    harness = _harness(elapsed=192.0)
    harness.during_render = lambda: setattr(
        harness.mass.player_queues.queue, "state", PlaybackState.PAUSED
    )
    _decide(harness)

    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    assert term.state is TermState.DECIDED
    assert harness.mass.streams.overlays == []


async def test_a_pause_during_the_render_keeps_the_clip_for_the_resume() -> None:
    """A clip written into a pause is held, so the resume does not pay for it again."""
    harness = _harness(elapsed=192.0)
    queue = harness.mass.player_queues.queue
    harness.during_render = lambda: setattr(queue, "state", PlaybackState.PAUSED)
    _decide(harness)

    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    assert term.state is TermState.DECIDED
    assert term.rendered_pcm is not None
    assert len(harness.rendered) == 1

    queue.state = PlaybackState.PLAYING
    await harness._pump_terms("queue-1")

    assert len(harness.rendered) == 1
    assert term.state is TermState.SCHEDULED
    assert harness.mass.streams.overlays


async def test_a_held_clip_is_placed_at_the_track_end_in_a_smart_fade_queue() -> None:
    """A clip placed after the fact is timed to the boundary the decision was made on."""
    harness = _harness(elapsed=192.0)
    harness.mass.streams.crossfade_mode = CrossfadeMode.SMART_CROSSFADE
    harness.mass.config.crossfade_duration = 8
    queue = harness.mass.player_queues.queue
    harness.during_render = lambda: setattr(queue, "state", PlaybackState.PAUSED)
    _decide(harness, _plan(sections=(_section(allow_post=True),)))

    await harness._pump_terms("queue-1")
    assert harness._term_store("queue-1").terms()[0].rendered_pcm is not None

    queue.state = PlaybackState.PLAYING
    await harness._pump_terms("queue-1")

    # 8 s of track left, held through the pause: the clip lands on that end
    assert harness.mass.streams.overlays[0][3] == pytest.approx(8.0 - TERM_START_BIAS_SECONDS)


async def test_a_held_clip_is_dropped_when_the_bus_format_changed() -> None:
    """A clip in the old flow format can not be mixed into a new one."""
    harness = _harness(elapsed=192.0)
    queue = harness.mass.player_queues.queue
    harness.during_render = lambda: setattr(queue, "state", PlaybackState.PAUSED)
    _decide(harness)
    await harness._pump_terms("queue-1")
    queue.state = PlaybackState.PLAYING
    harness.mass.streams.format = AudioFormat(
        content_type=ContentType.PCM_F32LE, sample_rate=44100, bit_depth=32, channels=2
    )

    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    assert term.dropped_reason == "bus_replaced"
    assert harness.mass.streams.overlays == []


async def test_a_skip_during_the_render_drops_the_term() -> None:
    """The listener moved to the next item while the term was being written."""
    harness = _harness(elapsed=192.0)

    def _skip() -> None:
        harness.mass.player_queues.queue.current = "b"
        harness.mass.player_queues.queue.elapsed = 0.0

    harness.during_render = _skip
    _decide(harness)

    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    assert term.dropped_reason == "skipped"
    assert harness.mass.streams.overlays == []


async def test_a_seek_during_the_render_drops_the_term() -> None:
    """A seek invalidates the clip that was written for the old position."""
    harness = _harness(elapsed=192.0)

    def _seek() -> None:
        harness.mass.player_queues.queue.elapsed = 40.0

    harness.during_render = _seek
    _decide(harness)

    await harness._pump_terms("queue-1")

    assert harness._term_store("queue-1").terms()[0].dropped_reason == "seeked"
    assert harness.mass.streams.overlays == []


async def test_a_bus_that_disappeared_during_the_render_drops_the_term() -> None:
    """A session that ended while the term rendered can not take it."""
    harness = _harness(elapsed=192.0)
    harness.during_render = lambda: harness.mass.streams.active.clear()
    _decide(harness)

    await harness._pump_terms("queue-1")

    assert harness._term_store("queue-1").terms()[0].dropped_reason == "bus_replaced"
    assert harness.mass.streams.overlays == []


async def test_a_boundary_the_player_already_passed_is_dropped_as_skipped() -> None:
    """A term whose item is behind the playhead is never caught up."""
    harness = _harness(elapsed=0.0, current="b")
    _decide(harness)

    await harness._pump_terms("queue-1")

    assert harness._term_store("queue-1").terms()[0].dropped_reason == "skipped"


async def test_a_term_for_a_later_item_waits_its_turn() -> None:
    """A boundary further down the queue is not rendered while another item plays."""
    harness = _harness(elapsed=120.0)
    _decide(harness, _plan(queue_item_id="b", successor_id="c"))

    await harness._pump_terms("queue-1")

    assert harness.rendered == []
    assert harness._term_store("queue-1").terms()[0].state is TermState.DECIDED


async def test_a_seek_within_the_item_keeps_the_term() -> None:
    """A backward seek that leaves the boundary ahead only moves the timing."""
    harness = _harness(elapsed=192.0)
    _decide(harness)
    term = harness._term_store("queue-1").terms()[0]
    # a first pass remembers the position, the second one sees the seek
    term.last_elapsed = 190.0
    term.last_seen_at = asyncio.get_running_loop().time()
    harness.mass.player_queues.queue.elapsed = 100.0

    await harness._pump_terms("queue-1")

    assert term.state is TermState.DECIDED
    assert term.dropped_reason is None


async def test_a_seek_past_the_boundary_drops_the_term() -> None:
    """A seek over the boundary leaves nothing to speak over."""
    harness = _harness(elapsed=199.0)
    _decide(harness)
    term = harness._term_store("queue-1").terms()[0]
    term.last_elapsed = 10.0
    term.last_seen_at = asyncio.get_running_loop().time()
    harness.mass.player_queues.queue.elapsed = 199.0

    await harness._pump_terms("queue-1")

    assert term.dropped_reason == "seeked"


async def test_an_outro_without_a_successor_is_placed_at_the_end() -> None:
    """The end of the queue is a boundary like any other."""
    harness = _harness(durations=(200,), current="a", elapsed=192.0)
    _decide(
        harness,
        _plan(
            queue_item_id="a",
            successor_id=None,
            sections=(_section(allow_post=True),),
        ),
    )

    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    assert term.state is TermState.SCHEDULED
    assert harness.mass.streams.overlays[0][3] == pytest.approx(8.0 - TERM_START_BIAS_SECONDS)


async def test_an_outro_loses_its_boundary_when_the_queue_moves_on() -> None:
    """The end of the queue is no boundary any more once another item is queued behind it."""
    harness = _harness(durations=(200, 200), current="b", elapsed=0.0)
    _decide(harness, _plan(queue_item_id="a", successor_id=None))

    await harness._pump_terms("queue-1")

    assert harness._term_store("queue-1").terms()[0].dropped_reason == "skipped"


async def test_a_dropped_term_releases_its_guard_and_gap() -> None:
    """A break that never airs must not count in the guard, and its gap is open again."""
    harness = _harness(elapsed=190.0)
    harness.mass.streams.crossfade_mode = CrossfadeMode.SMART_CROSSFADE
    harness.mass.player_queues.queue.items[0].media_type = MediaType.UNKNOWN
    state = harness._dj_queues["queue-1"]
    section = _section()
    section.history_events = [("transition", (3, 12.0))]
    state.history = {"transition": [(3, 12.0)]}
    _decide(harness, _plan(sections=(section,)))

    await harness._pump_terms("queue-1")

    store = harness._term_store("queue-1")
    assert store.terms()[0].dropped_reason == "unknown_boundary"
    assert state.history.get("transition") == []
    # a boundary nothing can be timed on is closed for good, not opened again
    assert store.state(("a", "b")) is BoundaryState.UNREACHABLE


async def test_a_term_whose_successor_changed_is_dropped() -> None:
    """A queue reordered under the term makes its boundary a different one."""
    harness = _harness(durations=(200, 200, 200), current="a", elapsed=192.0)
    _decide(harness, _plan(queue_item_id="a", successor_id="c"))

    await harness._pump_terms("queue-1")

    assert harness._term_store("queue-1").terms()[0].dropped_reason == "no_successor"


async def test_an_undecided_term_needs_a_bus() -> None:
    """Without a live session the term is reported once and never airs."""
    harness = _harness(elapsed=192.0)
    harness.mass.streams.active.clear()
    _decide(harness)

    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    # the bus may appear with the play, so the first pass only starts the grace
    assert term.state is TermState.DECIDED
    term.bus_missing_since = asyncio.get_running_loop().time() - BUS_GRACE_SECONDS - 1

    await harness._pump_terms("queue-1")

    assert term.dropped_reason == "no_bus"
    assert harness.rendered == []


async def test_a_play_after_a_short_pause_gets_its_own_bus_grace() -> None:
    """A play/pause race before the grace runs out does not lose the boundary to it."""
    harness = _harness(elapsed=192.0)
    harness.mass.streams.active.clear()
    _decide(harness)
    await harness._pump_terms("queue-1")
    term = harness._term_store("queue-1").terms()[0]
    assert term.state is TermState.DECIDED

    # buffering or a short pause interrupts while the bus is still missing
    term.bus_missing_since = asyncio.get_running_loop().time() - BUS_GRACE_SECONDS - 1
    harness.mass.player_queues.queue.state = PlaybackState.PAUSED
    await harness._pump_terms("queue-1")
    assert term.bus_missing_since is None

    # the next play waits out its own grace instead of being written off at once
    harness.mass.player_queues.queue.state = PlaybackState.PLAYING
    await harness._pump_terms("queue-1")
    assert term.state is TermState.DECIDED
    assert term.dropped_reason is None

    harness.mass.streams.active.add("queue-1")
    await harness._pump_terms("queue-1")

    assert term.state is TermState.SCHEDULED
    assert len(harness.mass.streams.overlays) == 1


async def test_a_term_waits_for_a_bus_that_appears_after_the_resume() -> None:
    """A term switched on into a pause is spoken once the play's bus registers."""
    harness = _harness(elapsed=192.0, state=PlaybackState.PAUSED)
    harness.mass.streams.active.clear()
    _decide(harness)

    await harness._pump_terms("queue-1")
    assert harness._term_store("queue-1").terms()[0].state is TermState.DECIDED
    # the paused queue is looked at again on its own, without a queue event
    assert harness.mass.timers[-1][0] == pytest.approx(PAUSED_TERM_PUMP_SECONDS)

    harness.mass.player_queues.queue.state = PlaybackState.PLAYING
    await harness._pump_terms("queue-1")
    # the play's flow session is not up yet, so the term keeps waiting
    assert harness._term_store("queue-1").terms()[0].state is TermState.DECIDED

    harness.mass.streams.active.add("queue-1")
    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    assert term.state is TermState.SCHEDULED
    assert len(harness.mass.streams.overlays) == 1


async def test_a_bus_that_was_replaced_drops_the_pending_term() -> None:
    """A term that saw a bus and lost it reports the replacement."""
    harness = _harness(elapsed=120.0)
    _decide(harness)
    term = harness._term_store("queue-1").terms()[0]
    term.bus_seen = True
    harness.mass.streams.active.clear()

    await harness._pump_terms("queue-1")

    assert term.dropped_reason == "bus_replaced"


async def test_stopping_the_queue_ends_every_term() -> None:
    """A stopped playback takes its terms with it; a pause does not."""
    harness = _harness(elapsed=120.0)
    _decide(harness)
    harness.mass.player_queues.queue.state = PlaybackState.IDLE

    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    assert term.state is TermState.DROPPED
    assert term.dropped_reason == "flow_ended"


async def test_a_queue_played_to_its_end_ends_every_term() -> None:
    """An ended queue keeps its items, and ends its terms with them."""
    harness = _harness(elapsed=120.0)
    _decide(harness)
    harness.mass.player_queues.queue.ended = True

    await harness._pump_terms("queue-1")

    assert harness._term_store("queue-1").terms()[0].dropped_reason == "flow_ended"


async def test_a_scheduled_term_is_aired_when_the_ledger_says_so() -> None:
    """The streams ledger is the one truth about what was heard."""
    harness = _harness(elapsed=192.0)
    _decide(harness)
    await harness._pump_terms("queue-1")
    term = harness._term_store("queue-1").terms()[0]
    harness.mass.streams.ledger = [
        SimpleNamespace(
            label="AI DJ transition",
            state=OverlayProgramState.AIRED,
            start_frame=term.scheduled_frame,
            frames=1,
            reason=None,
        )
    ]

    await harness._pump_terms("queue-1")

    assert term.state is TermState.AIRED


async def test_a_scheduled_term_the_ledger_dropped_is_dropped_with_its_reason() -> None:
    """What the bus reports about a placed term is kept, reason included."""
    harness = _harness(elapsed=192.0)
    _decide(harness)
    await harness._pump_terms("queue-1")
    term = harness._term_store("queue-1").terms()[0]
    harness.mass.streams.ledger = [
        SimpleNamespace(
            label="AI DJ transition",
            state=OverlayProgramState.DROPPED,
            start_frame=term.scheduled_frame,
            frames=1,
            reason=ScheduleReason.BUS_REPLACED,
        )
    ]

    await harness._pump_terms("queue-1")

    assert term.state is TermState.DROPPED
    assert term.dropped_reason == "bus_replaced"


async def test_a_scheduled_term_aired_at_its_deadline() -> None:
    """A placed term ends even when no ledger row reaches it."""
    harness = _harness(elapsed=192.0)
    _decide(harness)
    await harness._pump_terms("queue-1")
    term = harness._term_store("queue-1").terms()[0]
    term.seconds_left = 0.05
    # the item was heard for a second since the last pass
    queue = harness.mass.player_queues.queue
    term.last_elapsed = float(queue.corrected_elapsed_time) - 1.0
    term.last_seen_at = asyncio.get_running_loop().time() - 1.0

    await harness._pump_terms("queue-1")

    assert term.state is TermState.AIRED


async def test_a_paused_queue_holds_the_deadline() -> None:
    """A pause pauses the deadline instead of skipping the term."""
    harness = _harness(elapsed=192.0)
    _decide(harness)
    await harness._pump_terms("queue-1")
    term = harness._term_store("queue-1").terms()[0]
    term.seconds_left = 0.05
    term.last_seen_at = asyncio.get_running_loop().time() - 1.0
    harness.mass.player_queues.queue.state = PlaybackState.PAUSED

    await harness._pump_terms("queue-1")

    assert term.state is TermState.SCHEDULED
    assert term.seconds_left == pytest.approx(0.05)


async def test_a_scheduled_term_the_player_skipped_is_cancelled_and_dropped() -> None:
    """A placed term whose boundary is jumped over does not air somewhere else."""
    harness = _harness(elapsed=192.0)
    _decide(harness)
    await harness._pump_terms("queue-1")
    term = harness._term_store("queue-1").terms()[0]
    harness.mass.player_queues.queue.current = "c"

    await harness._pump_terms("queue-1")

    assert term.state is TermState.DROPPED
    assert term.dropped_reason == "skipped"
    assert harness.mass.streams.cancelled == [
        ("queue-1", "AI DJ transition", "skipped", term.scheduled_frame)
    ]


async def test_a_scheduled_term_whose_successor_plays_on_waits_for_the_ledger() -> None:
    """Crossing the boundary normally is not a skip."""
    harness = _harness(elapsed=192.0)
    _decide(harness)
    await harness._pump_terms("queue-1")
    term = harness._term_store("queue-1").terms()[0]
    harness.mass.player_queues.queue.current = "b"
    harness.mass.player_queues.queue.elapsed = 0.0

    await harness._pump_terms("queue-1")

    assert term.state is TermState.SCHEDULED


async def test_a_boundary_closer_than_the_head_is_too_late() -> None:
    """A head that no longer fits leaves no start to render for, so the term is not paid for."""
    harness = _harness(elapsed=190.0)
    # a fifteen second break against a ten second boundary: the head can not fit
    harness.rendered_pcm = _pcm(15.0)
    _decide(
        harness,
        _plan(sections=(_section(allow_pre=True),)),
        head_seconds=TALK_OVER_MAX_PRE,
    )
    assert _terms(harness)[0].head_seconds == TALK_OVER_MAX_PRE

    await harness._pump_terms("queue-1")

    term = _terms(harness)[0]
    assert term.state is TermState.DROPPED
    assert term.dropped_reason == "too_late"
    assert harness.rendered == []
    assert harness.mass.streams.overlays == []


async def test_a_repeat_one_boundary_stays_valid() -> None:
    """Repeat-one makes the running item its own successor, and its term still airs."""
    harness = _harness(durations=(200,), current="a", elapsed=192.0)
    _decide(
        harness,
        _plan(
            queue_item_id="a",
            successor_id="a",
            sections=(_section(allow_post=True),),
        ),
    )

    await harness._pump_terms("queue-1")

    assert harness._term_store("queue-1").terms()[0].state is TermState.SCHEDULED
    assert harness.mass.streams.overlays[0][3] == pytest.approx(8.0 - TERM_START_BIAS_SECONDS)


async def test_a_short_track_is_rendered_during_its_predecessor() -> None:
    """A track shorter than a render is served while the item before it still plays."""
    harness = _harness(durations=(200, 3, 200), elapsed=175.0)
    _decide(
        harness,
        _plan(
            queue_item_id="b",
            successor_id="c",
            sections=(_section(allow_post=True),),
        ),
    )

    await harness._pump_terms("queue-1")
    # twenty five seconds of the predecessor plus the three second track: not in window yet
    assert harness.rendered == []

    harness.mass.player_queues.queue.elapsed = 195.0
    await harness._pump_terms("queue-1")

    assert harness.rendered == [("rick", "transition")]
    # the term starts at the end of the short track, eight seconds from now, less the bias
    assert harness.mass.streams.overlays[0][3] == pytest.approx(8.0 - TERM_START_BIAS_SECONDS)


async def test_a_term_placed_ahead_ends_at_its_deadline() -> None:
    """A term placed ahead of its boundary still ends, even when no event follows it."""
    harness = _harness(durations=(200, 3, 200), elapsed=195.0)
    _decide(harness, _plan(queue_item_id="b", successor_id="c"))
    await harness._pump_terms("queue-1")
    term = harness._term_store("queue-1").terms()[0]
    assert term.state is TermState.SCHEDULED
    assert term.last_seen_at is not None

    term.seconds_left = 0.05
    term.last_seen_at = asyncio.get_running_loop().time() - 1.0
    await harness._pump_terms("queue-1")

    assert term.state is TermState.AIRED


async def test_a_term_ahead_of_an_item_without_a_duration_waits() -> None:
    """Without a known duration for an item in between, its boundary is not timed yet."""
    harness = _harness(durations=(200, None, 200), elapsed=196.0)
    _decide(harness, _plan(queue_item_id="c", successor_id=None))

    await harness._pump_terms("queue-1")

    assert harness.rendered == []
    assert harness._term_store("queue-1").terms()[0].state is TermState.DECIDED


async def test_a_term_rendered_ahead_is_dropped_when_the_queue_is_skipped() -> None:
    """Skipping past the boundary while the term rendered ahead loses it."""
    harness = _harness(durations=(200, 3, 200), elapsed=195.0)
    harness.during_render = lambda: setattr(harness.mass.player_queues.queue, "current", "c")
    _decide(harness, _plan(queue_item_id="b", successor_id="c"))

    await harness._pump_terms("queue-1")

    term = harness._term_store("queue-1").terms()[0]
    assert term.dropped_reason == "skipped"
    assert harness.mass.streams.overlays == []


async def test_a_term_placed_below_the_bus_lead_says_its_head_is_cut(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A start the bus pushes back for its own lead shrinks the head, and that is logged."""
    harness = _harness(elapsed=180.0)
    harness.rendered_pcm = _pcm(15.0)
    _decide(
        harness,
        _plan(sections=(_section(allow_pre=True, allow_post=True),)),
        head_seconds=TALK_OVER_MAX_PRE,
    )
    # the boundary comes closer while the term renders, as a reordered queue does
    harness.during_render = lambda: setattr(
        harness.mass.player_queues.queue.items[0], "duration", 193.5
    )

    with caplog.at_level(logging.WARNING):
        await harness._pump_terms("queue-1")

    # 13.5s left, a 12s head and the bias: the bus lead pushes the start back to 2s
    assert harness.mass.streams.overlays[0][3] == pytest.approx(1.25)
    assert "bus lead" in caplog.text


async def test_every_boundary_gets_exactly_one_placement() -> None:
    """Two decided boundaries are placed once each, never twice."""
    harness = _harness(elapsed=192.0)
    _decide(
        harness,
        _plan(queue_item_id="a", successor_id="b", sections=(_section("one"),)),
        _plan(queue_item_id="b", successor_id="c", sections=(_section("two"),)),
    )
    await harness._pump_terms("queue-1")
    await harness._pump_terms("queue-1")

    assert len(harness.mass.streams.overlays) == 1
    # the second boundary waits until its item plays
    terms = harness._term_store("queue-1").terms()
    assert [term.state for term in terms] == [TermState.SCHEDULED, TermState.DECIDED]


async def test_a_pump_reads_the_ledger_once_for_every_placed_term() -> None:
    """Every placed term settles against the pump's one read, not one read of its own."""
    harness = _harness(elapsed=192.0)
    _decide(
        harness,
        _plan(queue_item_id="a", successor_id="b", sections=(_section("one"),)),
        _plan(queue_item_id="b", successor_id="c", sections=(_section("two"),)),
    )
    store = harness._term_store("queue-1")
    # both boundaries stand on the bus, as they do once the second one came due
    for index, term in enumerate(store.terms()):
        term.mark_scheduled(
            start_frame=index * PCM_FORMAT.sample_rate,
            pcm_len=len(_pcm()),
            delay_seconds=1.0,
            seconds_left=30.0,
        )
    harness.mass.streams.ledger_reads = 0

    await harness._pump_terms("queue-1")

    # the pass settled both against the one answer: the running boundary still stands and
    # the one whose item is not playing is dropped, and neither cost a read of its own
    assert [term.state for term in store.terms()] == [TermState.SCHEDULED, TermState.DROPPED]
    assert harness.mass.streams.ledger_reads == 1


async def test_two_section_sets_at_one_boundary_are_one_term() -> None:
    """One boundary carries one term, whatever a later pass would speak there."""
    harness = _harness()
    _decide(harness, _plan(sections=(_section("one"),)))
    _decide(harness, _plan(sections=(_section("two"),)))

    terms = harness._term_store("queue-1").terms()
    assert len(terms) == 1
    assert terms[0].section_ids == ("one",)


async def test_deciding_the_same_boundary_twice_keeps_one_term() -> None:
    """Every replan re-decides the gaps, and must not stack terms on one boundary."""
    harness = _harness()
    _decide(harness)
    _decide(harness)

    assert len(harness._term_store("queue-1")) == 1


async def test_a_later_boundary_does_not_move_the_booked_pump() -> None:
    """One queue keeps one pump, booked for the earliest due term."""
    harness = _harness()

    harness._book_term_pump("queue-1", 5.0)
    due = harness._term_pump_due["queue-1"]

    harness._book_term_pump("queue-1", 60.0)

    assert len(harness.mass.timers) == 1
    assert harness.mass.timers[0][0] == pytest.approx(5.0)
    assert harness._term_pump_due["queue-1"] == pytest.approx(due)


async def test_an_earlier_boundary_books_its_own_pump() -> None:
    """A boundary that comes sooner than the booked one gets its own earlier pump."""
    harness = _harness()
    harness._book_term_pump("queue-1", 60.0)
    later_due = harness._term_pump_due["queue-1"]

    harness._book_term_pump("queue-1", 5.0)

    assert len(harness.mass.timers) == 2
    assert harness.mass.timers[-1][0] == pytest.approx(5.0)
    assert harness._term_pump_due["queue-1"] < later_due


async def test_a_far_boundary_books_only_the_pump_cap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A pass booked a whole window ahead reads a lead the producer's buffer has outgrown."""
    caplog.set_level(logging.DEBUG, logger="test_term_engine")
    # the live shape: 358s to the boundary, a 12s lead the host set and a producer that
    # led by 7s when the pass was booked
    harness = _harness(durations=(400, 200, 200), elapsed=42.0)
    bus: Any = harness.mass.streams
    bus.lead_seconds = 7.04
    harness.rendered_pcm = _pcm(12.0)
    _decide(
        harness,
        _plan(sections=(_section(allow_pre=True, allow_post=True),)),
        head_seconds=TALK_OVER_MAX_PRE,
    )

    await harness._pump_terms("queue-1")

    window = TALK_OVER_MAX_PRE + 7.04 + RENDER_ESTIMATE_SECONDS + TERM_SCHEDULE_TOLERANCE_SECONDS
    # the window alone would book this pass 338s out, where the lead it was decided on is
    # no longer the one the bus runs with
    assert 358.0 - window > TERM_PUMP_MAX_SECONDS
    assert harness.mass.timers[-1][0] == pytest.approx(TERM_PUMP_MAX_SECONDS)
    assert harness.rendered == []

    # the plateau: the buffer has filled, so the same boundary now sits inside the lead
    bus.lead_seconds = 35.39
    harness.mass.player_queues.queue.elapsed = 400.0 - 27.38
    term = _terms(harness)[0]
    term.last_seen_at = asyncio.get_running_loop().time() - 330.0

    await harness._pump_terms("queue-1")

    assert "late=True" in caplog.text
    assert term.state is TermState.DROPPED
    assert term.dropped_reason == "too_late"
    assert harness.rendered == []


async def test_a_window_nearer_than_the_cap_is_booked_as_it_is() -> None:
    """A boundary nearer than the cap books its window, and a later one does not move it."""
    harness = _harness(elapsed=165.0)
    bus: Any = harness.mass.streams
    bus.lead_seconds = 7.0
    _decide(
        harness,
        _plan(queue_item_id="a", successor_id="b"),
        _plan(queue_item_id="b", successor_id="c", sections=(_section("two"),)),
    )

    await harness._pump_terms("queue-1")

    # 35s to the boundary and a 29s window: the window is the earlier of the two, and the
    # boundary 200s behind it books its cap without moving this one
    window = (
        7.0 + RENDER_ESTIMATE_SECONDS + TERM_SCHEDULE_TOLERANCE_SECONDS + TERM_RENDER_MARGIN_SECONDS
    )
    assert 0.0 < 35.0 - window < TERM_PUMP_MAX_SECONDS
    assert len(harness.mass.timers) == 1
    assert harness.mass.timers[0][0] == pytest.approx(35.0 - window)
    assert harness.mass.timers[0][0] == pytest.approx(min(35.0 - window, TERM_PUMP_MAX_SECONDS))


async def test_a_placed_terms_deadline_is_booked_without_the_cap() -> None:
    """After a placement the pass is the term's own way out, not a window to re-read."""
    harness = _harness(elapsed=174.0)
    bus: Any = harness.mass.streams
    bus.lead_seconds = 7.0
    harness.rendered_pcm = _pcm(12.0)
    _decide(
        harness,
        _plan(sections=(_section(allow_pre=True),)),
        head_seconds=TALK_OVER_MAX_PRE,
    )

    await harness._pump_terms("queue-1")

    term = _terms(harness)[0]
    assert term.state is TermState.SCHEDULED
    assert term.seconds_left is not None
    assert term.seconds_left > TERM_PUMP_MAX_SECONDS
    assert harness.mass.timers[-1][0] == pytest.approx(term.seconds_left)


async def test_a_due_window_renders_at_once_and_books_no_pass() -> None:
    """An open window is not capped: the pump renders it now instead of waiting again."""
    harness = _harness(elapsed=192.0)
    bus: Any = harness.mass.streams
    bus.lead_seconds = LEAD_MIN_LEAD_SECONDS
    _decide(harness)

    await harness._pump_terms("queue-1")

    term = _terms(harness)[0]
    assert term.state is TermState.SCHEDULED
    assert harness.rendered != []
    # the only pass left is the placement's deadline, not another look at the window
    assert len(harness.mass.timers) == 1
    assert harness.mass.timers[0][0] == pytest.approx(term.seconds_left)


async def test_the_booked_due_is_consumed_when_the_pump_runs() -> None:
    """A pump clears its booking at the door, so a later term can book again."""
    harness = _harness()
    harness._book_term_pump("queue-1", 5.0)
    assert "queue-1" in harness._term_pump_due

    await harness._pump_terms("queue-1")

    assert "queue-1" not in harness._term_pump_due
    harness._book_term_pump("queue-1", 5.0)
    assert len(harness.mass.timers) == 2


async def test_a_newly_booked_pump_does_not_abort_a_running_one() -> None:
    """A timer that fires while a pump renders must not cancel that pump."""
    harness = _harness(elapsed=192.0)
    harness.render_gate = asyncio.Event()
    _decide(harness)
    harness._book_term_pump("queue-1", 0.0)
    running = harness.mass.fire(harness.mass.timers[-1][3])
    await harness.render_started.wait()

    harness._book_term_pump("queue-1", 5.0)
    later = harness.mass.fire(harness.mass.timers[-1][3])
    assert running.cancelled() is False

    harness.render_gate.set()
    await asyncio.gather(running, later)

    assert len(harness.mass.streams.overlays) == 1


async def test_a_term_is_never_a_queue_item() -> None:
    """The DJ plans terms, so it never rewrites the user's queue."""
    harness = _harness(elapsed=192.0)
    _decide(harness)

    await harness._pump_terms("queue-1")

    assert harness.mass.player_queues.update_calls == 0


async def test_dropping_the_terms_cancels_what_was_already_placed() -> None:
    """Letting go of a queue stops its placed terms instead of leaving them on the bus."""
    harness = _harness(elapsed=192.0)
    _decide(harness)
    await harness._pump_terms("queue-1")
    term = harness._term_store("queue-1").terms()[0]

    harness._drop_terms("queue-1", ScheduleReason.FLOW_ENDED)

    assert term.state is TermState.DROPPED
    assert term.dropped_reason == "flow_ended"
    assert harness.mass.streams.cancelled == [
        ("queue-1", "AI DJ transition", "flow_ended", term.scheduled_frame)
    ]


async def test_an_evicted_term_releases_its_guard_and_gap() -> None:
    """A term the bounded store gives up holds neither the guard nor its gap back."""
    harness = _harness()
    state = harness._dj_queues["queue-1"]
    farthest = _section()
    farthest.history_events = [("transition", (15, 0.0))]
    state.history = {"transition": [(15, 0.0)]}
    plans = [_plan(queue_item_id=f"q{index}", successor_id=f"s{index}") for index in range(16)]
    plans[-1] = _plan(queue_item_id="q15", successor_id="s15", sections=(farthest,))
    plans.append(_plan(queue_item_id="q16", successor_id="s16"))

    harness._decide_terms("queue-1", {"host_id": "rick"}, plans)

    store = harness._term_store("queue-1")
    assert len(store) == 16
    # the boundary furthest out is the one that had to go, and with it what it claimed
    assert store.get(("q15", "s15")) is None
    assert store.get(("q0", "s0")) is not None
    assert state.history.get("transition") == []
    # the slot went, so a later pass may decide that boundary again
    assert store.state(("q15", "s15")) is BoundaryState.OPEN
    assert store.state(("q0", "s0")) is BoundaryState.DECIDED


async def test_the_store_is_gone_after_a_reset() -> None:
    """A disarmed DJ forgets the boundaries of the show it played."""
    harness = _harness()
    _decide(harness)

    harness._reset_terms("queue-1")

    assert harness._term_store("queue-1").terms() == ()


async def test_the_term_key_survives_a_replan_with_the_same_sections() -> None:
    """The boundary, not an index and not the sections, identifies a term."""
    harness = _harness()
    _decide(harness)
    term = harness._term_store("queue-1").terms()[0]

    assert term.key == term_key("a", "b")


def test_a_section_without_post_marks_the_term_it_decides() -> None:
    """The term carries the post flag of the sections it speaks, so placement can honour it."""
    harness = _harness()
    _decide(harness, _plan(sections=(_section(allow_post=False),)))

    assert _terms(harness)[0].allow_post is False


def test_a_section_that_may_post_marks_the_term_it_decides() -> None:
    """A section that allows a post decides a term that may run over the boundary."""
    harness = _harness()
    _decide(harness, _plan(sections=(_section(allow_post=True),)))

    assert _terms(harness)[0].allow_post is True


def test_a_term_of_two_sections_may_post_only_when_both_allow_it() -> None:
    """A merged break carries over only when every section of it allows the post."""
    harness = _harness()
    _decide(
        harness,
        _plan(sections=(_section("one", allow_post=True), _section("two", allow_post=False))),
    )

    assert _terms(harness)[0].allow_post is False


async def test_a_term_that_may_not_post_ends_before_the_boundary() -> None:
    """A break without a post starts its whole clip before the boundary and ends there."""
    harness = _harness(elapsed=182.0)
    harness.rendered_pcm = _pcm(8.0)
    _decide(harness, _plan(sections=(_section(allow_post=False),)))

    await harness._pump_terms("queue-1")

    term = _terms(harness)[0]
    assert term.state is TermState.SCHEDULED
    start_in_seconds = harness.mass.streams.overlays[0][3]
    # no start bias is taken off, so the clip ends on the boundary and not past it
    assert start_in_seconds == pytest.approx(18.0 - 8.0)
    assert start_in_seconds + 8.0 == pytest.approx(18.0)
    assert harness.mass.streams.heads == [pytest.approx(8.0, abs=1e-3)]


async def test_a_term_that_can_not_end_before_its_boundary_is_dropped_as_too_late(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A boundary too close for the clip is given up instead of letting it run over."""
    caplog.set_level(logging.INFO, logger="test_term_engine")
    harness = _harness(elapsed=194.0)
    _fast_renders(harness)
    harness.rendered_pcm = _pcm(8.0)
    _decide(harness, _plan(sections=(_section(allow_post=False),)))

    await harness._pump_terms("queue-1")

    term = _terms(harness)[0]
    assert term.state is TermState.DROPPED
    assert term.dropped_reason == "too_late"
    # nothing is mixed into the stream for a break that can not be placed
    assert harness.mass.streams.overlays == []
    # the drop names its reason, so the log tells a late boundary from a lost one
    assert any("too_late" in record.getMessage() for record in caplog.records)


def _key(item_id: str = "a", successor_id: str | None = "b") -> tuple[str, str | None]:
    """Return the boundary key of the term the default plan decides."""
    return (item_id, successor_id)


async def _pumps(harness: TermEngineHarness, count: int) -> None:
    """Run the pump the given number of times, each a whole period after the last."""
    due_by_queue: dict[str, float] = getattr(harness, "_term_boundary_replan_due", {})
    harness._term_boundary_replan_due = due_by_queue
    for _ in range(count):
        # every pump gets its own period, so a rate limit can not hide a retry
        due_by_queue["queue-1"] = asyncio.get_running_loop().time() - 1.0
        await harness._pump_terms("queue-1")


async def test_a_boundary_that_can_never_be_timed_is_dropped_once_and_left_alone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A live source serves no duration, so nothing may keep asking for that boundary."""
    harness = _harness(durations=(None, 200, 200))
    _decide(harness)

    await harness._pump_terms("queue-1")

    term = _terms(harness)[0]
    assert term.dropped_reason == "no_duration"
    assert harness.rendered == []
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="test_term_engine"):
        await _pumps(harness, 10)

    store = harness._term_store("queue-1")
    assert store.state(_key()) is BoundaryState.UNREACHABLE
    # no pass is asked for a boundary that can not be spoken
    assert harness.replanned == []
    assert not [record for record in caplog.records if "no_duration" in record.getMessage()]
    # and a pass that would replan it decides nothing there a second time
    _decide(harness)
    assert _terms(harness) == [term]
    assert harness.rendered == []


async def test_a_boundary_whose_fade_cannot_be_told_is_dropped_once_and_left_alone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A queue that says nothing about its fade has no boundary anything can be timed on."""
    harness = _harness(elapsed=190.0)
    harness.mass.streams.crossfade_mode = CrossfadeMode.SMART_CROSSFADE
    harness.mass.player_queues.queue.items[0].media_type = MediaType.UNKNOWN
    _decide(harness)

    await harness._pump_terms("queue-1")

    term = _terms(harness)[0]
    assert term.dropped_reason == "unknown_boundary"
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="test_term_engine"):
        await _pumps(harness, 10)

    store = harness._term_store("queue-1")
    assert store.state(_key()) is BoundaryState.UNREACHABLE
    assert harness.replanned == []
    assert not [record for record in caplog.records if "unknown_boundary" in record.getMessage()]
    _decide(harness)
    assert _terms(harness) == [term]


async def test_a_boundary_the_bus_refuses_by_format_is_dropped_once_and_left_alone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A program the bus can not take in its format is no boundary to try again either."""
    harness = _harness(elapsed=192.0)
    harness.mass.streams.result = ScheduleResult(
        scheduled=False,
        anchor_frame=0,
        frames=0,
        late_by_seconds=0.0,
        lead_shortfall_seconds=0.0,
        reason=ScheduleReason.FORMAT,
    )
    _decide(harness)

    await harness._pump_terms("queue-1")

    term = _terms(harness)[0]
    assert term.dropped_reason == "format"
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="test_term_engine"):
        await _pumps(harness, 10)

    store = harness._term_store("queue-1")
    assert store.state(_key()) is BoundaryState.UNREACHABLE
    assert harness.replanned == []
    assert not [record for record in caplog.records if "format" in record.getMessage()]
    _decide(harness)
    assert _terms(harness) == [term]


def test_a_decided_boundary_is_not_decided_again() -> None:
    """A boundary that already holds a living term keeps it, whatever a later pass plans."""
    harness = _harness()
    _decide(harness)
    first = _terms(harness)[0]

    _decide(harness, _plan(sections=(_section("news"),)))

    store = harness._term_store("queue-1")
    assert _terms(harness) == [first]
    assert store.state(_key()) is BoundaryState.DECIDED
    assert store.is_settled(_key()) is True


def test_a_dropped_term_reopens_its_boundary() -> None:
    """A break that never aired leaves its boundary open, and it is decided afresh."""
    harness = _harness()
    _decide(harness)
    store = harness._term_store("queue-1")
    first = _terms(harness)[0]

    harness._drop_term("queue-1", store, first, TermDropReason.SKIPPED)

    assert store.state(_key()) is BoundaryState.OPEN
    assert store.is_settled(_key()) is False

    _decide(harness)

    second = store.term_of(_key())
    assert second is not None
    assert second is not first
    assert second.state is TermState.DECIDED


async def test_an_aired_term_keeps_its_boundary_closed() -> None:
    """What the listener heard was decided, so the pump asks for no second pass on it."""
    harness = _harness()
    _decide(harness)
    store = harness._term_store("queue-1")
    _terms(harness)[0].mark_aired()

    await harness._pump_terms("queue-1")

    assert store.state(_key()) is BoundaryState.DECIDED
    assert harness.replanned == []


def test_the_horizon_counts_only_the_boundaries_that_are_open() -> None:
    """A settled or unreachable boundary takes no slot of the store a pass may decide."""
    harness = _harness(durations=(200, 200, 200, 200, 200, 200))
    store = harness._term_store("queue-1")
    window = [{"item_id": f"i{index}", "available": True, "duration": 200} for index in range(6)]
    store.settle([("i1", "i2")])
    store.add(
        ProgramTerm(
            queue_item_id="i2",
            successor_id="i3",
            section_ids=("transition",),
            head_seconds=0.0,
        )
    )
    closed = store.add(
        ProgramTerm(
            queue_item_id="i3",
            successor_id="i4",
            section_ids=("transition",),
            head_seconds=0.0,
        )
    )
    assert closed is not None
    store.drop(closed.key, "no_duration")

    horizon = harness._term_gap_horizon("queue-1", window)

    assert horizon == {("i0", "i1"), ("i4", "i5")}


def test_a_full_store_still_decides_the_boundaries_that_are_due_first() -> None:
    """
    A store filled with far boundaries decides the near ones instead of staying silent.

    The store bound is what a pass has to respect, but a boundary that is due before
    everything it holds is no boundary to leave undecided: the term it takes the slot of
    is due later, and the nearer break comes first.
    """
    harness = _harness(durations=tuple([200] * 20))
    store = harness._term_store("queue-1")
    for index in range(TERMS_PER_QUEUE - 2):
        store.add(
            ProgramTerm(
                queue_item_id=f"q{index}",
                successor_id=f"q{index + 1}",
                section_ids=("transition",),
                head_seconds=0.0,
            )
        )
    window = [{"item_id": f"i{index}", "available": True, "duration": 200} for index in range(6)]

    horizon = harness._term_gap_horizon("queue-1", window)

    # every boundary of the window is due before the q* terms the store holds, so the pass
    # serves all of them and gives those up for it
    assert horizon == {("i0", "i1"), ("i1", "i2"), ("i2", "i3"), ("i3", "i4"), ("i4", "i5")}


def test_a_store_full_of_boundaries_that_are_nearer_decides_nothing() -> None:
    """A full store whose terms all come due first is no reason to give one up."""
    harness = _harness(durations=tuple([200] * 20))
    store = harness._term_store("queue-1")
    for index in range(TERMS_PER_QUEUE):
        store.add(
            ProgramTerm(
                queue_item_id=f"q{index}",
                successor_id=f"q{index + 1}",
                section_ids=("transition",),
                head_seconds=0.0,
            ),
            position=index,
        )
    window = [{"item_id": f"i{index}", "available": True, "duration": 200} for index in range(6)]

    # the window starts twenty tracks into the queue, behind every term the store holds
    horizon = harness._term_gap_horizon("queue-1", window, index_base=20)

    # nothing in the window is due before what the store holds, so the pass gives nothing up
    assert horizon == set()


def test_a_nearer_boundary_never_gives_the_shows_end_up() -> None:
    """The end of the show is due last of all, so no nearer break takes the slot it holds."""
    harness = _harness(durations=tuple([200] * 20))
    store = harness._term_store("queue-1")
    outro = ProgramTerm(
        queue_item_id="last", successor_id=None, section_ids=("transition",), head_seconds=0.0
    )
    store.add(outro, position=400)
    for index in range(TERMS_PER_QUEUE - 1):
        store.add(
            ProgramTerm(
                queue_item_id=f"q{index}",
                successor_id=f"q{index + 1}",
                section_ids=("transition",),
                head_seconds=0.0,
            ),
            position=200 + index,
        )
    window = [{"item_id": f"i{index}", "available": True, "duration": 200} for index in range(6)]

    horizon = harness._term_gap_horizon("queue-1", window)
    harness._decide_terms(
        "queue-1",
        {},
        [
            TermBoundaryPlan(
                queue_item_id=f"i{index}",
                successor_id=f"i{index + 1}",
                sections=(_section(),),
                index=index,
            )
            for index in range(len(window) - 1)
        ],
        horizon=horizon,
    )

    assert horizon == {("i0", "i1"), ("i1", "i2"), ("i2", "i3"), ("i3", "i4"), ("i4", "i5")}
    # the break at the end of the show keeps the slot it holds: nothing speaks after it
    assert store.term_of(("last", None)) is outro


def test_a_boundary_a_pass_leaves_silent_is_named(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A pass that decides nothing at a boundary says so, so a silent DJ can be traced."""
    harness = _harness()
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="test_term_engine"):
        harness._decide_terms("queue-1", {}, [], horizon={_key()})

    assert "speaks no break at 1 of the 1 boundaries it served: a" in caplog.text
    assert harness._term_store("queue-1").state(_key()) is BoundaryState.DECIDED


def test_a_full_store_gives_up_only_as_many_slots_as_it_holds() -> None:
    """The store's bound holds while a nearer boundary takes a slot: nothing else is added."""
    harness = _harness(durations=tuple([200] * 20))
    harness._term_stores = {"queue-1": TermStore("queue-1", max_terms=2)}
    store = harness._term_store("queue-1")
    for index in range(2):
        store.add(
            ProgramTerm(
                queue_item_id=f"q{index}",
                successor_id=f"q{index + 1}",
                section_ids=("transition",),
                head_seconds=0.0,
            ),
            position=30 + index,
        )
    window = [{"item_id": f"i{index}", "available": True, "duration": 200} for index in range(6)]

    horizon = harness._term_gap_horizon("queue-1", window)

    # two held terms are two slots to take: the nearest two boundaries take them, and the
    # rest of the window waits until something comes due
    assert horizon == {("i0", "i1"), ("i1", "i2")}


def test_the_horizon_asks_the_store_that_holds_the_terms() -> None:
    """A queue whose store keeps fewer terms decides fewer boundaries, whatever the default."""
    harness = _harness(durations=tuple([200] * 20))
    harness._term_stores = {"queue-1": TermStore("queue-1", max_terms=2)}
    window = [{"item_id": f"i{index}", "available": True, "duration": 200} for index in range(6)]

    horizon = harness._term_gap_horizon("queue-1", window)

    assert horizon == {("i0", "i1"), ("i1", "i2")}


async def test_an_evicted_term_hands_its_rendered_clip_back() -> None:
    """A term the bounded store gives up does not keep the clip it rendered."""
    harness = _harness()
    _decide(harness)
    term = _terms(harness)[0]
    # a render that landed in a pause is held by the term until it can be placed
    term.keep_rendered(_pcm(8.0), PCM_FORMAT)
    assert term.rendered_pcm is not None

    harness._term_evicted("queue-1", term)

    assert term.rendered_pcm is None
    assert term.rendered_format is None

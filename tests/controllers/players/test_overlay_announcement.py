"""Tests for playing an announcement over the queue's stream instead of on the device."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import ContentType, PlayerFeature, PlayerType
from music_assistant_models.errors import PlayerCommandFailed
from music_assistant_models.media_items import AudioFormat

from music_assistant.constants import ATTR_ANNOUNCEMENT_IN_PROGRESS
from music_assistant.controllers.players import PlayerController
from music_assistant.controllers.players.announcements import (
    ANNOUNCEMENT_MODE_DEVICE,
    ANNOUNCEMENT_MODE_OVERLAY,
)
from music_assistant.controllers.streams.controller import ScheduleReason, ScheduleResult
from music_assistant.controllers.streams.program_schedule import (
    PRIORITY_ANNOUNCEMENT,
    ProgramKind,
)
from tests.common import MockProvider, attach_mock

PLAYER_ID = "player_1"
QUEUE_ID = "queue-1"
FLOW_PCM_FORMAT = AudioFormat(
    content_type=ContentType.PCM_S16LE, sample_rate=48000, bit_depth=16, channels=2
)
ANNOUNCEMENT_URL = "http://ma/announcement/player_1.mp3"
# the streams controller's answer when the clip was mixed into the queue's stream
OVERLAY_SCHEDULED = ScheduleResult(
    scheduled=True,
    anchor_frame=0,
    frames=0,
    late_by_seconds=0.0,
    lead_shortfall_seconds=0.0,
    reason=None,
)
OVERLAY_REFUSED = ScheduleResult(
    scheduled=False,
    anchor_frame=0,
    frames=0,
    late_by_seconds=0.0,
    lead_shortfall_seconds=0.0,
    reason=ScheduleReason.NO_BUS,
)


class _Harness:
    """A controller with one player and the gate's surroundings mocked."""

    def __init__(
        self,
        mock_mass: MagicMock,
        *,
        program_bus: bool = True,
        flow_mode: bool = True,
        active_bus: bool = True,
        clip_seconds: float = 3.0,
    ) -> None:
        self.mass = mock_mass
        self.controller = PlayerController(mock_mass)
        mock_mass.players = self.controller
        provider = MockProvider("test_provider", instance_id="test", mass=mock_mass)
        self.player, self.protocol = attach_mock(provider, PLAYER_ID, "Player 1")
        # the mock config would make every player a flow mode player
        self.protocol._config.get_value = MagicMock(return_value=False)  # type: ignore[method-assign]
        if not flow_mode:
            self.protocol._attr_supported_features.add(PlayerFeature.ENQUEUE)
        self.protocol._cache.clear()
        self.player.refresh_state(signal_event=False)

        mock_mass.player_queues.get = MagicMock(
            return_value=SimpleNamespace(queue_id=QUEUE_ID, current_item=None, items=[])
        )
        mock_mass.streams.program_bus_enabled = MagicMock(return_value=program_bus)
        # the announcement volume settings fall back to their defaults, as on a player
        # nobody configured them for
        mock_mass.config.get_raw_player_config_value = MagicMock(
            side_effect=lambda _player_id, _key, default=None: default
        )
        # the same read through the player config, which is where upstream asks for it now
        mock_mass.config.get_player_value = MagicMock(
            side_effect=lambda _player_id, _key, default=None: default
        )
        mock_mass.streams.program_bus_active = MagicMock(return_value=active_bus)
        mock_mass.streams.program_bus_format = MagicMock(return_value=FLOW_PCM_FORMAT)
        mock_mass.streams.schedule_overlay = MagicMock(return_value=OVERLAY_SCHEDULED)
        mock_mass.streams.get_announcement_url = MagicMock(return_value=ANNOUNCEMENT_URL)

        self.render = MagicMock()
        self.render.wait_ready = AsyncMock(return_value=True)
        self.render.wait_finished = AsyncMock(return_value=clip_seconds)

        async def _get_stream(_pcm_format: AudioFormat) -> Any:
            yield b"\x01\x00" * 64

        self.render.get_stream = MagicMock(side_effect=_get_stream)
        renderer = mock_mass.streams.announcement_renderer
        renderer.register = MagicMock(return_value=self.render)
        renderer.unregister = AsyncMock()
        self.device_play = AsyncMock()
        self.controller._play_announcement = self.device_play  # type: ignore[method-assign]
        self.controller._play_native_announcement = AsyncMock()  # type: ignore[method-assign]
        self.controller._resolve_ready_announce_player = AsyncMock(  # type: ignore[method-assign]
            return_value=None
        )

    async def announce(self, **kwargs: Any) -> None:
        """Play the announcement on the player through the controller."""
        await self.controller.play_announcement(PLAYER_ID, ANNOUNCEMENT_URL, **kwargs)

    def scheduled(self) -> bool:
        """Return whether an overlay was scheduled on the queue's stream."""
        return cast("MagicMock", self.mass.streams.schedule_overlay).called


@pytest.fixture
def mock_mass() -> MagicMock:
    """Return a MusicAssistant stand-in."""
    mass = MagicMock()
    mass.closing = False
    mass.config.get_raw_core_config_value = MagicMock(return_value="GLOBAL")
    mass.config.get_raw_player_config_value = MagicMock(return_value=None)
    return mass


@pytest.mark.parametrize(
    ("program_bus", "flow_mode", "active_bus", "expected_overlay", "reason"),
    [
        (False, True, True, False, "the program bus setting is the kill switch"),
        (True, True, False, False, "the queue has no live flow session"),
        (True, False, True, False, "the player does not stream in flow mode"),
        (True, True, True, True, "the player's own flow session mixes programs"),
    ],
)
async def test_the_overlay_gate(
    mock_mass: MagicMock,
    program_bus: bool,
    flow_mode: bool,
    active_bus: bool,
    expected_overlay: bool,
    reason: str,
) -> None:
    """Only a switched-on bus, a flow mode player and a live session mix the clip."""
    harness = _Harness(
        mock_mass,
        program_bus=program_bus,
        flow_mode=flow_mode,
        active_bus=active_bus,
    )

    await harness.announce(mode=ANNOUNCEMENT_MODE_OVERLAY)

    assert harness.scheduled() is expected_overlay, reason
    # the clip is heard exactly once: either over the stream or on the device
    assert harness.device_play.await_count == (0 if expected_overlay else 1), reason


async def test_an_overlay_on_a_group_is_mixed_once_into_the_group_stream(
    mock_mass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A group overlay rides the group's own stream instead of becoming one clip per member."""
    harness = _Harness(mock_mass)
    provider = MockProvider("test_provider", instance_id="test", mass=mock_mass)
    group, _group_protocol = attach_mock(provider, "group_1", "Group", player_type=PlayerType.GROUP)
    harness.controller._players = {group.player_id: group}
    group.refresh_state(signal_event=False)
    # the members could otherwise take the fan-out route a device announcement uses
    monkeypatch.setattr(
        harness.controller, "_members_announce_in_step", MagicMock(return_value=True)
    )
    member_runner = AsyncMock()
    monkeypatch.setattr(harness.controller, "_run_member_announcement", member_runner)

    await harness.controller.play_announcement(
        group.player_id, ANNOUNCEMENT_URL, mode=ANNOUNCEMENT_MODE_OVERLAY
    )

    # one mix on the group's queue, heard by every member in step ...
    harness.mass.streams.schedule_overlay.assert_called_once()
    assert harness.mass.streams.schedule_overlay.call_args.args[0] == QUEUE_ID
    # ... and no per-member announcement of its own
    member_runner.assert_not_awaited()
    harness.device_play.assert_not_awaited()


async def test_the_default_mode_never_touches_the_queue_stream(mock_mass: MagicMock) -> None:
    """Callers that do not ask for it keep today's announcement route."""
    harness = _Harness(mock_mass, active_bus=True)

    await harness.announce()

    assert harness.scheduled() is False
    harness.device_play.assert_awaited_once()


async def test_the_device_mode_is_still_accepted_explicitly(mock_mass: MagicMock) -> None:
    """An explicit device mode is the same route as before."""
    harness = _Harness(mock_mass, active_bus=True)

    await harness.announce(mode=ANNOUNCEMENT_MODE_DEVICE)

    assert harness.scheduled() is False
    harness.device_play.assert_awaited_once()


async def test_an_unknown_mode_is_rejected(mock_mass: MagicMock) -> None:
    """A mode nobody implements fails instead of playing something else."""
    harness = _Harness(mock_mass)

    with pytest.raises(PlayerCommandFailed, match="Invalid mode"):
        await harness.announce(mode="loudspeaker")

    assert harness.scheduled() is False
    harness.device_play.assert_not_awaited()


async def test_the_overlay_is_mixed_in_the_stream_format_with_its_label(
    mock_mass: MagicMock,
) -> None:
    """The rendered clip is converted to the session's PCM format and labelled for the log."""
    harness = _Harness(mock_mass)
    before = asyncio.all_tasks()

    await harness.announce(mode=ANNOUNCEMENT_MODE_OVERLAY)

    # nothing of the overlay runs on its own: it is mixed by the stream that is already playing
    assert asyncio.all_tasks() == before
    # the exact duration is known before anything is placed
    harness.render.wait_finished.assert_awaited_once()
    harness.render.get_stream.assert_called_once_with(FLOW_PCM_FORMAT)
    # the clip is placed once, not once per consumer or per chunk
    harness.mass.streams.schedule_overlay.assert_called_once()
    queue_id, pcm = harness.mass.streams.schedule_overlay.call_args.args
    assert queue_id == QUEUE_ID
    assert pcm == b"\x01\x00" * 64
    kwargs = harness.mass.streams.schedule_overlay.call_args.kwargs
    assert kwargs["label"].startswith("announcement:")
    assert kwargs["priority"] == PRIORITY_ANNOUNCEMENT
    assert kwargs["kind"] is ProgramKind.ANNOUNCEMENT
    harness.device_play.assert_not_awaited()
    # the announcement does not stay marked as in progress, the player kept playing
    assert not harness.player.extra_data.get(ATTR_ANNOUNCEMENT_IN_PROGRESS)


async def test_an_empty_render_fails_instead_of_ducking_without_audio(mock_mass: MagicMock) -> None:
    """Nothing is placed when the clip could not be rendered."""
    harness = _Harness(mock_mass, clip_seconds=0.0)

    with pytest.raises(PlayerCommandFailed, match="not available"):
        await harness.announce(mode=ANNOUNCEMENT_MODE_OVERLAY)

    assert harness.scheduled() is False
    harness.device_play.assert_not_awaited()


async def test_a_stream_that_stopped_mixing_falls_back_to_the_device(
    mock_mass: MagicMock,
) -> None:
    """A session that ended between the gate and the placement still gets the announcement."""
    harness = _Harness(mock_mass)
    harness.mass.streams.schedule_overlay.return_value = OVERLAY_REFUSED

    await harness.announce(mode=ANNOUNCEMENT_MODE_OVERLAY)

    harness.device_play.assert_awaited_once()


async def test_a_level_asked_for_is_reported_once_while_the_clip_still_plays(
    mock_mass: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """An overlay keeps the player's volume: a wanted level is said to be left out, once."""
    harness = _Harness(mock_mass)

    with caplog.at_level(logging.INFO):
        await harness.announce(mode=ANNOUNCEMENT_MODE_OVERLAY, volume_level=55)

    # the announcement is still heard, over the stream and not on the device
    assert harness.scheduled() is True
    harness.device_play.assert_not_awaited()
    reported = [
        record.getMessage()
        for record in caplog.records
        if record.name == harness.controller.logger.name
        and "is not applied over the queue's stream" in record.getMessage()
    ]
    assert len(reported) == 1
    # the level that would have been set on the device is the one reported
    assert "55" in reported[0]


async def test_the_gate_is_reported_once_per_call(
    mock_mass: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """A player without flow mode announces on the device and says why, once."""
    harness = _Harness(mock_mass, flow_mode=False)

    with caplog.at_level(logging.INFO):
        await harness.announce(mode=ANNOUNCEMENT_MODE_OVERLAY)

    reasons = [
        record.getMessage()
        for record in caplog.records
        if record.name == harness.controller.logger.name
        and "not mixing it over the queue's stream" in record.getMessage()
    ]
    assert len(reasons) == 1
    assert "the player does not stream in flow mode" in reasons[0]


async def test_the_clip_is_rendered_completely_before_it_is_placed(mock_mass: MagicMock) -> None:
    """The announcement is only mixed in once its whole audio exists, and only once."""
    harness = _Harness(mock_mass)
    rendered = False

    async def _finished() -> float:
        nonlocal rendered
        rendered = True
        return 3.0

    def _place(*_args: Any, **_kwargs: Any) -> ScheduleResult:
        assert rendered, "the announcement was placed before its render finished"
        return OVERLAY_SCHEDULED

    harness.render.wait_finished = AsyncMock(side_effect=_finished)
    harness.mass.streams.schedule_overlay = MagicMock(side_effect=_place)

    await harness.announce(mode=ANNOUNCEMENT_MODE_OVERLAY)

    harness.render.wait_finished.assert_awaited_once()
    harness.render.get_stream.assert_called_once()
    harness.mass.streams.schedule_overlay.assert_called_once()
    harness.device_play.assert_not_awaited()


async def test_a_refusal_says_why_instead_of_blaming_the_bus(
    mock_mass: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """A format refusal is logged as a format refusal, not as a missing session."""
    harness = _Harness(mock_mass)
    harness.mass.streams.schedule_overlay.return_value = ScheduleResult(
        scheduled=False,
        anchor_frame=0,
        frames=0,
        late_by_seconds=0.0,
        lead_shortfall_seconds=0.0,
        reason=ScheduleReason.FORMAT,
    )

    with caplog.at_level(logging.INFO):
        await harness.announce(mode=ANNOUNCEMENT_MODE_OVERLAY)

    reported = [
        record.getMessage()
        for record in caplog.records
        if "not mixed into the queue's stream" in record.getMessage()
    ]
    assert len(reported) == 1
    assert "format" in reported[0]
    # the device fallback repeats the real reason instead of blaming the bus
    fallbacks = [
        record.getMessage()
        for record in caplog.records
        if "playing it as a device announcement" in record.getMessage()
    ]
    assert len(fallbacks) == 1
    assert "format" in fallbacks[0]


async def test_two_announcements_are_told_apart_by_their_labels(mock_mass: MagicMock) -> None:
    """Every placement gets its own label, so two announcements both play and stay distinct."""
    harness = _Harness(mock_mass)

    await harness.announce(mode=ANNOUNCEMENT_MODE_OVERLAY)
    await harness.announce(mode=ANNOUNCEMENT_MODE_OVERLAY)

    assert harness.mass.streams.schedule_overlay.call_count == 2
    labels = [call.kwargs["label"] for call in harness.mass.streams.schedule_overlay.call_args_list]
    assert len(set(labels)) == 2
    assert all(label.startswith("announcement:") for label in labels)
    harness.device_play.assert_not_awaited()


async def test_a_placed_announcement_is_kept_until_the_ledger_says_it_aired(
    mock_mass: MagicMock,
) -> None:
    """The clip is retained for a bus change and the retention is subscribed once."""
    harness = _Harness(mock_mass)

    await harness.announce(mode=ANNOUNCEMENT_MODE_OVERLAY)

    harness.mass.streams.add_program_bus_listener.assert_called_once()
    label = harness.mass.streams.schedule_overlay.call_args.kwargs["label"]
    assert harness.controller._announcement_retention.retained_labels() == (label,)


async def test_a_queue_that_is_removed_releases_its_retained_clips(mock_mass: MagicMock) -> None:
    """A player that disappears does not leave its rendered clips in the retention."""
    harness = _Harness(mock_mass)
    await harness.announce(mode=ANNOUNCEMENT_MODE_OVERLAY)
    label = harness.mass.streams.schedule_overlay.call_args.kwargs["label"]
    assert harness.controller._announcement_retention.retained_labels() == (label,)

    harness.controller.on_queue_removed(QUEUE_ID)

    assert harness.controller._announcement_retention.retained_labels() == ()
    # the streams controller forgets what it kept for the queue as well
    harness.mass.streams.on_queue_removed.assert_called_once_with(QUEUE_ID)

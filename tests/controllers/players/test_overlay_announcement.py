"""Tests for playing an announcement over the queue's stream instead of on the device."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import ContentType, PlayerFeature
from music_assistant_models.errors import PlayerCommandFailed
from music_assistant_models.media_items import AudioFormat

from music_assistant.constants import ATTR_ANNOUNCEMENT_IN_PROGRESS
from music_assistant.controllers.players import PlayerController
from music_assistant.controllers.players.announcements import (
    ANNOUNCEMENT_MODE_DEVICE,
    ANNOUNCEMENT_MODE_OVERLAY,
)
from music_assistant.controllers.streams.constants import CONF_STREAM_PROGRAM_BUS
from tests.common import MockPlayer, MockProvider

PLAYER_ID = "player_1"
QUEUE_ID = "queue-1"
FLOW_PCM_FORMAT = AudioFormat(
    content_type=ContentType.PCM_S16LE, sample_rate=48000, bit_depth=16, channels=2
)
ANNOUNCEMENT_URL = "http://ma/announcement/player_1.mp3"


class _Harness:
    """A controller with one player and the gate's surroundings mocked."""

    def __init__(
        self,
        mock_mass: MagicMock,
        *,
        program_bus: bool = True,
        flow_mode: bool = True,
        claimed: bool = False,
        active_bus: bool = True,
        clip_seconds: float = 3.0,
    ) -> None:
        self.mass = mock_mass
        self.controller = PlayerController(mock_mass)
        provider = MockProvider("test_provider", instance_id="test", mass=mock_mass)
        self.player = MockPlayer(provider, PLAYER_ID, "Player 1")
        # the mock config would make every player a flow mode player
        self.player._config.get_value = MagicMock(return_value=False)  # type: ignore[method-assign]
        if not flow_mode:
            self.player._attr_supported_features.add(PlayerFeature.ENQUEUE)
        self.player._cache.clear()
        self.controller._players = {PLAYER_ID: self.player}
        mock_mass.players = self.controller

        mock_mass.player_queues.get = MagicMock(return_value=SimpleNamespace(queue_id=QUEUE_ID))
        mock_mass.streams.get_config_value = MagicMock(
            side_effect=lambda key, *_args, **_kwargs: (
                program_bus if key == CONF_STREAM_PROGRAM_BUS else None
            )
        )
        mock_mass.streams.program_bus_required = MagicMock(return_value=claimed)
        mock_mass.streams.program_bus_active = MagicMock(return_value=active_bus)
        mock_mass.streams.program_bus_format = MagicMock(return_value=FLOW_PCM_FORMAT)
        mock_mass.streams.schedule_overlay = MagicMock(return_value=True)
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
    ("program_bus", "flow_mode", "claimed", "active_bus", "expected_overlay", "reason"),
    [
        (False, True, True, True, False, "the program bus setting is the kill switch"),
        (True, True, False, False, False, "no claim and no live flow session"),
        (True, False, True, True, False, "the player does not stream in flow mode"),
        (True, True, True, False, True, "an owner claimed the queue's stream"),
        (True, True, False, True, True, "the player's own flow session mixes programs"),
    ],
)
async def test_the_overlay_gate(
    mock_mass: MagicMock,
    program_bus: bool,
    flow_mode: bool,
    claimed: bool,
    active_bus: bool,
    expected_overlay: bool,
    reason: str,
) -> None:
    """Only a switched-on bus, a flow mode player and a claim or live session mix the clip."""
    harness = _Harness(
        mock_mass,
        program_bus=program_bus,
        flow_mode=flow_mode,
        claimed=claimed,
        active_bus=active_bus,
    )

    await harness.announce(mode=ANNOUNCEMENT_MODE_OVERLAY)

    assert harness.scheduled() is expected_overlay, reason
    # the clip is heard exactly once: either over the stream or on the device
    assert harness.device_play.await_count == (0 if expected_overlay else 1), reason


async def test_the_default_mode_never_touches_the_queue_stream(mock_mass: MagicMock) -> None:
    """Callers that do not ask for it keep today's announcement route."""
    harness = _Harness(mock_mass, claimed=True, active_bus=True)

    await harness.announce()

    assert harness.scheduled() is False
    harness.device_play.assert_awaited_once()


async def test_the_device_mode_is_still_accepted_explicitly(mock_mass: MagicMock) -> None:
    """An explicit device mode is the same route as before."""
    harness = _Harness(mock_mass, claimed=True, active_bus=True)

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
    harness = _Harness(mock_mass, claimed=True)
    before = asyncio.all_tasks()

    await harness.announce(mode=ANNOUNCEMENT_MODE_OVERLAY)

    # nothing of the overlay runs on its own: it is mixed by the stream that is already playing
    assert asyncio.all_tasks() == before
    # the exact duration is known before anything is placed
    harness.render.wait_finished.assert_awaited_once()
    harness.render.get_stream.assert_called_once_with(FLOW_PCM_FORMAT)
    queue_id, pcm = harness.mass.streams.schedule_overlay.call_args.args
    assert queue_id == QUEUE_ID
    assert pcm == b"\x01\x00" * 64
    assert harness.mass.streams.schedule_overlay.call_args.kwargs == {"label": "announcement"}
    harness.device_play.assert_not_awaited()
    # the announcement does not stay marked as in progress, the player kept playing
    assert not harness.player.extra_data.get(ATTR_ANNOUNCEMENT_IN_PROGRESS)


async def test_an_empty_render_fails_instead_of_ducking_without_audio(mock_mass: MagicMock) -> None:
    """Nothing is placed when the clip could not be rendered."""
    harness = _Harness(mock_mass, claimed=True, clip_seconds=0.0)

    with pytest.raises(PlayerCommandFailed, match="not available"):
        await harness.announce(mode=ANNOUNCEMENT_MODE_OVERLAY)

    assert harness.scheduled() is False
    harness.device_play.assert_not_awaited()


async def test_a_stream_that_stopped_mixing_falls_back_to_the_device(
    mock_mass: MagicMock,
) -> None:
    """A session that ended between the gate and the placement still gets the announcement."""
    harness = _Harness(mock_mass, claimed=True)
    harness.mass.streams.schedule_overlay.return_value = False

    await harness.announce(mode=ANNOUNCEMENT_MODE_OVERLAY)

    harness.device_play.assert_awaited_once()


async def test_the_gate_is_reported_once_per_call(
    mock_mass: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """A player without flow mode announces on the device and says why, once."""
    harness = _Harness(mock_mass, flow_mode=False, claimed=True)

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

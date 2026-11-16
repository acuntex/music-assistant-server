"""Tests for which queue requirements put a player's stream in flow mode."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from music_assistant_models.enums import PlayerFeature

from music_assistant.controllers.streams.controller import StreamsController


def _controller() -> StreamsController:
    """Build a controller with empty requirement registries."""
    ctrl = StreamsController.__new__(StreamsController)
    ctrl._flow_mode_owners = {}
    ctrl._boundary_mix_owners = {}
    ctrl._lead_hold_owners = {}
    return ctrl


def _player(*, gapless: bool) -> Any:
    """Return a player stand-in reporting the given gapless capability."""
    return cast(
        "Any",
        SimpleNamespace(
            supports_feature=lambda feature: gapless and feature == PlayerFeature.GAPLESS_PLAYBACK,
        ),
    )


def test_a_lead_hold_needs_the_flow_even_on_a_gapless_player() -> None:
    """A lead starts before the item it belongs to ends, which only the flow mixes."""
    ctrl = _controller()
    ctrl._lead_hold_owners["q1"] = {"ai_radio"}

    assert ctrl._needs_flow_mode("q1", _player(gapless=True)) is True


def test_a_boundary_mix_alone_keeps_a_gapless_player_on_its_own_items() -> None:
    """A gapless player mixes its own boundaries, so the flow is not needed for that."""
    ctrl = _controller()
    ctrl._boundary_mix_owners["q1"] = {"ai_radio"}

    assert ctrl._needs_flow_mode("q1", _player(gapless=True)) is False


def test_a_boundary_mix_needs_the_flow_on_a_player_that_is_not_gapless() -> None:
    """A player without gapless cannot mix two items, so the flow has to."""
    ctrl = _controller()
    ctrl._boundary_mix_owners["q1"] = {"ai_radio"}

    assert ctrl._needs_flow_mode("q1", _player(gapless=False)) is True


def test_an_explicit_flow_mode_owner_always_wins() -> None:
    """An owner that needs the flow asks for it directly."""
    ctrl = _controller()
    ctrl._flow_mode_owners["q1"] = {"player_queues"}

    assert ctrl._needs_flow_mode("q1", _player(gapless=True)) is True


def test_an_unclaimed_queue_stays_on_its_per_item_streams() -> None:
    """Nothing asks for the flow, so the player gets its items one by one."""
    ctrl = _controller()

    assert ctrl._needs_flow_mode("q1", _player(gapless=True)) is False

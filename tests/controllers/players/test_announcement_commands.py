"""The announcement controller's API commands are registered under the right names."""

from __future__ import annotations

from unittest.mock import MagicMock

from music_assistant_models.auth import Scope

from music_assistant.controllers.players.controller import PlayerController
from tests.common import use_api_command_registry


def test_the_tts_engines_command_is_not_shadowed_by_the_queue_cleanup() -> None:
    """The queue cleanup is a lifecycle hook, not the TTS engine listing command."""
    mass = MagicMock()
    mass.closing = False
    mass.config.get_raw_core_config_value = MagicMock(return_value="GLOBAL")
    controller = PlayerController(mass)
    use_api_command_registry(mass, controller)

    assert "players/tts_engines" in mass.command_handlers
    handler = mass.command_handlers["players/tts_engines"]
    assert handler.target.__name__ == "get_announcement_tts_engines"
    assert handler.required_scope is Scope.PLAYERS_CONTROL
    # the queue cleanup is a lifecycle hook, so it does not shadow a command
    assert not hasattr(PlayerController.on_queue_removed, "api_cmd")
    assert PlayerController.get_announcement_tts_engines.api_cmd == "players/tts_engines"

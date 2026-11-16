"""Regression tests for the program bus switch living in the core streams config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from music_assistant.controllers.streams import StreamsController

REPO_ROOT = Path(__file__).resolve().parents[3]


async def _entries() -> dict[str, Any]:
    """Return the streams config entries by key."""
    controller: Any = StreamsController.__new__(StreamsController)
    return {entry.key: entry for entry in await controller.get_config_entries()}


def _authoring_texts() -> dict[str, str]:
    """Return the program bus switch's authored label and description."""
    authoring: dict[str, Any] = json.loads(
        (REPO_ROOT / "music_assistant/strings.json").read_text("utf-8")
    )
    return cast("dict[str, str]", authoring["config_entries"]["stream_program_bus"])


def _built_english() -> dict[str, str]:
    """Return the English strings the server ships."""
    return cast(
        "dict[str, str]",
        json.loads((REPO_ROOT / "music_assistant/translations/en.json").read_text("utf-8")),
    )


async def test_the_program_bus_is_a_core_streams_switch() -> None:
    """The bus mixes programs for every feature, so its switch is not a provider setting."""
    entry = (await _entries())["stream_program_bus"]

    assert entry.category == "playback"
    assert entry.default_value is False


async def test_the_program_bus_switch_is_not_hidden_in_the_advanced_section() -> None:
    """A core feature behind an advanced toggle is one nobody finds."""
    entry = (await _entries())["stream_program_bus"]

    assert entry.advanced is False


def test_the_program_bus_switch_texts_match_the_generated_english_source() -> None:
    """The label and description are the authored strings, not a stale copy of them."""
    authored = _authoring_texts()
    built = _built_english()

    for field in ("label", "description"):
        assert built[f"common.config_entries.stream_program_bus.{field}"] == authored[field]

"""Tests for the per-engine render cool-downs of one queue DJ."""

from __future__ import annotations

import pytest

from music_assistant.providers.ai_radio.constants import (
    AI_COOLDOWN_SECONDS,
    RENDER_FAILURES_BEFORE_COOLDOWN,
    TTS_COOLDOWN_SECONDS,
)
from music_assistant.providers.ai_radio.models import RenderHealth, RenderPart

START = 100.0


def _fail(health: RenderHealth, part: RenderPart, now: float, times: int) -> list[bool]:
    """Fail the given engine the given number of times and return what each failure decided."""
    return [health.record_failure(part, now) for _ in range(times)]


def test_a_fresh_health_renders_freely() -> None:
    """A DJ that had no failure renders on both engines."""
    health = RenderHealth()

    assert not health.cooling_down(START)
    assert not health.ai_cooling_down(START)
    assert not health.tts_cooling_down(START)
    assert health.ai.failures == 0
    assert health.tts.failures == 0


def test_the_two_engines_pause_for_the_designed_times() -> None:
    """The AI engine pauses fifteen minutes, the voice five."""
    assert AI_COOLDOWN_SECONDS == 15 * 60
    assert TTS_COOLDOWN_SECONDS == 5 * 60


def test_three_ai_failures_pause_only_the_ai_engine() -> None:
    """An AI query that keeps failing pauses the queries, not the voice."""
    health = RenderHealth()

    assert _fail(health, RenderPart.AI, START, RENDER_FAILURES_BEFORE_COOLDOWN) == [
        False,
        False,
        True,
    ]

    assert health.ai_cooling_down(START)
    assert health.ai.cooldown_until == pytest.approx(START + AI_COOLDOWN_SECONDS)
    # the voice never failed, so it is not held back by the query
    assert not health.tts_cooling_down(START)
    assert health.tts.failures == 0
    assert health.tts.cooldown_until is None


def test_three_tts_failures_pause_only_the_voice() -> None:
    """A voice that keeps failing pauses the voice, not the queries."""
    health = RenderHealth()

    assert _fail(health, RenderPart.TTS, START, RENDER_FAILURES_BEFORE_COOLDOWN) == [
        False,
        False,
        True,
    ]

    assert health.tts_cooling_down(START)
    assert health.tts.cooldown_until == pytest.approx(START + TTS_COOLDOWN_SECONDS)
    assert not health.ai_cooling_down(START)
    assert health.ai.failures == 0
    assert health.ai.cooldown_until is None


def test_failures_of_both_engines_never_add_up() -> None:
    """Two failures each are four failures, but neither engine is in its pause."""
    health = RenderHealth()

    _fail(health, RenderPart.AI, START, RENDER_FAILURES_BEFORE_COOLDOWN - 1)
    _fail(health, RenderPart.TTS, START, RENDER_FAILURES_BEFORE_COOLDOWN - 1)

    assert not health.ai_cooling_down(START)
    assert not health.tts_cooling_down(START)
    assert not health.cooling_down(START)


def test_both_paused_holds_the_contribution_until_the_later_engine_returns() -> None:
    """With both engines paused nothing is spoken, and the later pause is what decides."""
    health = RenderHealth()
    _fail(health, RenderPart.AI, START, RENDER_FAILURES_BEFORE_COOLDOWN)
    _fail(health, RenderPart.TTS, START, RENDER_FAILURES_BEFORE_COOLDOWN)

    assert health.cooling_down(START)

    # the voice comes back first, and the paused query still blocks the contribution
    voice_returns = START + TTS_COOLDOWN_SECONDS
    assert not health.tts_cooling_down(voice_returns)
    assert health.cooling_down(voice_returns)
    assert health.resume_in(voice_returns) == pytest.approx(
        AI_COOLDOWN_SECONDS - TTS_COOLDOWN_SECONDS
    )

    # a pause ends exactly at its last paused second
    query_returns = START + AI_COOLDOWN_SECONDS
    assert not health.ai_cooling_down(query_returns)
    assert not health.cooling_down(query_returns)
    assert health.resume_in(query_returns) == 0.0


def test_a_pause_does_not_extend_while_it_runs() -> None:
    """A render that was already in flight when the pause started fails late: no new pause."""
    health = RenderHealth()
    _fail(health, RenderPart.TTS, START, RENDER_FAILURES_BEFORE_COOLDOWN)
    paused_until = health.tts.cooldown_until

    assert health.record_failure(RenderPart.TTS, START + 5) is False

    assert health.tts.failures == RENDER_FAILURES_BEFORE_COOLDOWN + 1
    assert health.tts.cooldown_until == paused_until


def test_a_success_clears_only_the_engine_it_talked_to() -> None:
    """A query that comes back says nothing about the voice, and the other way round."""
    health = RenderHealth()
    _fail(health, RenderPart.AI, START, 2)
    _fail(health, RenderPart.TTS, START, 1)

    health.record_success(RenderPart.AI, START + 1)

    assert health.ai.failures == 0
    assert health.tts.failures == 1


def test_a_success_during_a_running_pause_leaves_it_alone() -> None:
    """A call that was already in flight proves nothing, so the pause runs its course."""
    health = RenderHealth()
    _fail(health, RenderPart.TTS, START, RENDER_FAILURES_BEFORE_COOLDOWN)
    paused_until = health.tts.cooldown_until

    health.record_success(RenderPart.TTS, START + 5)

    assert health.tts.failures == RENDER_FAILURES_BEFORE_COOLDOWN
    assert health.tts.cooldown_until == paused_until

    # the first call after the pause decides: a success lifts it
    assert paused_until is not None
    health.record_success(RenderPart.TTS, paused_until)

    assert health.tts.failures == 0
    assert health.tts.cooldown_until is None


def test_a_whole_render_that_came_back_clears_both_engines() -> None:
    """A contribution that aired used both engines, so neither count survives it."""
    health = RenderHealth()
    _fail(health, RenderPart.AI, START, 2)
    _fail(health, RenderPart.TTS, START, 2)

    health.record_render_success(START + 1)

    assert health.ai.failures == 0
    assert health.tts.failures == 0

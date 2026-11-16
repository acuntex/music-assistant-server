"""The render health of a queue DJ, as the term pump reads it."""

from __future__ import annotations

import asyncio

from music_assistant.providers.ai_radio.terms import TermState

from .test_term_engine import _decide, _harness


async def test_a_cooldown_renders_nothing() -> None:
    """While the render health pauses the DJ, no engine is asked again."""
    harness = _harness(elapsed=192.0)
    state = harness._dj_queues["queue-1"]
    state.render_health.tts.cooldown_until = asyncio.get_running_loop().time() + 60
    _decide(harness)

    await harness._pump_terms("queue-1")

    assert harness.rendered == []
    assert harness._term_store("queue-1").terms()[0].state is TermState.DECIDED

"""Unit tests for the one-off cleanup of the DJ items an older release spliced in."""

from __future__ import annotations

import logging
from typing import Any

from music_assistant.controllers.player_queues.constants import (
    CACHE_CATEGORY_PLAYER_QUEUE_ITEMS,
    CACHE_CATEGORY_PLAYER_QUEUE_STATE,
)
from music_assistant.providers.ai_radio.constants import (
    ATTR_QUEUE_DJ,
    ATTR_SESSION_ID,
    CONF_LEGACY_DJ_ITEMS_CLEANED,
)
from music_assistant.providers.ai_radio.upgrade import remove_legacy_dj_items

LOGGER = logging.getLogger("test.ai_radio.upgrade")


class FakeQueueItem:
    """Minimal queue item carrying only what the cleanup reads."""

    def __init__(
        self, item_id: str, *, dj: bool = False, extra: dict[str, Any] | None = None
    ) -> None:
        """Initialize the item with its extra attributes."""
        attributes = dict(extra or {})
        if dj:
            attributes[ATTR_QUEUE_DJ] = True
        self.queue_item_id = item_id
        self.extra_attributes = attributes


class FakeQueue:
    """Queue snapshot with the playback pointers the guard is read from."""

    def __init__(
        self,
        queue_id: str,
        current_index: int | None = None,
        index_in_buffer: int | None = None,
    ) -> None:
        """Initialize the queue's pointers."""
        self.queue_id = queue_id
        self.current_index = current_index
        self.index_in_buffer = index_in_buffer


class FakePlayerQueues:
    """Queue controller stand-in holding the given queues and their items."""

    domain = "player_queues"

    def __init__(
        self,
        queues: dict[str, list[FakeQueueItem]],
        guards: dict[str, tuple[int | None, int | None]] | None = None,
    ) -> None:
        """Initialize with the queues that are loaded and the items their players own."""
        self.queues = queues
        self.guards = guards or {}
        self.updates: list[tuple[str, list[str]]] = []
        self.unreadable: set[str] = set()

    def all(self) -> tuple[FakeQueue, ...]:
        """Return the queues that are loaded."""
        return tuple(
            FakeQueue(queue_id, *self.guards.get(queue_id, (None, None)))
            for queue_id in self.queues
        )

    def get(self, queue_id: str) -> FakeQueue | None:
        """Return a queue snapshot when the queue is loaded."""
        if queue_id not in self.queues:
            return None
        current_index, index_in_buffer = self.guards.get(queue_id, (None, None))
        return FakeQueue(queue_id, current_index, index_in_buffer)

    def items(self, queue_id: str, limit: int = 500, offset: int = 0) -> list[FakeQueueItem]:
        """Return one page of the queue's items."""
        if queue_id in self.unreadable:
            raise RuntimeError(f"queue {queue_id} is unreadable")
        return self.queues[queue_id][offset : offset + limit]

    def update_items(self, queue_id: str, items: list[FakeQueueItem]) -> None:
        """Replace the queue's items."""
        if queue_id in self.unreadable:
            raise RuntimeError(f"queue {queue_id} is unreadable")
        self.updates.append((queue_id, [item.queue_item_id for item in items]))
        self.queues[queue_id] = list(items)


class FakeCache:
    """Cache stand-in exposing the persisted queue entries of the given ids."""

    def __init__(self, persisted: set[str]) -> None:
        """Initialize with the queues that were persisted by a previous run."""
        self.persisted = persisted
        self.calls: list[tuple[str, int]] = []

    async def get_all(
        self, provider: str = "default", category: int = 0, base_class: Any = None
    ) -> dict[str, Any]:
        """Return one entry per persisted queue for the queue categories."""
        self.calls.append((provider, category))
        if category not in (CACHE_CATEGORY_PLAYER_QUEUE_STATE, CACHE_CATEGORY_PLAYER_QUEUE_ITEMS):
            return {}
        return {queue_id: [] for queue_id in self.persisted}


class BrokenCache:
    """Cache stand-in that cannot be read at all."""

    async def get_all(
        self, provider: str = "default", category: int = 0, base_class: Any = None
    ) -> dict[str, Any]:
        """Fail as an unreadable cache database does."""
        raise RuntimeError("cache database is gone")


class FakeConfig:
    """Provider config stand-in holding the raw values of one instance."""

    def __init__(self) -> None:
        """Initialize with no stored values."""
        self.values: dict[str, Any] = {}

    def get_raw_provider_config_value(self, instance_id: str, key: str, default: Any = None) -> Any:
        """Return the stored raw value for the key."""
        return self.values.get(key, default)

    def set_raw_provider_config_value(
        self, instance_id: str, key: str, value: Any, **kwargs: Any
    ) -> None:
        """Store the raw value for the key."""
        self.values[key] = value


class BrokenConfig(FakeConfig):
    """Provider config stand-in that cannot write the marker."""

    def set_raw_provider_config_value(
        self, instance_id: str, key: str, value: Any, **kwargs: Any
    ) -> None:
        """Fail as a config that cannot be written does."""
        raise RuntimeError("config is read-only")


class FakeMass:
    """Mass stand-in serving the queue, cache and config pieces of the cleanup."""

    def __init__(
        self,
        queues: dict[str, list[FakeQueueItem]],
        persisted: set[str],
        config: FakeConfig | None = None,
        guards: dict[str, tuple[int | None, int | None]] | None = None,
    ) -> None:
        """Initialize with the loaded queues and the persisted queue ids."""
        self.player_queues = FakePlayerQueues(queues, guards)
        self.cache = FakeCache(persisted)
        self.config = config or FakeConfig()


async def test_legacy_dj_items_are_removed_and_the_rest_kept() -> None:
    """An old DJ clip goes, and items without its mark stay where they are."""
    mass = FakeMass(
        {
            "queue-1": [
                FakeQueueItem("music-1"),
                FakeQueueItem("break", dj=True),
                # a non-DJ item that carries other attributes is no clip of the DJ
                FakeQueueItem("music-2", extra={"ai_radio_pre_overlap": 4}),
            ]
        },
        {"queue-1"},
    )

    assert await remove_legacy_dj_items(mass, LOGGER, "ai_radio") is True

    assert mass.player_queues.updates == [("queue-1", ["music-1", "music-2"])]
    assert [item.queue_item_id for item in mass.player_queues.queues["queue-1"]] == [
        "music-1",
        "music-2",
    ]
    assert mass.config.values[CONF_LEGACY_DJ_ITEMS_CLEANED] is True


async def test_a_marked_cleanup_removes_nothing_on_the_next_start() -> None:
    """The marker makes the cleanup a one-off, even when a clip is still lying around."""
    config = FakeConfig()
    first = FakeMass({"queue-1": [FakeQueueItem("break", dj=True)]}, {"queue-1"}, config)
    assert await remove_legacy_dj_items(first, LOGGER, "ai_radio") is True

    second = FakeMass({"queue-1": [FakeQueueItem("late-break", dj=True)]}, {"queue-1"}, config)

    assert await remove_legacy_dj_items(second, LOGGER, "ai_radio") is False
    assert second.player_queues.updates == []
    assert [item.queue_item_id for item in second.player_queues.queues["queue-1"]] == ["late-break"]


async def test_a_queue_that_cannot_be_checked_is_tried_again_on_the_next_start(
    caplog: Any,
) -> None:
    """A failing queue is logged, holds the marker back and is checked again next start."""
    config = FakeConfig()
    mass = FakeMass(
        {
            "queue-1": [FakeQueueItem("music-1"), FakeQueueItem("break-1", dj=True)],
            "queue-2": [FakeQueueItem("music-2"), FakeQueueItem("break-2", dj=True)],
        },
        {"queue-1", "queue-2"},
        config,
    )
    mass.player_queues.unreadable = {"queue-2"}

    with caplog.at_level(logging.WARNING):
        assert await remove_legacy_dj_items(mass, LOGGER, "ai_radio") is False

    assert mass.player_queues.updates == [("queue-1", ["music-1"])]
    assert CONF_LEGACY_DJ_ITEMS_CLEANED not in config.values
    assert "queue-2" in caplog.text

    # the next start checks the whole pass again, and this time it can finish
    mass.player_queues.unreadable = set()
    assert await remove_legacy_dj_items(mass, LOGGER, "ai_radio") is True
    assert mass.player_queues.updates[-1] == ("queue-2", ["music-2"])
    assert config.values[CONF_LEGACY_DJ_ITEMS_CLEANED] is True


async def test_a_queue_that_was_not_loaded_yet_is_checked_on_the_next_start() -> None:
    """A persisted queue that is not loaded holds the marker back until it can be checked."""
    config = FakeConfig()
    # queue-2 is persisted but its player did not show up before this start's cleanup
    first = FakeMass({"queue-1": [FakeQueueItem("music-1")]}, {"queue-1", "queue-2"}, config)

    assert await remove_legacy_dj_items(first, LOGGER, "ai_radio") is False
    assert first.player_queues.updates == []
    assert CONF_LEGACY_DJ_ITEMS_CLEANED not in config.values

    second = FakeMass(
        {"queue-1": [FakeQueueItem("music-1")], "queue-2": [FakeQueueItem("break", dj=True)]},
        {"queue-1", "queue-2"},
        config,
    )

    assert await remove_legacy_dj_items(second, LOGGER, "ai_radio") is True
    assert second.player_queues.updates == [("queue-2", [])]
    assert config.values[CONF_LEGACY_DJ_ITEMS_CLEANED] is True


async def test_an_unreadable_cache_holds_the_marker_back_without_raising() -> None:
    """A cache that cannot be listed leaves the cleanup undone instead of failing the start."""
    config = FakeConfig()
    mass = FakeMass({"queue-1": [FakeQueueItem("break", dj=True)]}, {"queue-1"}, config)
    mass.cache = BrokenCache()  # type: ignore[assignment]

    assert await remove_legacy_dj_items(mass, LOGGER, "ai_radio") is False
    assert mass.player_queues.updates == []
    assert CONF_LEGACY_DJ_ITEMS_CLEANED not in config.values


async def test_a_marker_that_cannot_be_written_does_not_raise() -> None:
    """A config that refuses the marker leaves the cleanup to the next start."""
    mass = FakeMass({"queue-1": [FakeQueueItem("break", dj=True)]}, {"queue-1"}, BrokenConfig())

    assert await remove_legacy_dj_items(mass, LOGGER, "ai_radio") is False
    assert mass.player_queues.updates == [("queue-1", [])]


async def test_a_clip_the_player_already_owns_stays() -> None:
    """A clip the player is playing or holds next stays: removing it would shift its indexes."""
    mass = FakeMass(
        {
            "queue-1": [
                FakeQueueItem("music-1"),
                FakeQueueItem("owned-break", dj=True),
                FakeQueueItem("pending-break", dj=True),
                FakeQueueItem("music-2"),
            ]
        },
        {"queue-1"},
        guards={"queue-1": (1, 1)},
    )

    assert await remove_legacy_dj_items(mass, LOGGER, "ai_radio") is True

    assert mass.player_queues.updates == [("queue-1", ["music-1", "owned-break", "music-2"])]


async def test_a_clip_the_running_session_just_spliced_stays() -> None:
    """A retry pass keeps the clips of the session that is planning right now."""
    mass = FakeMass(
        {
            "queue-1": [
                FakeQueueItem("live-break", dj=True, extra={ATTR_SESSION_ID: "dj_live"}),
                FakeQueueItem("old-break", dj=True, extra={ATTR_SESSION_ID: "dj_old"}),
                FakeQueueItem("unlabelled-break", dj=True),
            ]
        },
        {"queue-1"},
    )

    assert await remove_legacy_dj_items(mass, LOGGER, "ai_radio", {"queue-1": "dj_live"}) is True

    assert mass.player_queues.updates == [("queue-1", ["live-break"])]


async def test_a_start_without_a_loaded_queue_waits_and_scans_nothing() -> None:
    """No queue registered yet: no cache scan, no marker, and the QUEUE_ADDED pass cleans up."""
    config = FakeConfig()
    # the queue was persisted, but its player has not registered yet
    first = FakeMass({}, {"queue-1"}, config)

    assert await remove_legacy_dj_items(first, LOGGER, "ai_radio") is False
    assert first.cache.calls == []
    assert CONF_LEGACY_DJ_ITEMS_CLEANED not in config.values

    # the queue registers: the pass the QUEUE_ADDED event triggers does the cleanup
    first.player_queues.queues["queue-1"] = [
        FakeQueueItem("music"),
        FakeQueueItem("old-break", dj=True),
    ]
    assert await remove_legacy_dj_items(first, LOGGER, "ai_radio") is True
    assert first.player_queues.updates == [("queue-1", ["music"])]
    assert config.values[CONF_LEGACY_DJ_ITEMS_CLEANED] is True

    # the next start reads the marker and never scans the queue cache again
    second = FakeMass({"queue-1": [FakeQueueItem("late-break", dj=True)]}, {"queue-1"}, config)
    assert await remove_legacy_dj_items(second, LOGGER, "ai_radio") is False
    assert second.cache.calls == []
    assert second.player_queues.updates == []


async def test_a_long_queue_is_walked_in_pages() -> None:
    """A queue longer than one page is walked whole, so a clip on the last page goes too."""
    items = [FakeQueueItem(f"music-{index}") for index in range(1200)]
    items[0] = FakeQueueItem("old-break-first", dj=True)
    items[1100] = FakeQueueItem("old-break-last", dj=True)
    mass = FakeMass({"queue-1": items}, {"queue-1"})

    assert await remove_legacy_dj_items(mass, LOGGER, "ai_radio") is True

    left = [item.queue_item_id for item in mass.player_queues.queues["queue-1"]]
    assert "old-break-first" not in left
    assert "old-break-last" not in left
    assert len(left) == 1198
    # one update for the whole queue, not one per page
    assert len(mass.player_queues.updates) == 1

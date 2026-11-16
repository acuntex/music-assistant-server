"""Tests for the queue's reported next item, as the playback tracker sets it."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from music_assistant_models.enums import PlaybackState, RepeatMode
from music_assistant_models.queue_item import QueueItem

from music_assistant.controllers.player_queues.controller import PlayerQueuesController
from music_assistant.controllers.player_queues.playback_tracker import PlaybackTrackerMixin
from music_assistant.controllers.player_queues.queue_loader import QueueLoaderMixin


def _item(item_id: str, *, available: bool = True) -> QueueItem:
    """Return a queue item that may or may not be able to play."""
    return QueueItem(
        queue_id="q1", queue_item_id=item_id, name=item_id, duration=200, available=available
    )


def _tracker(
    items: list[QueueItem],
    *,
    repeat_mode: RepeatMode = RepeatMode.OFF,
    sleep_timer_target: Any = None,
    current: str = "one",
) -> tuple[Any, Any]:
    """
    Return a tracker harness plus the queue it reports on.

    The harness binds the controller's own lookups, so the reported next item goes through
    the repeat, wrap and sleep-timer rules the queue plays by.

    :param items: The queue's items, in playing order.
    :param repeat_mode: The queue's repeat setting.
    :param sleep_timer_target: The target the sleep timer stops at, if one is armed.
    :param current: The queue item the player reports as playing.
    """
    queue = SimpleNamespace(
        queue_id="q1",
        active=True,
        state=PlaybackState.PLAYING,
        flow_mode=False,
        repeat_mode=repeat_mode,
        corrected_elapsed_time=10.0,
        playback_speed=1.0,
        current_index=None,
        current_item=None,
        next_item=None,
    )
    data = SimpleNamespace(queue=queue, items=items, sleep_timer_target=sleep_timer_target)
    tracker = SimpleNamespace()
    tracker._queue_data = {"q1": data}
    tracker.queue_data_or_none = lambda queue_id: tracker._queue_data.get(queue_id)
    tracker.get_item = lambda queue_id, key: PlayerQueuesController.get_item(tracker, queue_id, key)
    tracker.index_by_id = lambda queue_id, item_id: PlayerQueuesController.index_by_id(
        tracker, queue_id, item_id
    )
    tracker._get_next_index = lambda queue_id, index, **kwargs: QueueLoaderMixin._get_next_index(
        tracker, queue_id, index, **kwargs
    )
    tracker.get_next_item = lambda queue_id, cur_index: PlayerQueuesController.get_next_item(
        tracker, queue_id, cur_index
    )
    tracker._parse_player_current_item_id = lambda _queue_id, _player: current
    return tracker, queue


def _next_item_id(tracker: Any, queue: Any) -> str | None:
    """Play the queue's item and return the next item it reports."""
    player = SimpleNamespace(corrected_elapsed_time=10.0)
    PlaybackTrackerMixin._update_current_index_from_player(tracker, queue, cast("Any", player))

    return queue.next_item.queue_item_id if queue.next_item is not None else None


def test_the_next_item_is_the_item_after_the_playing_one() -> None:
    """Without repeat settings the queue reports the item that follows it."""
    tracker, queue = _tracker([_item("one"), _item("two")])

    assert _next_item_id(tracker, queue) == "two"


def test_the_next_item_skips_an_item_that_cannot_play() -> None:
    """An unplayable item is no answer to 'what comes next', so the one behind it is."""
    tracker, queue = _tracker([_item("one"), _item("two", available=False), _item("three")])

    assert _next_item_id(tracker, queue) == "three"


def test_the_next_item_at_the_end_of_the_queue_is_none() -> None:
    """Nothing left to play is None, which end-of-queue and the sleep timer read."""
    tracker, queue = _tracker([_item("one"), _item("two", available=False)], current="one")

    assert _next_item_id(tracker, queue) is None


def test_repeat_one_reports_the_playing_item_as_the_next_one() -> None:
    """Repeat one plays the item again, so it is what the queue reports as coming next."""
    tracker, queue = _tracker(
        [_item("one"), _item("two")], repeat_mode=RepeatMode.ONE, current="one"
    )

    assert _next_item_id(tracker, queue) == "one"


def test_repeat_all_wraps_from_the_last_item_to_the_first() -> None:
    """With repeat all the queue starts over, so the first item is the next one."""
    tracker, queue = _tracker(
        [_item("one"), _item("two")], repeat_mode=RepeatMode.ALL, current="two"
    )

    assert _next_item_id(tracker, queue) == "one"


def test_an_armed_sleep_timer_reports_no_next_item_at_its_target() -> None:
    """A sleep timer ends the queue at that item, so nothing follows it."""
    tracker, queue = _tracker(
        [_item("one"), _item("two")],
        sleep_timer_target=SimpleNamespace(queue_item_id="one"),
        current="one",
    )

    assert _next_item_id(tracker, queue) is None

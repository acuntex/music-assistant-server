"""
One-off upgrades for the AI Radio plugin.

Kept apart from the runtime code: each cleanup here runs once against data a previous
release wrote, so it is idempotent, reports what it could not do, and never blocks the
provider start.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from music_assistant.controllers.player_queues.constants import (
    CACHE_CATEGORY_PLAYER_QUEUE_ITEMS,
    CACHE_CATEGORY_PLAYER_QUEUE_STATE,
)
from music_assistant.controllers.player_queues.helpers import committed_index

from .constants import ATTR_QUEUE_DJ, ATTR_SESSION_ID, CONF_LEGACY_DJ_ITEMS_CLEANED

if TYPE_CHECKING:
    from music_assistant_models.queue_item import QueueItem

    from music_assistant.mass import MusicAssistant

# the queue controller serves its items in pages, so a long queue is walked in several reads
QUEUE_PAGE_SIZE = 500


async def remove_legacy_dj_items(
    mass: MusicAssistant,
    logger: logging.Logger,
    instance_id: str,
    live_session_ids: Mapping[str, str] | None = None,
) -> bool:
    """
    Remove the DJ clips an older release spliced into the persisted queues, once.

    The DJ speaks over the queue's stream now, so a clip that is still an item of the
    queue is a leftover: it would air as its own row next to the break spoken over the
    boundary. Every persisted queue is checked before the marker is written, so a queue
    that could not be read (or was not loaded yet) is picked up by the next start instead
    of keeping its clip forever.

    :param mass: The MusicAssistant instance.
    :param logger: The provider's logger.
    :param instance_id: The provider instance the marker is stored on.
    :param live_session_ids: The session id of every queue's running DJ, so a clip that
        session spliced a moment ago is not taken for a leftover.
    :return: Whether this pass completed and wrote the marker.
    """
    try:
        if mass.config.get_raw_provider_config_value(
            instance_id, CONF_LEGACY_DJ_ITEMS_CLEANED, False
        ):
            return False
        if not mass.player_queues.all():
            # the players register after the provider loads and bring their queues with
            # them: scanning the cache now would only find queues that cannot be reached
            logger.debug("No queue is loaded yet - the old DJ clips wait for theirs")
            return False
        queue_ids = await _persisted_queue_ids(mass)
    except Exception as err:
        logger.warning("Could not read the queues for the DJ clip cleanup: %s", err)
        return False
    complete = True
    live_session_ids = live_session_ids or {}
    for queue_id in sorted(queue_ids):
        try:
            removed = _remove_queue_dj_items(mass, logger, queue_id, live_session_ids.get(queue_id))
        except Exception as err:
            # one unreadable queue must not keep the others' leftovers in place, and the
            # pass is retried in full on the next start
            complete = False
            logger.warning("Could not check queue %s for old DJ clips: %s", queue_id, err)
            continue
        if removed is None:
            complete = False
        elif removed:
            logger.info("Removed %d old DJ clip(s) from queue %s", removed, queue_id)
    if not complete:
        logger.info("Not every queue could be checked for old DJ clips - retrying on next start")
        return False
    try:
        mass.config.set_raw_provider_config_value(
            instance_id, CONF_LEGACY_DJ_ITEMS_CLEANED, True, immediate=True
        )
    except Exception as err:
        logger.warning("Could not mark the DJ clip cleanup as done: %s", err)
        return False
    return True


def _remove_queue_dj_items(
    mass: MusicAssistant,
    logger: logging.Logger,
    queue_id: str,
    live_session_id: str | None,
) -> int | None:
    """
    Remove the DJ clips of one queue.

    :param mass: The MusicAssistant instance.
    :param logger: The provider's logger.
    :param queue_id: The queue to clean.
    :param live_session_id: The session id of the queue's running DJ, if it has one.
    :return: How many items were removed, or None when the queue is not loaded.
    """
    queue = mass.player_queues.get(queue_id)
    if queue is None:
        # its player did not register yet; the next start can still reach its items
        logger.debug(
            "Queue %s is not loaded yet - its old DJ clips wait for the next start", queue_id
        )
        return None
    items = _queue_items(mass, queue_id)
    # everything up to the guard is what the player already owns: taking an item out there
    # would shift the indexes the queue's load walks, so it keeps playing or skips it itself
    guard_index = committed_index(queue)
    if guard_index is None:
        guard_index = -1
    kept: list[QueueItem] = []
    removed = 0
    for index, item in enumerate(items):
        if not item.extra_attributes.get(ATTR_QUEUE_DJ):
            kept.append(item)
            continue
        if index <= guard_index:
            kept.append(item)
            continue
        if live_session_id is not None and (
            item.extra_attributes.get(ATTR_SESSION_ID) == live_session_id
        ):
            # the session spliced this clip a moment ago and is planning again right now
            kept.append(item)
            continue
        removed += 1
    if not removed:
        return 0
    mass.player_queues.update_items(queue_id, kept)
    return removed


def _queue_items(mass: MusicAssistant, queue_id: str) -> list[QueueItem]:
    """
    Return all items of a queue.

    :param mass: The MusicAssistant instance.
    :param queue_id: The queue to read.
    """
    items: list[QueueItem] = []
    offset = 0
    while True:
        page = mass.player_queues.items(queue_id, limit=QUEUE_PAGE_SIZE, offset=offset)
        items.extend(page)
        if len(page) < QUEUE_PAGE_SIZE:
            return items
        offset += QUEUE_PAGE_SIZE


async def _persisted_queue_ids(mass: MusicAssistant) -> set[str]:
    """
    Return the ids of the queues a previous run persisted.

    :param mass: The MusicAssistant instance.
    """
    queue_ids: set[str] = set()
    for category in (CACHE_CATEGORY_PLAYER_QUEUE_STATE, CACHE_CATEGORY_PLAYER_QUEUE_ITEMS):
        stored = await mass.cache.get_all(provider=mass.player_queues.domain, category=category)
        queue_ids.update(stored)
    return queue_ids

"""Unit tests for the in-memory terms of one queue's AI DJ."""

from __future__ import annotations

import pytest

from music_assistant.providers.ai_radio.terms import (
    TERMS_PER_QUEUE,
    BoundaryState,
    ProgramTerm,
    TermState,
    TermStore,
    term_key,
)


def _term(
    item_id: str = "a",
    successor_id: str | None = "b",
    section_ids: tuple[str, ...] = ("transition",),
    head_seconds: float = 0.0,
) -> ProgramTerm:
    """Return a minimal term for the given boundary."""
    return ProgramTerm(
        queue_item_id=item_id,
        successor_id=successor_id,
        section_ids=section_ids,
        head_seconds=head_seconds,
    )


def test_the_key_is_the_boundary() -> None:
    """A term is identified by the boundary it speaks at, never by its sections."""
    term = _term(item_id="a", successor_id="b", section_ids=("one", "two"))
    assert term.key == ("a", "b")
    assert term.key == term_key("a", "b")


def test_an_outro_key_has_no_successor() -> None:
    """The end of a queue is a boundary like any other."""
    assert _term(item_id="a", successor_id=None).key == ("a", None)


def test_a_new_term_is_decided() -> None:
    """Deciding is the first state, and it is not terminal."""
    term = _term()
    assert term.state is TermState.DECIDED
    assert term.is_terminal is False
    assert term.dropped_reason is None


def test_rendering_and_scheduling_keep_the_term_alive() -> None:
    """A term that is rendered or placed is still to be heard."""
    term = _term()
    term.mark_rendering()
    assert term.state is TermState.RENDERING
    term.mark_scheduled(start_frame=480, pcm_len=960, delay_seconds=4.0, seconds_left=9.0)
    assert term.state is TermState.SCHEDULED
    assert term.scheduled_frame == 480
    assert term.pcm_len == 960
    assert term.delay_seconds == 4.0
    assert term.is_terminal is False


def test_a_dropped_term_carries_its_reason() -> None:
    """Why a term never aired is part of its state."""
    term = _term()
    term.mark_dropped("skipped")
    assert term.state is TermState.DROPPED
    assert term.dropped_reason == "skipped"
    assert term.is_terminal is True


def test_an_aired_term_stays_aired() -> None:
    """A term that was heard can not be dropped afterwards."""
    term = _term()
    term.mark_scheduled(start_frame=0, pcm_len=0, delay_seconds=0.0, seconds_left=1.0)
    term.mark_aired()
    term.mark_dropped("too_late")
    assert term.state is TermState.AIRED
    assert term.dropped_reason is None


def test_a_restore_starts_with_an_empty_store() -> None:
    """Terms live in the session, so a fresh store starts empty."""
    store = TermStore("queue")
    store.add(_term())
    assert TermStore("queue").active() == ()


def test_adding_the_same_boundary_twice_keeps_one_term() -> None:
    """Every pass over a queue re-decides its gaps, and must not duplicate a term."""
    store = TermStore("queue")
    first = store.add(_term())
    second = store.add(_term())
    assert second is first
    assert len(store) == 1


def test_one_boundary_holds_one_term_whatever_its_sections() -> None:
    """Another section set at the same boundary does not buy a second term there."""
    store = TermStore("queue")
    first = store.add(_term(section_ids=("weather",)))
    second = store.add(_term(section_ids=("news",)))
    assert second is first
    assert len(store) == 1


def test_a_different_boundary_is_its_own_term() -> None:
    """Another successor is another boundary."""
    store = TermStore("queue")
    store.add(_term(item_id="a", successor_id="b"))
    store.add(_term(item_id="a", successor_id="c"))
    assert len(store) == 2


def test_i_can_look_a_term_up_by_its_key() -> None:
    """The engine drives a term through its key after a lookup."""
    store = TermStore("queue")
    term = store.add(_term())
    assert store.get(term.key) is term
    assert store.get(term_key("x", "y")) is None


def test_dropping_a_term_by_key_records_the_reason() -> None:
    """The engine drops a boundary it can no longer reach."""
    store = TermStore("queue")
    term = store.add(_term())
    dropped = store.drop(term.key, "skipped")
    assert dropped is term
    assert term.state is TermState.DROPPED
    assert term.dropped_reason == "skipped"
    assert store.active() == ()


def test_dropping_a_term_twice_keeps_the_first_reason() -> None:
    """The reason a term first ended is the one the log tells."""
    store = TermStore("queue")
    term = store.add(_term())
    store.drop(term.key, "skipped")
    assert store.drop(term.key, "too_late") is None
    assert term.dropped_reason == "skipped"


def test_drop_all_ends_every_active_term() -> None:
    """A stopped queue or a replaced bus ends what it still had to say."""
    store = TermStore("queue")
    first = store.add(_term(item_id="a"))
    second = store.add(_term(item_id="b", successor_id=None))
    aired = store.add(_term(item_id="c", section_ids=("news",)))
    assert aired is not None
    aired.mark_aired()
    ended = store.drop_all("flow_ended")
    assert {term.key for term in ended} == {first.key, second.key}
    assert first.dropped_reason == "flow_ended"
    assert second.dropped_reason == "flow_ended"
    assert len(store.active()) == 0


def test_drop_all_leaves_terminal_terms_alone() -> None:
    """A term that already aired is not rewritten by a later stop."""
    store = TermStore("queue")
    aired = store.add(_term(item_id="a"))
    aired.mark_aired()
    assert store.drop_all("flow_ended") == ()
    assert aired.state is TermState.AIRED


def test_the_room_counts_the_terms_still_to_be_heard() -> None:
    """A term that aired keeps its boundary, but it takes no room a pass could decide."""
    store = TermStore("queue", max_terms=2)
    first = store.add(_term(item_id="a", section_ids=("one",)))
    assert first is not None
    store.add(_term(item_id="b", section_ids=("one",)))
    assert store.room() == 0

    first.mark_aired()

    assert store.room() == 1
    # the reserve a pass has to keep free comes off the same answer
    assert store.room(reserve=1) == 0
    assert len(store) == 2


def test_the_store_is_bounded() -> None:
    """A long session does not grow its term store without end."""
    store = TermStore("queue")
    for index in range(TERMS_PER_QUEUE):
        store.add(_term(item_id=f"i{index}"))
    assert len(store) == TERMS_PER_QUEUE
    for index in range(TERMS_PER_QUEUE):
        assert store.get(term_key(f"i{index}", "b")) is not None
    store.add(_term(item_id="new"))
    assert len(store) == TERMS_PER_QUEUE
    # terms are added in playing order, so the last one is the boundary furthest away: that
    # is the one to give up, never the nearest one that is about to be due
    assert store.get(term_key("i0", "b")) is not None
    assert store.get(term_key(f"i{TERMS_PER_QUEUE - 1}", "b")) is None
    assert store.get(term_key("new", "b")) is not None


def test_an_ended_term_is_given_up_before_a_live_one() -> None:
    """A term that already aired is cheaper to forget than one still to come."""
    store = TermStore("queue")
    aired = store.add(_term(item_id="a"))
    aired.mark_aired()
    for index in range(TERMS_PER_QUEUE - 1):
        store.add(_term(item_id=f"i{index}"))
    live = store.add(_term(item_id="live"))
    assert len(store) == TERMS_PER_QUEUE
    assert store.get(term_key("a", "b")) is None
    assert live in store.active()


def test_evicting_a_live_term_gives_up_the_farthest() -> None:
    """The nearest boundary keeps its term, the one furthest out is the one dropped."""
    store = TermStore("queue", max_terms=2)
    near = store.add(_term(item_id="a"))
    far = store.add(_term(item_id="b"))
    store.add(_term(item_id="c"))
    assert far.state is TermState.DROPPED
    assert far.dropped_reason == "ledger_evicted"
    assert near.state is TermState.DECIDED
    assert len(store) == 2


def test_an_evicted_live_term_is_reported() -> None:
    """A term the bounded store gives up is handed to the caller, so a bus can stop it."""
    evicted: list[ProgramTerm] = []
    store = TermStore("queue", max_terms=1, on_evicted=evicted.append)
    first = store.add(_term(item_id="a"))
    store.add(_term(item_id="b"))
    assert evicted == [first]
    assert first.dropped_reason == "ledger_evicted"


def test_an_evicted_terminal_term_is_not_reported() -> None:
    """A term that already aired needs no bus to stop it."""
    evicted: list[ProgramTerm] = []
    store = TermStore("queue", max_terms=2, on_evicted=evicted.append)
    aired = store.add(_term(item_id="a"))
    aired.mark_aired()
    store.add(_term(item_id="b"))
    store.add(_term(item_id="c"))
    assert evicted == []


def test_a_rendered_term_keeps_its_audio_until_it_is_placed() -> None:
    """A clip written while nothing is heard is held for the resume, not thrown away."""
    term = _term()
    pcm = b"\x00\x00" * 8
    term.keep_rendered(pcm)
    assert term.rendered_pcm == pcm
    assert term.state is TermState.DECIDED


def test_a_placed_term_releases_its_audio() -> None:
    """The bus owns the audio once the term is placed, so the term keeps only its size."""
    term = _term()
    term.keep_rendered(b"\x00\x00" * 8)
    term.mark_scheduled(start_frame=0, pcm_len=16, delay_seconds=1.0, seconds_left=2.0)
    assert term.rendered_pcm is None
    assert term.pcm_len == 16


def test_a_dropped_term_releases_its_audio() -> None:
    """A term that never airs does not hold a clip for the rest of the session."""
    term = _term()
    term.keep_rendered(b"\x00\x00" * 8)
    term.mark_dropped("paused too long")
    assert term.rendered_pcm is None


def test_a_terminal_boundary_can_be_planned_again() -> None:
    """Repeat-one brings the same boundary back, and it earns a fresh term."""
    store = TermStore("queue")
    first = store.add(_term())
    first.mark_aired()
    second = store.add(_term())
    assert second is not first
    assert second.state is TermState.DECIDED
    assert len(store) == 1


def test_a_store_holds_the_program_of_its_queue() -> None:
    """The station and host a queue's terms render with are kept with the store."""
    store = TermStore("queue")
    assert store.program is None
    store.program = {"host_id": "rick"}
    assert store.program == {"host_id": "rick"}


@pytest.mark.parametrize("state", [TermState.AIRED, TermState.DROPPED])
def test_both_ending_states_are_terminal(state: TermState) -> None:
    """Only aired and dropped end a term."""
    term = _term()
    term.state = state
    assert term.is_terminal is True


def test_a_boundary_nobody_looked_at_is_open() -> None:
    """A boundary the store never saw is open, and nothing is settled about it."""
    store = TermStore("queue")

    assert store.state(("a", "b")) is BoundaryState.OPEN
    assert store.is_settled(("a", "b")) is False
    assert store.term_of(("a", "b")) is None


def test_a_boundary_with_a_term_is_decided() -> None:
    """A living term settles the boundary it speaks at."""
    store = TermStore("queue")
    term = store.add(_term())

    assert store.state(("a", "b")) is BoundaryState.DECIDED
    assert store.is_settled(("a", "b")) is True
    assert store.term_of(("a", "b")) is term


def test_a_boundary_a_pass_left_silent_is_decided_without_a_term() -> None:
    """A pass that looked at a boundary and planned nothing there settled it too."""
    store = TermStore("queue")

    store.settle([("a", "b")])

    assert store.state(("a", "b")) is BoundaryState.DECIDED
    assert store.term_of(("a", "b")) is None
    assert store.active() == ()
    # a later pass may still decide to speak there
    term = store.add(_term())
    assert store.term_of(("a", "b")) is term


def test_a_pass_that_finds_nothing_again_settles_a_dropped_boundary() -> None:
    """A boundary whose term was dropped is settled by the pass that plans nothing there."""
    store = TermStore("queue")
    term = store.add(_term())
    store.drop(term.key, "skipped")
    assert store.state(("a", "b")) is BoundaryState.OPEN

    store.settle([("a", "b")])

    assert store.state(("a", "b")) is BoundaryState.DECIDED
    assert store.term_of(("a", "b")) is None
    assert term.dropped_reason == "skipped"


def test_a_pass_never_reopens_what_a_term_already_spoke() -> None:
    """An aired term's boundary stays closed, whatever a later pass evaluated."""
    store = TermStore("queue")
    term = store.add(_term())
    term.mark_aired()

    store.settle([("a", "b")])

    assert store.state(("a", "b")) is BoundaryState.DECIDED
    assert store.term_of(("a", "b")) is term


def test_a_silent_boundary_can_be_woken_again() -> None:
    """A settled boundary without a term is what a missed pass opens again."""
    store = TermStore("queue")
    store.settle([("a", "b")])

    store.reopen(("a", "b"))

    assert store.state(("a", "b")) is BoundaryState.OPEN
    assert ("a", "b") not in store.boundaries()


def test_waking_a_boundary_that_holds_a_term_keeps_the_term() -> None:
    """A term is not displaced by a pass that thinks the boundary was missed."""
    store = TermStore("queue")
    term = store.add(_term())

    store.reopen(("a", "b"))

    assert store.term_of(("a", "b")) is term
    assert store.state(("a", "b")) is BoundaryState.DECIDED


@pytest.mark.parametrize("reason", ["no_duration", "unknown_boundary", "format", "no_successor"])
def test_a_term_that_cannot_be_served_closes_its_boundary(reason: str) -> None:
    """A boundary with no way to be spoken is closed for good, with the reason kept."""
    store = TermStore("queue")
    term = store.add(_term())

    store.drop(term.key, reason)

    assert store.state(("a", "b")) is BoundaryState.UNREACHABLE
    assert store.reason(("a", "b")) == reason
    assert store.state(("a", "b")) is not BoundaryState.OPEN


@pytest.mark.parametrize("reason", ["skipped", "too_late", "seeked", "render_failed"])
def test_a_term_that_only_missed_its_moment_reopens_its_boundary(reason: str) -> None:
    """A boundary that was merely too late or skipped is open for the next pass."""
    store = TermStore("queue")
    term = store.add(_term())

    store.drop(term.key, reason)

    assert store.state(("a", "b")) is BoundaryState.OPEN
    assert store.reason(("a", "b")) is None


def test_a_closed_boundary_is_never_decided_again() -> None:
    """Nothing decides a boundary twice once it was found unservable."""
    store = TermStore("queue")
    term = store.add(_term())
    store.drop(term.key, "no_duration")

    assert store.add(_term()) is None
    assert store.active() == ()
    assert store.term_of(("a", "b")) is term
    assert store.state(("a", "b")) is BoundaryState.UNREACHABLE


def test_an_aired_term_keeps_its_boundary_settled() -> None:
    """What the listener heard is decided: no pass opens it again."""
    store = TermStore("queue")
    term = store.add(_term())
    term.mark_aired()

    assert store.state(("a", "b")) is BoundaryState.DECIDED
    assert store.is_settled(("a", "b")) is True


def test_a_closed_boundary_is_not_evicted_for_room() -> None:
    """A closed boundary holds no slot, so room for a term never reopens it."""
    store = TermStore("queue", max_terms=1)
    closed = store.add(_term(item_id="closed", successor_id="next"))
    store.drop(closed.key, "no_duration")
    live = store.add(_term(item_id="live", successor_id="next"))

    assert store.term_of(closed.key) is closed
    assert store.term_of(live.key) is live
    assert len(store.active()) == 1


def test_the_store_forgets_the_boundaries_of_an_item_that_left() -> None:
    """A boundary of an item the queue no longer holds says nothing about the next pass."""
    store = TermStore("queue")
    closed = store.add(_term(item_id="gone", successor_id="next"))
    store.drop(closed.key, "no_duration")
    store.settle([("also-gone", "next")])
    live = store.add(_term(item_id="live", successor_id="next"))

    store.retain({"live", "next"})

    assert store.state(("gone", "next")) is BoundaryState.OPEN
    assert store.state(("also-gone", "next")) is BoundaryState.OPEN
    assert store.term_of(live.key) is live


def test_a_living_term_is_kept_when_its_item_is_pruned() -> None:
    """Only what can no longer air is forgotten: a living term still has to be driven."""
    store = TermStore("queue")
    term = store.add(_term(item_id="gone", successor_id="next"))

    store.retain({"next"})

    assert store.term_of(term.key) is term


def test_a_boundary_decided_again_keeps_the_place_it_plays_at() -> None:
    """Eviction follows the play position, not the order the boundaries were added in."""
    store = TermStore("queue", max_terms=3)
    # the store fills up in playing order, and the nearest boundary is settled without a term
    store.settle([("a", "b")])
    middle = store.add(_term(item_id="b", successor_id="c"), position=1)
    far = store.add(_term(item_id="c", successor_id="d"), position=5)
    # a later pass decides the nearest boundary again, so it is the store's newest entry
    store.reopen(("a", "b"))
    near = store.add(_term(item_id="a", successor_id="b"), position=0)
    assert store.boundaries()[-1] == near.key
    assert len(store) == 3

    store.add(_term(item_id="d", successor_id="e"), position=9)

    # the boundary that is due next survived, the one furthest out made room
    assert store.term_of(near.key) is near
    assert store.term_of(middle.key) is middle
    assert store.term_of(far.key) is None
    assert len(store) == 3


def test_a_full_store_never_gives_up_the_boundary_that_plays_now() -> None:
    """The boundary the running item ends at is due next, whatever else the store holds."""
    evicted: list[ProgramTerm] = []
    store = TermStore("queue", max_terms=3, on_evicted=evicted.append)
    running = store.add(_term(item_id="a", successor_id="b"), position=0)
    store.set_running(running.key)
    store.add(_term(item_id="b", successor_id="c"), position=1)
    other = store.add(_term(item_id="c", successor_id="d"), position=2)

    store.add(_term(item_id="d", successor_id="e"), position=3)

    assert store.term_of(running.key) is running
    assert evicted == [other]
    assert len(store) == 3


def test_the_running_boundary_is_kept_though_it_sorts_the_furthest_out() -> None:
    """A term that is on the bus now is never the one a pass gives up for room."""
    store = TermStore("queue", max_terms=2)
    running = store.add(_term(item_id="b", successor_id="c"), position=99)
    store.set_running(running.key)
    near = store.add(_term(item_id="a", successor_id="b"), position=0)

    store.add(_term(item_id="c", successor_id="d"), position=1)

    assert store.term_of(running.key) is running
    assert store.term_of(near.key) is None
    assert len(store) == 2

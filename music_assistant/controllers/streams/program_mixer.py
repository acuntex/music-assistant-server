"""
Mix scheduled programs into the PCM of one live flow session.

The bus counts every frame the session emits and sums each program over the music while the
frame it was scheduled for goes by, so a program needs no stream of its own: the flow is neither
restarted nor reopened, no second ffmpeg input is used, and no byte of it is lost or moved. A
chunk that ends on a partial frame carries those last bytes into the next chunk, so the stream
the bus lets out stays frame aligned and no anchor behind it shifts; the chunk it leaves in can
therefore be up to one frame short of the bytes that came in, and one frame long where it took
the rest of the frame before it. Where a program lands, and which program gives way for another,
is decided by the rules in program_schedule; this module only turns a placement into audio.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from music_assistant_models.enums import ContentType

from music_assistant.controllers.streams.constants import VOICE_OVER_RAMP
from music_assistant.controllers.streams.program_schedule import (
    ProgramPriority,
    ProgramWindow,
    ScheduleReason,
    split_window,
    window_collisions,
)
from music_assistant.controllers.streams.smart_fades.filters import (
    MIX_CEILING_DB,
    VOICE_OVER_DUCK_DEPTH,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from music_assistant_models.media_items import AudioFormat


# how long a bus waits before it reports the carried bytes of a partial frame again
UNALIGNED_WARN_SECONDS = 60.0
# how much audio the bus mixes before it hands the event loop back to the stream, so a
# chunk that arrives in one piece (a crossfade, a long source read) does not stall it
MIX_SLICE_SECONDS = 0.1


def _pcm_mix_codec(pcm_format: AudioFormat) -> str | None:
    """
    Return the codec a program is mixed in for the given PCM format, if it can be mixed.

    :param pcm_format: PCM format of the flow's audio.
    """
    if pcm_format.content_type == ContentType.PCM_F32LE and pcm_format.bit_depth == 32:
        return "f32"
    if pcm_format.content_type == ContentType.PCM_S16LE and pcm_format.bit_depth == 16:
        return "s16"
    return None


@dataclass(frozen=True)
class BusPlacement:
    """A program the bus took: where it starts and how much of it is still to be heard."""

    # frame the voice starts at: the anchor the caller asked for, or the bus's produced edge
    start_frame: int
    # frames of the program the bus really plays; shorter than what was offered when the
    # placement had to drop the front of it
    frames: int


class _ProgramOverlay:
    """
    Sum a scheduled program over the flow's music, starting at an absolute frame.

    The music keeps its exact length; it is ducked under the voice where the voice plays and
    the voice is added sample-wise. 32-bit float and 16-bit integer PCM are mixed, anything
    else can not be placed at all.
    """

    def __init__(
        self,
        voice_pcm: bytes,
        pcm_format: AudioFormat,
        start_frame: int,
        stop_frame: int | None = None,
        priority: ProgramPriority = ProgramPriority.TERM,
        voice_gain: float = 1.0,
        duck_depth: float | None = None,
        on_dropped: Callable[[], None] | None = None,
    ) -> None:
        """
        Prepare the program.

        :param voice_pcm: The program's audio, decoded to the flow's PCM format.
        :param pcm_format: PCM format of both streams.
        :param start_frame: Frame of the session at which the voice starts.
        :param stop_frame: Frame the program may not be heard past, if any.
        :param priority: How strongly the program claims its window.
        :param voice_gain: Linear gain the program's own audio is mixed in with.
        :param duck_depth: Share of the music the ducking pulls down, or None for the depth
            a boundary's voice-over uses.
        :param on_dropped: Called when the program is dropped before its first frame was
            produced, so whoever holds its audio can keep it for the item it belongs to.
        """
        self._frame_size = (pcm_format.bit_depth // 8) * pcm_format.channels
        self._channels = pcm_format.channels
        self._sample_rate = pcm_format.sample_rate
        self._codec = _pcm_mix_codec(pcm_format)
        # the sample format the chunk is read and written in; the sum itself is computed in
        # double and rounded back into it, as the frame-by-frame sum was
        self._dtype = "<f4" if self._codec == "f32" else "<i2"
        self.label = ""
        # the level the queue mixes its programs at, taken when the program is placed: a
        # program that already airs keeps the level it started with
        self._voice_gain = voice_gain
        self._duck_depth = VOICE_OVER_DUCK_DEPTH if duck_depth is None else duck_depth
        self.voice_pcm = voice_pcm
        self.voice_frames = len(voice_pcm) // self._frame_size if self._frame_size else 0
        # the voice's samples, read once and kept for every chunk it is mixed into
        self._voice_samples: np.ndarray | None = None
        self.start_frame = max(0, start_frame)
        self._ramp = max(1, int(VOICE_OVER_RAMP * self._sample_rate))
        # the duck-in ramp never lasts longer than the voice: a shorter one would still be
        # on its way down when the voice ends, stepping instead of reaching full duck
        self.duck_head = min(self._ramp, max(1, self.voice_frames))
        self.duck_tail = self._ramp
        # the mix is capped at the ceiling the boundary's voice-over limiter holds, so a
        # full-scale voice summed over full-scale music saturates instead of wrapping
        self.ceiling = 10 ** (MIX_CEILING_DB / 20)
        self._stop_frame = stop_frame
        self.priority = priority
        self.on_dropped = on_dropped
        # identity the bus can hand out in a window description and match again
        self.key = 0
        self._voice_end = self.start_frame + self.voice_frames
        self.end_frame = self._end_of(self._voice_end + self.duck_tail)

    @property
    def usable(self) -> bool:
        """Return whether there is a voice to mix at all."""
        return self._codec is not None and self._frame_size > 0 and self.voice_frames > 0

    def covers(self, first_frame: int, frames: int) -> bool:
        """
        Return whether the given frame range touches the program at all.

        :param first_frame: First frame of the range.
        :param frames: Number of frames in the range.
        """
        return (
            self.usable and first_frame < self.end_frame and first_frame + frames > self.start_frame
        )

    def trim_to(self, frame: int) -> bool:
        """
        Drop the voice up to the given frame, so a late anchor starts where the flow is.

        :param frame: First frame that is still to be produced.
        :return: Whether anything of the voice is left to mix.
        """
        if frame <= self.start_frame:
            return True
        skip = min(frame - self.start_frame, self.voice_frames)
        self.voice_pcm = self.voice_pcm[skip * self._frame_size :]
        self._voice_samples = None
        self.voice_frames -= skip
        self.start_frame += skip
        self.duck_head = min(self._ramp, max(1, self.voice_frames))
        self._voice_end = self.start_frame + self.voice_frames
        self.end_frame = self._end_of(self._voice_end + self.duck_tail)
        return self.usable

    def mix(self, chunk: bytes, first_frame: int) -> bytes:
        """
        Return the chunk with the voice summed in where it plays.

        Only the frames the program covers are read, summed and written back; everything
        outside its window is passed through as the bytes it came in as.

        :param chunk: One chunk of music, in the flow's PCM format.
        :param first_frame: Frame of the session the chunk starts at.
        """
        frames = len(chunk) // self._frame_size if self._frame_size else 0
        if not self.covers(first_frame, frames):
            return chunk
        first = max(0, self.start_frame - first_frame)
        last = min(frames, self.end_frame - first_frame)
        if last <= first:
            return chunk
        head = first * self._frame_size
        tail = last * self._frame_size
        return (
            chunk[:head]
            + self._mix_region(chunk[head:tail], first_frame + first, last - first)
            + chunk[tail:]
        )

    def _end_of(self, frame: int) -> int:
        """Return the program's end, kept inside the window it may be heard in."""
        if self._stop_frame is None:
            return frame
        return max(self.start_frame, min(frame, self._stop_frame))

    def _mix_region(self, region: bytes, first_position: int, frames: int) -> bytes:
        """
        Return the given frames of music, ducked and with the voice summed in.

        :param region: The frames to mix, in the flow's PCM format.
        :param first_position: Frame of the session the region starts at.
        :param frames: Number of frames in the region.
        """
        # read where the audio is mixed: numpy is a heavy module the server's start-up
        # import graph has to stay free of
        import numpy as np  # noqa: PLC0415

        channels = self._channels
        positions = np.arange(first_position, first_position + frames, dtype=np.int64)
        relative = positions - self.start_frame
        # the boundary's voice-over pulls the music down over the ramp, holds it while the
        # voice plays and lets it back up over the tail once the voice ended
        ducked = self._duck_depth * np.minimum(1.0, (relative + 1) / self.duck_head)
        after = positions >= self._voice_end
        if after.any():
            ducked[after] = self._duck_depth * np.maximum(
                0.0, (self.end_frame - positions[after]) / self.duck_tail
            )
        music = (
            np.frombuffer(region, dtype=self._dtype).astype(np.float64).reshape(frames, channels)
        )
        spoken = (relative >= 0) & (relative < self.voice_frames)
        if self._codec == "f32":
            mixed = music * (1.0 - ducked)[:, None]
            if spoken.any():
                mixed[spoken] += self._voice(relative[spoken]) * self._voice_gain
            mixed = np.clip(mixed, -self.ceiling, self.ceiling)
            return mixed.astype(self._dtype).tobytes()
        # 16-bit PCM is mixed in whole steps: the ducked music and the program are each cut
        # to one before they are added, the way the frame-by-frame sum cut them
        steps = (music * (1.0 - ducked)[:, None]).astype(np.int64)
        if spoken.any():
            steps[spoken] += (self._voice(relative[spoken]) * self._voice_gain).astype(np.int64)
        return np.clip(steps, -32768, 32767).astype(self._dtype).tobytes()

    def _voice(self, frames: np.ndarray) -> np.ndarray:
        """
        Return the voice's samples at the given frames of the program.

        :param frames: Frame of the program to read.
        """
        if self._voice_samples is None:
            import numpy as np  # noqa: PLC0415

            self._voice_samples = (
                np.frombuffer(self.voice_pcm, dtype=self._dtype)
                .astype(np.float64)
                .reshape(self.voice_frames, self._channels)
            )
        return cast("np.ndarray", self._voice_samples[frames])


class FlowProgramBus:
    """
    Mix scheduled programs into the PCM of one flow session.

    The bus counts every frame the session emits and mixes each program while the frame
    it was scheduled for goes by, so a program needs no stream of its own: the flow is
    neither restarted nor reopened, no second ffmpeg input is used and no byte of it is lost.
    A chunk that ends in the middle of a frame carries those last bytes into the next chunk,
    so what leaves the bus is whole frames: a chunk can be up to one frame shorter than the
    bytes that came in, and one frame longer where it carries the rest of the frame before it.
    """

    def __init__(self, pcm_format: AudioFormat, logger: logging.Logger) -> None:
        """
        Initialize the bus for one flow session.

        :param pcm_format: PCM format the flow produces.
        :param logger: Logger of the audio sub-controller.
        """
        self._pcm_format = pcm_format
        self._logger = logger
        self._frame_size = (pcm_format.bit_depth // 8) * pcm_format.channels
        # the frames one slice of the mix holds; frame aligned, so a slice never splits one
        self._slice_bytes = self._frame_size * max(
            1, int(MIX_SLICE_SECONDS * pcm_format.sample_rate)
        )
        self.produced_frames = 0
        self._overlays: list[_ProgramOverlay] = []
        self._next_key = 0
        # why the last placement was refused, for the caller that asked for it
        self.last_refusal: ScheduleReason | None = None
        # bytes a chunk ended with that do not make up a whole frame yet
        self._residue = b""
        # when the carried bytes of a partial frame were last reported
        self._carry_reported_at: float | None = None
        # how loud the queue mixes its programs in and how far it ducks the music under them
        self._voice_gain = 1.0
        self._duck_depth: float | None = None

    @property
    def active(self) -> bool:
        """Return whether a scheduled program is still to be heard."""
        return bool(self._overlays)

    @property
    def pcm_format(self) -> AudioFormat:
        """Return the PCM format of the flow this bus belongs to."""
        return self._pcm_format

    @property
    def can_mix(self) -> bool:
        """Return whether a program can be mixed in the flow's PCM format at all."""
        return self._frame_size > 0 and _pcm_mix_codec(self._pcm_format) is not None

    @property
    def frame_size(self) -> int:
        """Return the size in bytes of one frame of the flow's PCM format."""
        return self._frame_size

    @property
    def voice_gain(self) -> float:
        """Return the gain the flow's programs are mixed in with."""
        return self._voice_gain

    @property
    def duck_depth(self) -> float:
        """Return the share of the music the flow's programs duck it down by."""
        return VOICE_OVER_DUCK_DEPTH if self._duck_depth is None else self._duck_depth

    def set_levels(self, *, voice_gain: float = 1.0, duck_depth: float | None = None) -> None:
        """
        Set the level this flow mixes its programs at, and how far it ducks the music.

        The level belongs to the queue, not to one program: the streams controller sets it
        from the queue's own setting, so every program of the queue is mixed the same way,
        the announcements included. A program that is already placed keeps the level it was
        placed with.

        :param voice_gain: Linear gain applied to a program's own audio, before it is summed
            over the music, so the music under it keeps its level.
        :param duck_depth: Share of the music the ducking pulls down, or None to keep the
            depth a boundary's voice-over uses.
        """
        self._voice_gain = voice_gain
        self._duck_depth = duck_depth

    def schedule(
        self,
        pcm: bytes,
        start_frame: int | None,
        *,
        label: str,
        stop_frame: int | None = None,
        priority: ProgramPriority = ProgramPriority.TERM,
        on_dropped: Callable[[], None] | None = None,
    ) -> int | None:
        """
        Place a program at a frame of the session.

        A frame that was already produced can not be mixed any more: the program is moved
        forward to the current frame (and logged as late) or dropped when its whole window
        has already gone by. A frame inside a program that is already scheduled is moved to
        the end of that program's window, so two programs never duck the music at once.

        :param pcm: The program's audio, in the flow's PCM format.
        :param start_frame: Frame the voice starts at, or None to start at the current frame.
        :param label: Name of the program, for the log.
        :param stop_frame: Frame the program may not be heard past, if any. A program that
            ends where an item ends is kept inside that item, so its duck ramp can not leak
            into the item after it.
        :param priority: How strongly the program claims its window.
        :param on_dropped: Called when the program is dropped before its first frame was
            produced, for a caller that took its audio out of a longer stream.
        :return: The frame it was anchored at, or None when it can not be mixed.
        """
        placement = self.place(
            pcm,
            start_frame,
            label=label,
            stop_frame=stop_frame,
            priority=priority,
            on_dropped=on_dropped,
        )
        return None if placement is None else placement.start_frame

    def place(
        self,
        pcm: bytes,
        start_frame: int | None,
        *,
        label: str,
        stop_frame: int | None = None,
        priority: ProgramPriority = ProgramPriority.TERM,
        on_dropped: Callable[[], None] | None = None,
    ) -> BusPlacement | None:
        """
        Place a program and answer with what the bus took of it, not only its anchor.

        Same placement as :meth:`schedule`, for a caller that has to record how much of the
        program is really played: the answer belongs to this one placement, so it can not be
        confused with another program that carries the same label.

        :param pcm: The program's audio, in the flow's PCM format.
        :param start_frame: Frame the voice starts at, or None to start at the current frame.
        :param label: Name of the program, for the log.
        :param stop_frame: Frame the program may not be heard past, if any.
        :param priority: How strongly the program claims its window.
        :param on_dropped: Called when the program is dropped before its first frame was
            produced.
        :return: The placement, or None when it can not be mixed.
        """
        self.last_refusal = None
        if not pcm or not self._frame_size:
            self._logger.debug("PROGRAM-BUS schedule label=%s dropped: no audio", label)
            self.last_refusal = ScheduleReason.FORMAT
            return None
        frames = len(pcm) // self._frame_size
        anchor = self.produced_frames if start_frame is None else start_frame
        anchor = self._clear_of_scheduled(anchor, frames)
        overlay = _ProgramOverlay(
            pcm,
            self._pcm_format,
            anchor,
            stop_frame,
            priority,
            voice_gain=self._voice_gain,
            duck_depth=self._duck_depth,
            on_dropped=on_dropped,
        )
        overlay.label = label
        overlay.key = self._next_key
        self._next_key += 1
        if not overlay.usable:
            self._logger.debug(
                "PROGRAM-BUS schedule label=%s dropped: %s can not be mixed",
                label,
                self._pcm_format.content_type,
            )
            self.last_refusal = ScheduleReason.FORMAT
            return None
        if overlay.end_frame <= overlay.start_frame:
            # only the frame the program may not pass can shrink it to nothing (a program
            # with audio always starts before its own end): it did not lose a window to
            # another program, its window had already closed when it was placed
            self._logger.debug(
                "PROGRAM-BUS schedule label=%s dropped: its window was closed before it",
                label,
            )
            self.last_refusal = ScheduleReason.TOO_LATE
            return None
        clamp = max(0, self.produced_frames - overlay.start_frame)
        if clamp and not overlay.trim_to(self.produced_frames):
            self._logger.debug(
                "PROGRAM-BUS schedule label=%s dropped: its whole window was already produced",
                label,
            )
            self.last_refusal = ScheduleReason.TOO_LATE
            return None
        self._overlays.append(overlay)
        self._logger.debug(
            "PROGRAM-BUS schedule label=%s anchor=%d clamp=%d",
            label,
            overlay.start_frame,
            clamp,
        )
        if clamp:
            self._logger.debug(
                "PROGRAM-BUS schedule label=%s started late by %.3fs",
                label,
                clamp / max(self._pcm_format.sample_rate, 1),
            )
        return BusPlacement(start_frame=overlay.start_frame, frames=overlay.voice_frames)

    def scheduled_voice_frames(self, label: str) -> int | None:
        """
        Return how much of the program with the given label is still to be heard.

        A program placed after the frame it was anchored on already went by starts with
        what is left of it, so this is shorter than the audio it was scheduled with.

        :param label: Label the program was scheduled with.
        """
        for overlay in reversed(self._overlays):
            if overlay.label == label:
                return overlay.voice_frames
        return None

    async def feed(self, chunk: bytes) -> bytes:
        """
        Mix the programs that are due into the chunk, keeping its frames and their order.

        A chunk whose end is not a whole frame can not be counted to its last byte: those
        bytes are prepended to the next chunk instead, so the stream the bus lets out stays
        frame aligned and no byte of it is lost or moved. The carry is reported, rate
        limited, so a stream that keeps ending inside a frame does not fill the log with it.

        A chunk is mixed in slices of about a tenth of a second, with the event loop handed
        back between them: a chunk that arrives in one piece (a crossfade, a long source
        read) must not stall the stream that is producing it. The chunk still leaves the bus
        as one piece, in whole frames: up to one frame short of the bytes it came in with,
        and one frame long where it carries the rest of the frame before it. The last chunk of
        a session has no successor to carry into, so :meth:`close` gives those bytes up.

        :param chunk: One chunk of music, in the flow's PCM format.
        """
        if self._residue:
            chunk = self._residue + chunk
            self._residue = b""
        first_frame = self.produced_frames
        frames = len(chunk) // self._frame_size if self._frame_size else 0
        if self._frame_size:
            usable = frames * self._frame_size
            if usable != len(chunk):
                self._residue = chunk[usable:]
                self._report_carry(len(self._residue))
                chunk = chunk[:usable]
        if self._overlays and any(
            overlay.covers(first_frame, frames) for overlay in self._overlays
        ):
            output = bytearray()
            for offset in range(0, len(chunk), self._slice_bytes):
                part = chunk[offset : offset + self._slice_bytes]
                position = self.produced_frames
                for overlay in self._overlays:
                    part = overlay.mix(part, position)
                output.extend(part)
                # the frames count as produced once they are mixed: a program placed while
                # the loop runs here must not be anchored in them
                self.produced_frames = position + len(part) // self._frame_size
                if offset + self._slice_bytes < len(chunk):
                    await asyncio.sleep(0)
            output_bytes = bytes(output)
        else:
            output_bytes = chunk
            self.produced_frames = first_frame + frames
        if self._overlays:
            overdue = [overlay for overlay in self._overlays if overlay.end_frame <= first_frame]
            if overdue:
                self._overlays = [
                    overlay for overlay in self._overlays if overlay.end_frame > first_frame
                ]
                if not self._overlays:
                    self._logger.debug("PROGRAM-BUS drained at frame %d", self.produced_frames)
        return output_bytes

    def close(self) -> None:
        """
        End the bus, giving up what a session's last chunk left of a frame.

        A session normally ends between two frames, but a source whose last chunk did not end
        on one leaves up to a frame in the carry and no chunk follows to complete it. Those
        bytes are dropped and reported, so the log explains a stream that is up to one frame
        shorter than what was fed into it.
        """
        if not self._residue:
            return
        self._logger.debug(
            "PROGRAM-BUS ends with %d bytes of a partial frame that no chunk completes: dropped",
            len(self._residue),
        )
        self._residue = b""

    def cancel_all(self, reason: str) -> None:
        """
        Drop every program that is still to be heard.

        :param reason: Why they are dropped, for the log.
        """
        if not self._overlays:
            return
        dropped, self._overlays = self._overlays, []
        self._logger.debug("PROGRAM-BUS cancel reason=%s", reason)
        self._report_dropped(dropped)

    def cancel(self, label: str, reason: str, *, start_frame: int | None = None) -> bool:
        """
        Drop the program with the given label, if it is still to be heard.

        :param label: Label the program was scheduled with.
        :param reason: Why it is dropped, for the log.
        :param start_frame: Anchor to match, or None for every program with the label.
        :return: Whether a program was still to be heard under that label.
        """
        dropped = [
            overlay
            for overlay in self._overlays
            if overlay.label == label
            and (start_frame is None or overlay.start_frame == start_frame)
        ]
        if not dropped:
            return False
        self._overlays = [overlay for overlay in self._overlays if overlay not in dropped]
        self._logger.debug("PROGRAM-BUS cancel reason=%s label=%s", reason, label)
        self._report_dropped(dropped)
        return True

    def scheduled_windows(self) -> list[ProgramWindow]:
        """Return what is scheduled on the bus, as the placement rules see it."""
        return [
            ProgramWindow(
                key=overlay.key,
                label=overlay.label,
                start_frame=overlay.start_frame,
                end_frame=overlay.end_frame,
                priority=overlay.priority,
            )
            for overlay in self._overlays
        ]

    def window_collisions(self, start_frame: int, frames: int) -> list[ProgramWindow]:
        """
        Return the programs whose window a placement at the given frames would overlap.

        :param start_frame: Frame the new program would start at.
        :param frames: Number of frames the new program lasts.
        """
        return window_collisions(self.scheduled_windows(), start_frame, frames)

    def drop_window(
        self, start_frame: int, frames: int, *, below: ProgramPriority
    ) -> list[tuple[int, str]]:
        """
        Drop the programs in a window that rank below the given priority.

        :param start_frame: Frame the program that claims the window starts at.
        :param frames: Number of frames the program claims.
        :param below: Only programs ranked below this priority give way.
        :return: The anchor and label of every program that was dropped.
        """
        dropped, kept = split_window(self.scheduled_windows(), start_frame, frames, below=below)
        if not dropped:
            return []
        kept_keys = {window.key for window in kept}
        dropped_keys = {window.key for window in dropped}
        dropped_overlays = [overlay for overlay in self._overlays if overlay.key in dropped_keys]
        self._overlays = [overlay for overlay in self._overlays if overlay.key in kept_keys]
        self._report_dropped(dropped_overlays)
        self._logger.debug(
            "PROGRAM-BUS drop window start=%d frames=%d labels=%s",
            start_frame,
            frames,
            ",".join(window.label for window in dropped),
        )
        return [(window.start_frame, window.label) for window in dropped]

    # --- Private methods ---

    def _report_dropped(self, overlays: list[_ProgramOverlay]) -> None:
        """
        Tell whoever holds a program's audio that it will not be heard after all.

        Only a program that had not started yet is given back: one whose first frame already
        went out has been heard by the listener, and handing its audio back would speak that
        part a second time.

        :param overlays: The programs the bus just dropped.
        """
        for overlay in overlays:
            if overlay.start_frame < self.produced_frames:
                continue
            if overlay.on_dropped is not None:
                overlay.on_dropped()

    def _report_carry(self, residue: int) -> None:
        """
        Report the bytes a chunk carried past its last whole frame, rate limited.

        :param residue: Bytes that are carried into the next chunk because they do not make
            up a frame on their own.
        """
        now = time.monotonic()
        if (
            self._carry_reported_at is not None
            and now - self._carry_reported_at < UNALIGNED_WARN_SECONDS
        ):
            self._logger.debug(
                "PROGRAM-BUS carried %d bytes of a partial frame into the next chunk", residue
            )
            return
        self._carry_reported_at = now
        self._logger.warning(
            "PROGRAM-BUS carried %d bytes over: the chunk does not end on a whole frame of "
            "%d bytes",
            residue,
            self._frame_size,
        )

    def _clear_of_scheduled(self, anchor: int, frames: int) -> int:
        """
        Return the anchor moved past every scheduled program its window would overlap.

        Which programs an anchor collides with is the schedule's rule, not this module's:
        the bus asks it rather than deciding the same window twice.

        :param anchor: Frame the program would start at.
        :param frames: Number of frames the program lasts.
        """
        moved = True
        while moved:
            moved = False
            for window in window_collisions(self.scheduled_windows(), anchor, frames):
                anchor = window.end_frame
                moved = True
        return anchor

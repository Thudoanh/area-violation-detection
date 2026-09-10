"""Sequential local-video input with explicit resource ownership."""

import logging
import math
from pathlib import Path
from typing import Union

import cv2


logger = logging.getLogger(__name__)


class VideoFrameProvider:
    """Yield (zero-based frame_index, timestamp_sec, BGR frame).

    Prefer media timestamps; fall back to index / FPS when unavailable.
    Use a with block to release the capture even when a loop exits early.
    Iteration is single-pass; EOF and read errors also close the capture.
    """

    def __init__(self, video_path: Union[str, Path]):
        self.video_path = Path(video_path)
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        self._capture = cv2.VideoCapture(str(self.video_path))
        self._closed = False
        self._frame_index = 0
        self._last_timestamp = None
        self._fallback_logged = False
        try:
            if not self._capture.isOpened():
                raise ValueError(f"Cannot open video: {self.video_path}")
            self.fps = self._capture.get(cv2.CAP_PROP_FPS)
            self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception:
            self.close()
            raise

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        try:
            ok, frame = self._capture.read()
            if not ok:
                self.close()
                raise StopIteration
            timestamp = self._capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if (not math.isfinite(timestamp) or timestamp < 0 or
                    (self._last_timestamp is not None and timestamp <= self._last_timestamp)):
                if not math.isfinite(self.fps) or self.fps <= 0:
                    raise ValueError(
                        f"Invalid media timestamp and FPS in video: {self.video_path}"
                    )
                timestamp = self._frame_index / self.fps
                if not self._fallback_logged:
                    logger.warning("Invalid media timestamp in %s; using frame index / FPS",
                                   self.video_path)
                    self._fallback_logged = True
                if self._last_timestamp is not None and timestamp <= self._last_timestamp:
                    raise ValueError(f"Cannot produce increasing timestamps: {self.video_path}")
            result = self._frame_index, timestamp, frame
            self._frame_index += 1
            self._last_timestamp = timestamp
            return result
        except Exception:
            self.close()
            raise

    def close(self):
        """Release the capture; safe to call more than once."""
        if not self._closed:
            self._capture.release()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

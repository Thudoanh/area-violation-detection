from unittest.mock import Mock

import cv2
import pytest

from src.core.frame_provider import VideoFrameProvider


def mock_capture(tmp_path, monkeypatch, timestamps, fps=25.0, opened=True):
    path = tmp_path / "input.mp4"
    path.touch()  # Capture is mocked; no synthetic video is generated.
    capture = Mock()
    capture.isOpened.return_value = opened
    times = iter(timestamps)
    metadata = {cv2.CAP_PROP_FPS: fps, cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480, cv2.CAP_PROP_FRAME_COUNT: len(timestamps)}
    capture.get.side_effect = lambda key: next(times) if key == cv2.CAP_PROP_POS_MSEC else metadata[key]
    capture.read.side_effect = [(True, f"frame{i}") for i in range(len(timestamps))] + [(False, None)]
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    return path, capture


def test_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        VideoFrameProvider(tmp_path / "missing.mp4")


def test_unopenable_video(tmp_path, monkeypatch):
    path, capture = mock_capture(tmp_path, monkeypatch, [], opened=False)
    with pytest.raises(ValueError, match="Cannot open video"):
        VideoFrameProvider(path)
    capture.release.assert_called_once()


def test_media_timestamps_and_eof_release(tmp_path, monkeypatch):
    path, capture = mock_capture(tmp_path, monkeypatch, [0, 50, 120])
    with VideoFrameProvider(path) as provider:
        assert (provider.fps, provider.width, provider.height, provider.total_frames) == (25, 640, 480, 3)
        assert list(provider) == [(0, 0.0, "frame0"), (1, 0.05, "frame1"), (2, 0.12, "frame2")]
    capture.release.assert_called_once()


def test_fps_fallback_logged(tmp_path, monkeypatch, caplog):
    path, capture = mock_capture(tmp_path, monkeypatch, [0, 0, 0])
    with VideoFrameProvider(path) as provider:
        assert [t for _, t, _ in provider] == [0.0, 0.04, 0.08]
    assert "using frame index / FPS" in caplog.text
    capture.release.assert_called_once()


def test_invalid_fps_with_valid_media_time(tmp_path, monkeypatch):
    path, _ = mock_capture(tmp_path, monkeypatch, [0, 60], fps=0)
    with VideoFrameProvider(path) as provider:
        assert [t for _, t, _ in provider] == [0.0, 0.06]


def test_invalid_timestamp_and_fps_release(tmp_path, monkeypatch):
    path, capture = mock_capture(tmp_path, monkeypatch, [float("nan")], fps=0)
    with pytest.raises(ValueError, match="Invalid media timestamp and FPS"):
        list(VideoFrameProvider(path))
    capture.release.assert_called_once()


def test_early_exit_release(tmp_path, monkeypatch):
    path, capture = mock_capture(tmp_path, monkeypatch, [0, 40])
    with VideoFrameProvider(path) as provider:
        next(provider)
    capture.release.assert_called_once()


def test_read_exception_release(tmp_path, monkeypatch):
    path, capture = mock_capture(tmp_path, monkeypatch, [0])
    capture.read.side_effect = RuntimeError("decode failure")
    with pytest.raises(RuntimeError, match="decode failure"):
        next(VideoFrameProvider(path))
    capture.release.assert_called_once()

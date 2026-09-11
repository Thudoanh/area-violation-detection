from unittest.mock import Mock

import numpy as np
import pytest

from scripts import run_video
from src.core.config import load_config
from src.core.models import Detection, TrackedObject
from src.core.visualization import draw_detections, draw_temporal, draw_tracks
from src.violation.inside_frame_counter import InsideFrameStatus
from src.violation.state_machine import State
from src.zones.zone_manager import ZoneManager


@pytest.fixture(autouse=True)
def isolate_event_storage(tmp_path, monkeypatch):
    original = load_config

    def isolated():
        config = original()
        config["storage"]["sqlite_path"] = str(tmp_path / "events.db")
        config["storage"]["snapshot_dir"] = str(tmp_path / "snapshots")
        config["detector"]["weights"] = str(tmp_path / "mock.pt")
        return config

    monkeypatch.setitem(globals(), "load_config", isolated)


def fake_video(monkeypatch, tmp_path):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    provider = Mock(fps=25.0, width=64, height=48, total_frames=2, video_path="input.mp4")
    context = Mock()
    context.__enter__ = Mock(return_value=provider)
    context.__exit__ = Mock(return_value=False)
    provider.__iter__ = Mock(return_value=iter([(0, 0.0, frame), (1, 0.04, frame)]))
    monkeypatch.setattr(run_video, "VideoFrameProvider", Mock(return_value=context))
    detector = Mock()
    detector.detect.return_value = [Detection((10, 20, 30, 40), 0.9, 0, "person")]
    monkeypatch.setattr(run_video, "YOLO11Detector", Mock(return_value=detector))
    tracker = Mock()
    tracker.update.side_effect = lambda detections, frame, timestamp: [
        TrackedObject(12, d.bbox, d.confidence, d.class_name, timestamp)
        for d in detections
    ]
    monkeypatch.setattr(run_video, "ByteTrackTracker", Mock(return_value=tracker))
    output = tmp_path / "out.mp4"
    writer = Mock()
    writer.isOpened.return_value = True
    writer.release.side_effect = lambda: output.write_bytes(b"mock output")
    monkeypatch.setattr(run_video.cv2, "VideoWriter", Mock(return_value=writer))
    return provider, context, detector, writer, output


def test_video_loop_annotation_and_summary(tmp_path, monkeypatch, capsys):
    _, context, _, writer, output = fake_video(monkeypatch, tmp_path)
    run_video.process_video("input.mp4", load_config(), output)
    assert writer.write.call_count == 2
    assert writer.write.call_args.args[0].any()
    writer.release.assert_called_once()
    context.__exit__.assert_called_once()
    summary = capsys.readouterr().out
    assert "Processed frames: 2" in summary
    assert "Average detections/frame: 1.000" in summary
    assert "person: 2" in summary
    assert "truck: 0" in summary
    assert "Unique track IDs: 1" in summary
    assert "Average active tracks/frame: 1.000" in summary
    assert "person tracks: 1" in summary
    assert "truck tracks: 0" in summary
    run_video.ByteTrackTracker.assert_called_once_with(load_config()["tracking"], frame_rate=25.0)
    calls = run_video.ByteTrackTracker.return_value.update.call_args_list
    assert [call.args[2] for call in calls] == [0.0, 0.04]


def test_inference_error_releases_writer_and_capture(tmp_path, monkeypatch):
    _, context, detector, writer, output = fake_video(monkeypatch, tmp_path)
    detector.detect.side_effect = [[], RuntimeError("inference failure")]
    with pytest.raises(RuntimeError, match="inference failure"):
        run_video.process_video("input.mp4", load_config(), output)
    writer.release.assert_called_once()
    context.__exit__.assert_called_once()


def test_writer_open_error(tmp_path, monkeypatch):
    _, context, _, writer, output = fake_video(monkeypatch, tmp_path)
    writer.isOpened.return_value = False
    with pytest.raises(RuntimeError, match="Cannot open annotated video writer"):
        run_video.process_video("input.mp4", load_config(), output)
    writer.release.assert_called_once()
    context.__exit__.assert_called_once()


def test_empty_video(tmp_path, monkeypatch):
    provider, _, _, writer, output = fake_video(monkeypatch, tmp_path)
    provider.__iter__.return_value = iter([])
    with pytest.raises(ValueError, match="no decodable frames"):
        run_video.process_video("input.mp4", load_config(), output)
    writer.write.assert_not_called()
    assert not output.exists()


def test_invalid_fps(tmp_path, monkeypatch):
    provider, _, _, writer, output = fake_video(monkeypatch, tmp_path)
    provider.fps = 0
    with pytest.raises(ValueError, match="positive video FPS"):
        run_video.process_video("input.mp4", load_config(), output)
    writer.write.assert_not_called()


def test_refuse_overwrite_input_or_existing_output(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"original")
    with pytest.raises(ValueError, match="differ from input"):
        run_video.process_video(video, load_config(), video)
    with pytest.raises(FileExistsError, match="already exists"):
        run_video.process_video("another.mp4", load_config(), video)
    assert video.read_bytes() == b"original"


def test_visualization_preserves_input_and_draws_label(monkeypatch):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    put_text = Mock(wraps=run_video.cv2.putText)
    monkeypatch.setattr(run_video.cv2, "putText", put_text)
    annotated = draw_detections(frame, [Detection((10, 30, 60, 80), 0.87, 1, "motorcycle")])
    assert not frame.any()
    assert annotated.any()
    assert put_text.call_args.args[1] == "motorcycle 0.87"


def test_tracking_visualization_draws_id_and_preserves_input(monkeypatch):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    put_text = Mock(wraps=run_video.cv2.putText)
    monkeypatch.setattr(run_video.cv2, "putText", put_text)
    track = TrackedObject(12, (10, 30, 60, 80), 0.87, "motorcycle", 1.23)
    annotated = draw_tracks(frame, [track])
    assert not frame.any()
    assert annotated.any()
    assert put_text.call_args.args[1] == "motorcycle ID:12"


def test_temporal_overlay_shows_violation_only_after_confirmation(monkeypatch):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    track = TrackedObject(12, (10, 30, 60, 80), 0.87, "motorcycle", 4.0)
    put_text = Mock(wraps=run_video.cv2.putText)
    monkeypatch.setattr(run_video.cv2, "putText", put_text)

    pending = draw_temporal(
        frame, [track], {12: (InsideFrameStatus("SW", 1.0, 29), State.INSIDE_PENDING)},
    )
    assert not pending.any()
    put_text.assert_not_called()

    annotated = draw_temporal(
        frame, [track],
        {12: (InsideFrameStatus("SW", 1.0, 30), State.SUSPECTED_VIOLATION)},
    )

    assert not frame.any()
    assert annotated.any()
    assert put_text.call_args.args[1] == "VIOLATION motorcycle ID:12"


def test_tracking_error_releases_video_resources(tmp_path, monkeypatch):
    _, context, _, writer, output = fake_video(monkeypatch, tmp_path)
    run_video.ByteTrackTracker.return_value.update.side_effect = [[], RuntimeError("tracking failure")]
    with pytest.raises(RuntimeError, match="tracking failure"):
        run_video.process_video("input.mp4", load_config(), output)
    writer.release.assert_called_once()
    context.__exit__.assert_called_once()


def test_empty_detections_still_update_tracker(tmp_path, monkeypatch):
    _, _, detector, _, output = fake_video(monkeypatch, tmp_path)
    detector.detect.return_value = []
    run_video.process_video("input.mp4", load_config(), output)
    calls = run_video.ByteTrackTracker.return_value.update.call_args_list
    assert len(calls) == 2
    assert all(call.args[0] == [] for call in calls)


def test_summary_counts_each_id_once_when_class_changes(tmp_path, monkeypatch, capsys):
    _, _, _, _, output = fake_video(monkeypatch, tmp_path)
    run_video.ByteTrackTracker.return_value.update.side_effect = [
        [TrackedObject(12, (10, 20, 30, 40), 0.9, "car", 0.0)],
        [TrackedObject(12, (10, 20, 30, 40), 0.9, "truck", 0.04)],
    ]
    run_video.process_video("input.mp4", load_config(), output)
    summary = capsys.readouterr().out
    assert "Unique track IDs: 1" in summary
    assert "car tracks: 1" in summary
    assert "truck tracks: 0" in summary


def test_zone_pipeline_overlaps_and_unique_summary(tmp_path, monkeypatch, capsys):
    from src.core.models import Zone
    from src.zones.zone_manager import ZoneManager
    _, _, _, writer, output = fake_video(monkeypatch, tmp_path)
    path = tmp_path / "zones.json"
    polygon = [(0, 0), (50, 0), (50, 45), (0, 45)]
    ZoneManager("CAM_001", [Zone("SW", "CAM_001", "SIDEWALK", polygon),
                            Zone("AL", "CAM_001", "ALLOWED", polygon)]).save_to_json(path)
    config = load_config()
    config["zones"]["path"] = str(path)
    put_text = Mock(wraps=run_video.cv2.putText)
    monkeypatch.setattr(run_video.cv2, "putText", put_text)
    run_video.process_video("input.mp4", config, output)
    summary = capsys.readouterr().out
    assert "Tracks entering SIDEWALK: 1" in summary
    assert "Tracks entering ALLOWED: 1" in summary
    assert "Tracks entering IGNORE: 0" in summary
    labels = [call.args[1] for call in put_text.call_args_list]
    assert "SW SIDEWALK" in labels
    assert "person ID:12" in labels
    assert not any("SIDEWALK+ALLOWED" in label for label in labels)
    assert writer.write.call_count == 2


def test_zone_camera_mismatch_rejected(tmp_path, monkeypatch):
    from src.zones.zone_manager import ZoneManager
    _, _, _, writer, output = fake_video(monkeypatch, tmp_path)
    path = tmp_path / "zones.json"
    ZoneManager("OTHER", []).save_to_json(path)
    config = load_config()
    config["zones"]["path"] = str(path)
    with pytest.raises(ValueError, match="camera_id does not match"):
        run_video.process_video("input.mp4", config, output)
    writer.write.assert_not_called()


def test_video_loop_with_real_temporal_storage(tmp_path, monkeypatch, capsys):
    import sqlite3
    from pathlib import Path
    from src.core.models import Zone
    from src.zones.zone_manager import ZoneManager
    provider, _, detector, writer, output = fake_video(monkeypatch, tmp_path)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    provider.__iter__.return_value = iter((i, float(i), frame) for i in range(35))
    detector.detect.return_value = [Detection((10, 20, 30, 40), 0.9, 2, "car")]
    zone_file = tmp_path / "zone.json"
    ZoneManager("CAM_001", [Zone("SW", "CAM_001", "SIDEWALK",
        [(0, 0), (60, 0), (60, 45), (0, 45)])]).save_to_json(zone_file)
    config = load_config()
    config["zones"]["path"] = str(zone_file)
    run_video.process_video("input.mp4", config, output)
    assert writer.write.call_count == 35
    with sqlite3.connect(config["storage"]["sqlite_path"]) as connection:
        status, left_at, path = connection.execute("SELECT status,left_at,snapshot_path FROM events").fetchone()
    assert status == "CLOSED" and left_at is None
    assert Path(path).is_file()
    decoded = run_video.cv2.imdecode(np.fromfile(path, np.uint8), run_video.cv2.IMREAD_COLOR)
    assert decoded.any()  # Full frame contains annotations.
    summary = capsys.readouterr().out
    assert "Accepted events: 1" in summary
    assert "Snapshots saved: 1" in summary

import numpy as np
import pytest

import app


def sample_event():
    return {
        "event_id": "event1", "run_id": "run1", "video_id": "video.mp4",
        "camera_id": "WEB_DEMO", "zone_id": "WEB_ROI", "zone_type": "SIDEWALK",
        "track_id": 7, "object_class": "car", "entered_at": 10.0,
        "stationary_since": 12.0, "violation_at": 75.4, "left_at": None,
        "dwell_time_sec": 30.0, "confidence": 0.8764,
        "snapshot_path": "missing.jpg", "status": "CLOSED",
        "config_version": "cfg", "model_version": "model",
        "event_type": "SUSPECTED_AREA_OCCUPATION",
    }


def test_event_rows_are_mapped_to_vietnamese_columns():
    frame = app.events_to_dataframe([sample_event()])
    assert frame.columns.tolist() == app.EVENT_COLUMNS
    assert frame.iloc[0]["Thời điểm"] == "00:01:15.40"
    assert frame.iloc[0]["Phương tiện"] == "car"
    assert frame.iloc[0]["Độ tin cậy"] == 0.876


def test_empty_event_outputs_are_valid():
    assert app.events_to_dataframe([]).empty
    figure = app.build_charts([])
    assert "Không phát hiện" in figure.layout.annotations[0].text


def test_json_polygon_validation_and_bounds():
    image = np.zeros((100, 200, 3), np.uint8)
    assert app._parse_points("[[1,2],[30,4],[5,60]]", image) == [
        [1, 2], [30, 4], [5, 60]
    ]
    with pytest.raises(ValueError, match="dạng"):
        app._parse_points('{"x": 1}', image)
    with pytest.raises(ValueError, match="ngoài"):
        app._parse_points("[[1,2],[300,4],[5,60]]", image)


def test_polygon_preview_does_not_mutate_original():
    image = np.zeros((100, 200, 3), np.uint8)
    preview = app._draw_polygon(image, [[10, 10], [100, 10], [50, 80]])
    assert not image.any()
    assert preview.any()


def test_upload_returns_h264_preview_instead_of_original(monkeypatch, tmp_path):
    original = tmp_path / "camera.mov"
    playable = tmp_path / "camera_web.mp4"
    frame = np.zeros((100, 200, 3), np.uint8)
    monkeypatch.setattr(app, "_first_frame", lambda _: (frame, "200×100"))
    monkeypatch.setattr(
        app, "_ensure_browser_video",
        lambda _: (playable, "Video kết quả đã được chuyển sang H.264."),
    )
    result = app.on_upload(str(original))
    assert result[0] == str(playable)
    assert result[2] is frame
    assert "H.264" in result[5]


def test_gradio_app_builds_with_single_run_queue():
    demo = app.build_app()
    assert demo is not None

"""Web demo tiếng Việt cho hệ thống phát hiện chiếm dụng khu vực."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
from uuid import uuid4

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.config import load_config
from src.core.models import Zone
from src.pipeline.video_runner import VideoRunResult, process_video
from src.zones.auto_roi import AutoROISettings, ClassicalAutoROIDetector
from src.zones.zone_manager import ZoneManager, validate_zone

PROJECT_ROOT = Path(__file__).resolve().parent
WEB_OUTPUT_ROOT = PROJECT_ROOT / "data" / "events" / "web"

CSS = """
.main-title { text-align: center; margin-bottom: 1rem; }
.main-title p { color: #64748b; }
.roi-note { color: #854d0e; background: #fef9c3; padding: 10px; border-radius: 8px; }
footer { display: none !important; }
"""

EVENT_COLUMNS = [
    "Thời điểm", "Phương tiện", "Track ID", "Khu vực", "Loại vùng",
    "Thời gian dừng (giây)", "Độ tin cậy", "Trạng thái", "Event ID",
]


def _first_frame(video_path):
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError("Không thể mở video")
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Video không có khung hình đọc được")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        duration = total / fps if fps > 0 else 0.0
        info = (
            f"{width}×{height} | {total} khung hình | "
            f"{fps:.2f} FPS | {duration:.1f} giây"
        )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), info
    finally:
        capture.release()


def _draw_polygon(frame_rgb, points, source="Thủ công"):
    if frame_rgb is None:
        return None
    image = np.asarray(frame_rgb).copy()
    points = [[int(x), int(y)] for x, y in (points or [])]
    if not points:
        height, width = image.shape[:2]
        cv2.putText(
            image, "Nhấp chuot de them cac dinh ROI",
            (max(10, width // 2 - 185), max(30, height // 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
        )
        return image
    polygon = np.asarray(points, dtype=np.int32)
    if len(points) >= 3:
        overlay = image.copy()
        cv2.fillPoly(overlay, [polygon], (239, 68, 68))
        image = cv2.addWeighted(overlay, 0.28, image, 0.72, 0)
        cv2.polylines(image, [polygon], True, (220, 38, 38), 3, cv2.LINE_AA)
    elif len(points) == 2:
        cv2.line(image, tuple(points[0]), tuple(points[1]), (220, 38, 38), 2)
    for index, (x, y) in enumerate(points, start=1):
        cv2.circle(image, (x, y), 7, (34, 197, 94), -1)
        cv2.circle(image, (x, y), 7, (255, 255, 255), 2)
        cv2.putText(image, str(index), (x + 9, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(image, f"ROI: {len(points)} diem | {source}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (34, 197, 94), 2, cv2.LINE_AA)
    return image


def on_upload(video_path):
    if not video_path:
        return None, None, None, [], "[]", "", "Chưa tải video."
    try:
        frame, info = _first_frame(video_path)
        playable, codec_note = _ensure_browser_video(video_path)
    except (OSError, ValueError, cv2.error) as exc:
        raise gr.Error(str(exc)) from exc
    return (
        str(playable), _draw_polygon(frame, []), frame, [], "[]",
        f"{info} | {codec_note}", "Chọn hoặc đề xuất ROI.",
    )


def on_click(original_frame, points, evt: gr.SelectData):
    if original_frame is None:
        raise gr.Error("Hãy tải video trước.")
    x, y = (int(evt.index[0]), int(evt.index[1]))
    height, width = np.asarray(original_frame).shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise gr.Error("Điểm được chọn nằm ngoài khung hình.")
    updated = [list(point) for point in (points or [])]
    updated.append([x, y])
    return _draw_polygon(original_frame, updated), updated, json.dumps(updated), "ROI thủ công."


def _parse_points(coords_text, original_frame=None):
    try:
        raw = json.loads(coords_text)
        if not isinstance(raw, list):
            raise ValueError
        points = []
        for point in raw:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError
            x, y = point
            if isinstance(x, bool) or isinstance(y, bool):
                raise ValueError
            points.append([int(x), int(y)])
    except (json.JSONDecodeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Tọa độ phải có dạng [[x1,y1],[x2,y2],...].") from exc
    if original_frame is not None:
        height, width = np.asarray(original_frame).shape[:2]
        if any(x < 0 or y < 0 or x >= width or y >= height for x, y in points):
            raise ValueError("Tọa độ ROI nằm ngoài khung hình.")
    return points


def on_coords_submit(original_frame, coords_text):
    if original_frame is None:
        raise gr.Error("Hãy tải video trước.")
    try:
        points = _parse_points(coords_text, original_frame)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    return _draw_polygon(original_frame, points), points, "ROI từ tọa độ JSON."


def on_undo(original_frame, points):
    if original_frame is None:
        return None, [], "[]", "Chưa tải video."
    updated = [list(point) for point in (points or [])]
    if updated:
        updated.pop()
    return (_draw_polygon(original_frame, updated), updated,
            json.dumps(updated), "Đã hoàn tác điểm cuối.")


def on_clear(original_frame):
    if original_frame is None:
        return None, [], "[]", "Chưa tải video."
    return _draw_polygon(original_frame, []), [], "[]", "Đã xóa ROI."


def on_auto_roi(video_path, original_frame, num_frames, vote_ratio):
    if not video_path or original_frame is None:
        raise gr.Error("Hãy tải video trước.")
    try:
        auto_config = load_config().get("zones", {}).get("auto_detect", {})
        settings = AutoROISettings(
            num_frames=int(num_frames), vote_ratio=float(vote_ratio),
            min_area_ratio=float(auto_config.get("min_area_ratio", 0.005)),
            simplify_epsilon_ratio=float(
                auto_config.get("simplify_epsilon_ratio", 0.015)
            ),
        )
        _, polygon = ClassicalAutoROIDetector(settings).detect_from_video(video_path)
    except (OSError, ValueError, cv2.error) as exc:
        raise gr.Error(f"Không thể đề xuất ROI: {exc}") from exc
    if not polygon:
        raise gr.Error(
            "Không tìm thấy vùng kẻ/sơn đủ ổn định. Hãy giảm tỷ lệ bỏ phiếu hoặc vẽ ROI thủ công."
        )
    points = [[x, y] for x, y in polygon]
    return (_draw_polygon(original_frame, points, "Tự động đề xuất"), points,
            json.dumps(points), "Đã đề xuất ROI. Hãy kiểm tra trước khi chạy phát hiện.")


def _format_video_time(seconds):
    value = float(seconds)
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    remainder = value % 60
    return f"{hours:02d}:{minutes:02d}:{remainder:05.2f}"


def events_to_dataframe(events):
    rows = []
    for event in events:
        rows.append({
            "Thời điểm": _format_video_time(event["violation_at"]),
            "Phương tiện": event["object_class"],
            "Track ID": event["track_id"],
            "Khu vực": event["zone_id"],
            "Loại vùng": event["zone_type"],
            "Thời gian dừng (giây)": round(float(event["dwell_time_sec"]), 2),
            "Độ tin cậy": round(float(event["confidence"]), 3),
            "Trạng thái": event["status"],
            "Event ID": event["event_id"],
        })
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def build_charts(events):
    if not events:
        figure = go.Figure()
        figure.add_annotation(text="Không phát hiện sự kiện vi phạm", showarrow=False,
                              font=dict(size=18, color="#64748b"))
        figure.update_layout(height=430, margin=dict(l=20, r=20, t=30, b=20))
        return figure
    frame = pd.DataFrame(events)
    vehicle_counts = frame["object_class"].value_counts()
    figure = make_subplots(rows=1, cols=2,
                           subplot_titles=("Sự kiện theo loại phương tiện", "Dòng thời gian"))
    figure.add_trace(go.Bar(x=vehicle_counts.index, y=vehicle_counts.values,
                            marker_color="#2563eb", name="Sự kiện"), row=1, col=1)
    figure.add_trace(go.Scatter(
        x=frame["violation_at"], y=frame["dwell_time_sec"], mode="markers+lines",
        text=[f"{row.object_class} | Track {row.track_id}" for row in frame.itertuples()],
        hovertemplate="%{text}<br>Giây: %{x:.2f}<br>Dwell: %{y:.2f}<extra></extra>",
        marker=dict(size=10, color="#dc2626"), name="Thời điểm",
    ), row=1, col=2)
    figure.update_xaxes(title_text="Phương tiện", row=1, col=1)
    figure.update_yaxes(title_text="Số sự kiện", row=1, col=1)
    figure.update_xaxes(title_text="Thời gian video (giây)", row=1, col=2)
    figure.update_yaxes(title_text="Dwell (giây)", row=1, col=2)
    figure.update_layout(height=460, showlegend=False,
                         margin=dict(l=35, r=25, t=55, b=35))
    return figure


def build_summary(result: VideoRunResult, roi_source):
    return "\n".join([
        "| Chỉ số | Giá trị |",
        "|---|---:|",
        f"| Sự kiện được chấp nhận | **{result.accepted}** |",
        f"| Candidate vi phạm | {result.candidates} |",
        f"| Trùng lặp bị chặn | {result.suppressed} |",
        f"| Track đứng yên | {result.stationary_tracks} |",
        f"| Track duy nhất | {result.unique_tracks} |",
        f"| Khung hình đã xử lý | {result.processed_frames} |",
        f"| Tốc độ xử lý | {result.processing_fps:.2f} FPS |",
        f"| Nguồn ROI | {roi_source} |",
        "",
        f"**Run ID:** `{result.run_id}`  ",
        f"**SQLite:** `{result.database_path}`  ",
        f"**Video kết quả:** `{result.output_path}`",
    ])


def _ensure_browser_video(source_path):
    source = Path(source_path)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError, OSError):
            return source, "Gradio sẽ tự thử chuyển video sang định dạng phát được trên trình duyệt."
    target = source.with_name(f"{source.stem}_web.mp4")
    try:
        subprocess.run([
            ffmpeg, "-y", "-i", str(source), "-c:v", "libx264", "-preset", "fast",
            "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(target),
        ], check=True, capture_output=True, timeout=600)
        return target, "Video kết quả đã được chuyển sang H.264."
    except (OSError, subprocess.SubprocessError):
        return source, "Không chuyển được H.264; Gradio sẽ thử xử lý video gốc."


def run_detection(video_path, points, zone_type, confidence, nms, dwell_time,
                  roi_status, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Hãy tải video trước.")
    try:
        points = _parse_points(json.dumps(points or []))
        zone = Zone("WEB_ROI", "WEB_DEMO", zone_type,
                    [(int(x), int(y)) for x, y in points])
        validate_zone(zone)
    except ValueError as exc:
        raise gr.Error(f"ROI không hợp lệ: {exc}") from exc

    config = deepcopy(load_config())
    config["camera"]["camera_id"] = "WEB_DEMO"
    config["detector"]["confidence_threshold"] = float(confidence)
    config["detector"]["nms_threshold"] = float(nms)
    config["violation"]["min_dwell_time_sec"] = float(dwell_time)
    run_id = uuid4().hex
    run_directory = WEB_OUTPUT_ROOT / run_id
    output_path = run_directory / f"annotated_{Path(video_path).stem}.mp4"
    manager = ZoneManager("WEB_DEMO", [zone])

    def report(done, total, message):
        ratio = done / total if total and total > 0 else 0
        progress(min(ratio, 0.94), desc=message)

    try:
        result = process_video(
            video_path, config, output_path, zone_manager=manager,
            run_id=run_id, progress_callback=report,
        )
        progress(0.96, desc="Đang chuẩn bị video xem trước...")
        playable, codec_note = _ensure_browser_video(result.output_path)
    except (OSError, ValueError, RuntimeError, sqlite3.Error, cv2.error) as exc:
        raise gr.Error(f"Xử lý video thất bại: {exc}") from exc

    dataframe = events_to_dataframe(result.events)
    chart = build_charts(result.events)
    summary = build_summary(result, roi_status or "Thủ công") + f"\n\n{codec_note}"
    gallery = [
        (event["snapshot_path"],
         f"{_format_video_time(event['violation_at'])} | "
         f"{event['object_class']} | Track {event['track_id']}")
        for event in result.events if Path(event["snapshot_path"]).is_file()
    ]
    progress(1.0, desc="Hoàn tất.")
    return str(playable), dataframe, chart, summary, gallery or None


def build_app():
    defaults = load_config()
    detector_defaults = defaults.get("detector", {})
    violation_defaults = defaults.get("violation", {})
    auto_defaults = defaults.get("zones", {}).get("auto_detect", {})
    with gr.Blocks(title="Phát hiện chiếm dụng khu vực") as app:
        gr.HTML(
            "<div class='main-title'><h1>Phát hiện chiếm dụng khu vực</h1>"
            "<p>Tải video → xác định ROI → chạy phát hiện → xem sự kiện và bằng chứng</p></div>"
        )
        original_frame = gr.State(None)
        points_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Tải video")
                video_input = gr.File(
                    label="Chọn tệp video", file_types=["video"], type="filepath"
                )
                video_preview = gr.Video(
                    label="Xem trước video đầu vào", interactive=False, format="mp4"
                )
                video_info = gr.Textbox(label="Thông tin video", interactive=False)

                gr.Markdown("### 2. Thiết lập phát hiện")
                confidence = gr.Slider(0.1, 0.9,
                                       value=detector_defaults.get("confidence_threshold", 0.4), step=0.05,
                                       label="Ngưỡng tin cậy")
                nms = gr.Slider(0.1, 0.9,
                                value=detector_defaults.get("nms_threshold", 0.45), step=0.05,
                                label="Ngưỡng IoU/NMS")
                dwell = gr.Slider(1, 120,
                                  value=violation_defaults.get("min_dwell_time_sec", 30), step=1,
                                  label="Thời gian đứng yên tối thiểu (giây)")
                zone_type = gr.Dropdown(["SIDEWALK", "MONITORED"], value="SIDEWALK",
                                        label="Loại khu vực")

                gr.Markdown("### 3. Xác định ROI")
                gr.HTML("<div class='roi-note'>ROI tự động là đề xuất cho vùng kẻ/sơn có độ tương phản rõ. Hãy kiểm tra polygon trước khi chạy.</div>")
                coordinates = gr.Textbox(label="Tọa độ polygon (JSON)", value="[]")
                roi_status = gr.Textbox(label="Trạng thái ROI", value="Chưa tải video.",
                                         interactive=False)
                with gr.Accordion("Thiết lập ROI tự động", open=False):
                    auto_frames = gr.Slider(3, 30,
                                            value=auto_defaults.get("num_frames", 15), step=1,
                                            label="Số khung hình lấy mẫu")
                    vote_ratio = gr.Slider(0.3, 0.9,
                                           value=auto_defaults.get("vote_ratio", 0.6), step=0.05,
                                           label="Tỷ lệ bỏ phiếu ổn định")
                    auto_button = gr.Button("Tự động đề xuất ROI")
                with gr.Row():
                    undo_button = gr.Button("Hoàn tác")
                    clear_button = gr.Button("Xóa ROI", variant="stop")
                run_button = gr.Button("Chạy phát hiện", variant="primary", size="lg")

            with gr.Column(scale=2):
                roi_preview = gr.Image(label="Xem trước và chọn ROI", interactive=False,
                                       type="numpy", height=560)

        gr.Markdown("---")
        with gr.Tabs():
            with gr.Tab("Video kết quả"):
                video_output = gr.Video(label="Video đã chạy detection", format="mp4")
            with gr.Tab("Tổng quan"):
                summary_output = gr.Markdown()
                chart_output = gr.Plot(label="Thống kê sự kiện")
            with gr.Tab("Sự kiện"):
                table_output = gr.Dataframe(headers=EVENT_COLUMNS, interactive=False,
                                            label="Danh sách sự kiện")
            with gr.Tab("Ảnh bằng chứng"):
                gallery_output = gr.Gallery(label="Snapshot vi phạm", columns=4,
                                             object_fit="contain", height=320)

        video_input.change(
            on_upload, [video_input],
            [video_preview, roi_preview, original_frame, points_state,
             coordinates, video_info, roi_status],
        )
        roi_preview.select(
            on_click, [original_frame, points_state],
            [roi_preview, points_state, coordinates, roi_status],
        )
        coordinates.submit(
            on_coords_submit, [original_frame, coordinates],
            [roi_preview, points_state, roi_status],
        )
        undo_button.click(
            on_undo, [original_frame, points_state],
            [roi_preview, points_state, coordinates, roi_status],
        )
        clear_button.click(
            on_clear, [original_frame],
            [roi_preview, points_state, coordinates, roi_status],
        )
        auto_button.click(
            on_auto_roi, [video_input, original_frame, auto_frames, vote_ratio],
            [roi_preview, points_state, coordinates, roi_status],
        )
        run_button.click(
            run_detection,
            [video_input, points_state, zone_type, confidence, nms, dwell, roi_status],
            [video_output, table_output, chart_output, summary_output, gallery_output],
            concurrency_limit=1,
        )
    return app.queue(default_concurrency_limit=1, max_size=8)


if __name__ == "__main__":
    build_app().launch(
        server_name="127.0.0.1", share=False, css=CSS
    )

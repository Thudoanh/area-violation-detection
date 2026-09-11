# Solution Design

> Technical architecture source of truth — **Architecture Freeze v0.2**. Phạm vi/hành vi theo [Requirements](requirements.md). README chỉ là bản tóm tắt.

## 1. Design Principles

Thiết kế theo các nguyên tắc:

1. modular;
2. detector-agnostic;
3. rule-driven;
4. temporal-aware;
5. event-level, không chỉ frame-level;
6. clean-room implementation;
7. dễ thay đổi rule khi nghiệp vụ thay đổi.

---

## 2. High-Level Architecture

```text
Video
 ↓
Detector
 ↓
Tracker
 ↓
Zone Manager
 ↓
Inside Frame Counter
 ↓
State Machine
 ↓
Rule Engine
 ↓
Deduplicator
 ↓
Event Storage
```

Frame Provider đọc video file từ camera cố định và cung cấp timestamp theo giây.
Inside Frame Counter đếm quan sát liên tiếp theo track. `person` chỉ đi qua
detection/tracking và context; nhánh violation chỉ xử lý `VEHICLE_CLASSES`.

---

## 3. Repository Structure

```text
src/
├── core/
│   ├── config.py
│   ├── models.py
│   └── frame_provider.py
├── detection/
│   ├── base.py
│   └── <backend>_detector.py
├── tracking/
│   ├── base.py
│   └── bytetrack_tracker.py
├── zones/
│   ├── models.py
│   ├── zone_manager.py
│   ├── geometry.py
│   └── auto_roi.py
├── violation/
│   ├── inside_frame_counter.py
│   ├── state_machine.py
│   ├── rule_engine.py
│   └── deduplicator.py
├── storage/
│   ├── models.py
│   ├── event_repository.py
│   └── evidence_writer.py
└── pipeline/
    ├── pipeline.py
    └── video_runner.py
```

`app.py` là web demo Gradio tiếng Việt. CLI và web cùng gọi `video_runner.py`; mỗi
lượt chạy có `run_id` riêng. Web đọc lại event qua `EventRepository.list_by_run`
để tạo bảng, biểu đồ và gallery, còn SQLite vẫn là nguồn dữ liệu canonical.

---

Tên generic: `ZoneManager`, `ViolationEngine` (Rule Engine), `ViolationEvent`, `AreaMonitoringPipeline`. Cây trên là cấu trúc mục tiêu, chưa phải code đã có.

## 4. Core Data Models

```text
DETECTION_CLASSES = {person, motorcycle, bicycle, car, bus, truck}
VEHICLE_CLASSES = VIOLATION_TARGET_CLASSES = {motorcycle, bicycle, car, bus, truck}
```

`person` chỉ cung cấp context, không đi vào violation evaluation. Bbox dùng pixel xyxy; class ID được adapter chuẩn hóa, không dùng ID riêng của backend trong rule. Timestamp là giây từ đầu video; track ID được scope theo run/camera.

### Detection

```python
Detection(
    bbox=(x1, y1, x2, y2),
    confidence=float,
    class_id=int,
    class_name=str
)
```

### Track

```python
TrackedObject(
    track_id=int,
    bbox=(x1, y1, x2, y2),
    confidence=float,
    class_name=str,
    timestamp=float
)
```

### Zone

```python
Zone(
    zone_id=str,
    camera_id=str,
    zone_type=str,
    polygon=list[tuple[int, int]]
)
```

### Event

`ViolationEvent` dùng đúng schema bảng `events` tại mục Storage; `event_type` luôn là `SUSPECTED_AREA_OCCUPATION`. Không định nghĩa schema thứ hai ở model layer.

---

## 5. Detector Layer

Detector phải implement interface:

```python
class BaseDetector:
    def detect(self, frame) -> list[Detection]:
        raise NotImplementedError
```

Phase 2 ưu tiên pretrained backend:

- YOLOX;
- RTMDet/MMDetection;

Backend khác là thay đổi thiết kế cần cập nhật docs; không ràng buộc rule vào backend.

Pipeline chỉ nhận `list[Detection]`.

---

## 6. Tracking Layer

Tracker MVP: **ByteTrack**, đặt sau adapter:

```python
class BaseTracker:
    def update(self, detections, frame, timestamp: float) -> list[TrackedObject]:
        raise NotImplementedError
```

Yêu cầu:

- persistent ID;
- giữ class;
- giữ confidence;
- hỗ trợ lost track ngắn hạn.

---

## 7. Zone Manager

Zone manager chịu trách nhiệm:

- load/save polygon;
- đề xuất polygon vùng kẻ/sơn bằng temporal voting trên nhiều khung hình trong web demo;
- point-in-polygon;
- xác định zone theo camera;
- precedence bắt buộc: `IGNORE > ALLOWED > SIDEWALK / MONITORED`; `SIDEWALK + ALLOWED` trả NO VIOLATION. Membership trên biên được tính inside. Nếu nhiều target zone cùng chứa anchor, chọn zone đầu tiên theo thứ tự cấu hình để một track chỉ có một target zone hiệu lực; lưu thứ tự đó trong config version.

Anchor MVP:

```text
bottom_center = ((x1+x2)/2, y2)
```

Future:

- bbox overlap;
- segmentation mask overlap.

---

## 8. Shapely Zone Membership

Mỗi polygon được tạo và kiểm tra tính hợp lệ bằng Shapely. Membership dùng
`Polygon.covers(Point(bottom_center))`, vì vậy điểm nằm trên biên cũng được tính
là inside. Polygon Shapely được cache theo zone, không dựng lại ở mỗi frame.

---

## 9. Consecutive Inside-Frame Validation

Các timestamp cần lưu:

```text
entered_at
left_at
```

Mỗi quan sát hợp lệ của cùng `track_id` trong cùng target zone tăng bộ đếm:

```text
inside_frame_count += 1
```

Ra target zone, vào ALLOWED/IGNORE, đổi target zone hoặc mất track sẽ reset bộ
đếm. Candidate được tạo khi `inside_frame_count >= min_inside_frames`. Timestamp
video vẫn được lưu phục vụ audit, nhưng không quyết định threshold violation.

---

## 10. State Machine

```text
OUTSIDE
  │ enters
  ↓
ENTERING
  │ stable inside
  ↓
INSIDE_PENDING
  │ inside_frame_count >= min_inside_frames
  ↓
SUSPECTED_VIOLATION
  │ Rule Engine accepts + dedup allows + storage succeeds
  ↓
ALERTED
```

`SUSPECTED_VIOLATION` là trạng thái candidate nội bộ, không phải tên event. Rule Engine quyết định rule cuối cùng; chỉ chuyển `ALERTED` sau khi lưu event/snapshot thành công. Candidate bị dedup không tạo event mới.

`ENTERING` bắt đầu ở frame hợp lệ đầu tiên. `INSIDE_PENDING` giữ trạng thái cho
đến khi đủ N frame. `ALERTED` giữ lock đến khi episode đóng. Sau CLOSED, lần vào
lại bắt đầu episode mới và vẫn qua cooldown.

Exit transition:

```text
any active state
  │ outside / excluded / lost >= exit_grace_sec
  ↓
CLOSED
```

---

## 11. Rule Engine

Rule canonical (Rule Engine riêng):

```text
object.class ∈ VEHICLE_CLASSES
AND object is inside SIDEWALK or MONITORED zone
AND object is NOT inside ALLOWED zone
AND object is NOT inside IGNORE zone
AND consecutive_inside_frames >= min_inside_frames
→ SUSPECTED_AREA_OCCUPATION
```

---

## 12. Deduplication

Ba lớp:

### Per-track lock

Một track chỉ tạo một active event.

### Cooldown

Sau khi event đóng, không tạo event mới trong một khoảng thời gian nếu object vẫn gần vị trí cũ.

### Spatial dedup

Nếu tracker đổi ID nhưng violation mới xảy ra gần một recent event:

```text
distance(new_anchor, old_anchor) < threshold
AND same_camera_and_run
AND same_zone
AND same_object_class
AND (event_is_active OR seconds_since_closed < cooldown_sec)
```

thì suppress. So sánh với cả active event và event vừa đóng; active event không hết bảo vệ chỉ vì đã qua cooldown. Cooldown tính từ lúc đóng event. Không gộp hai track đồng thời được quan sát là hai phương tiện khác nhau chỉ vì ở gần nhau; spatial heuristic cần đánh giá trên scenario nhiều xe và ID switch.

---

## 13. Storage

MVP dùng **SQLite cho event metadata + filesystem cho snapshot**. JSON/JSONL chỉ là định dạng export/evaluation, không phải storage backend thay thế.

Bảng `events`:

```text
event_id
event_type
run_id
video_id
camera_id
zone_id
zone_type
track_id
object_class
entered_at
stationary_since
violation_at
left_at
dwell_time_sec
inside_frame_count
confidence
snapshot_path
status
config_version
model_version
```

---

`inside_frame_count` là bằng chứng chính cho threshold N-frame. Hai cột
`stationary_since` và `dwell_time_sec` được giữ trong SQLite để tương thích schema
cũ; chúng lần lượt chứa thời điểm bắt đầu chuỗi inside và khoảng thời gian media
từ lúc vào ROI đến lúc tạo event, không tham gia quyết định violation.
`confidence` là detection confidence, không phải xác suất vi phạm. Snapshot phải
được ghi thành công trước khi commit event. Lỗi storage không đánh dấu ALERTED.

## 14. Pipeline Orchestration

Pseudo-flow:

```python
for frame, timestamp in frame_provider:

    detections = detector.detect(frame)

    tracks = tracker.update(detections, frame, timestamp)

    for track in tracks:
        if track.class_name not in VEHICLE_CLASSES:
            continue  # person: context only
        zone_context = zone_manager.evaluate(track)
        inside = inside_frame_counter.update(track, zone_context, timestamp)
        state = state_machine.update(track, zone_context, inside.consecutive_frames, timestamp)

        candidate = rule_engine.evaluate(
            track=track,
            zone_context=zone_context,
            state=state,
            inside_frame_count=inside.consecutive_frames,
            timestamp=timestamp,
        )

        if candidate and deduplicator.allow(candidate):
            candidate.snapshot_path = evidence_writer.save(frame, candidate)
            event_repository.create(candidate)
            deduplicator.register(candidate)
            state_machine.mark_alerted(track, candidate)
```

---

Pseudo-flow trên chỉ minh họa nhánh phát event; pipeline còn phải cập nhật lost tracks, đóng event/lock sau grace, xử lý EOF và lỗi lưu trữ theo lifecycle đã nêu.

## 15. Config Example

Giá trị khởi tạo để thử nghiệm, không phải acceptance thresholds. Backend/model là ví dụ Phase 2, chưa phải kết quả chọn baseline.

```yaml
detector:
  backend: yolox
  model: yolox_s
  confidence_threshold: 0.4
  detection_classes: [person, motorcycle, bicycle, car, bus, truck]

tracking:
  backend: bytetrack

zones:
  path: configs/zones/cam01.json

violation:
  target_classes: [motorcycle, bicycle, car, bus, truck]
  min_inside_frames: 30
  exit_grace_sec: 3

dedup:
  cooldown_sec: 60
  spatial_distance_px: 50

storage:
  sqlite_path: data/events/events.db
  snapshot_dir: data/events/snapshots
```

---

## 16. Reference Architecture Adaptation

Từ repository tham khảo, project này giữ các pattern:

- module separation;
- detector → tracker → polygon zone → temporal filter → event;
- state machine;
- per-track processing;
- dedup/cooldown;
- evidence logging.

Các phần không dùng:

- source implementation của repo tham khảo;
- Ultralytics-specific wrapper;
- plate OCR;
- Turkish plate validation;
- hatched-area severity categories;
- automatic hatched-area detector.

Các phần viết mới:

- detector abstraction;
- sidewalk/monitored/allowed/ignore semantics;
- Shapely point-in-polygon;
- consecutive inside-frame validation;
- violation rules;
- event schema;
- evaluation data.

## 17. Architecture Freeze v0.2

| Decision | Chốt |
|---|---|
| Input MVP | Video file |
| Camera | Fixed-view |
| Detector architecture | Replaceable, detector-agnostic |
| Detector baseline | YOLO11n pretrained COCO (`weights/yolo11n.pt`) |
| Classes | person + motorcycle, bicycle, car, bus, truck |
| person | Context only, detection/tracking, không violation evaluation |
| Tracker | ByteTrack |
| ROI | Manual polygon |
| ROI membership | Bottom-center |
| Zones | SIDEWALK / MONITORED / ALLOWED / IGNORE |
| Zone precedence | IGNORE > ALLOWED > SIDEWALK / MONITORED |
| Zone geometry | Shapely `Polygon.covers(Point)` |
| Violation threshold | Consecutive tracked frames inside ROI |
| State management | Per-track state machine |
| Violation logic | Separate Rule Engine (ViolationEngine) |
| Dedup | Track lock + cooldown + spatial |
| Event | SUSPECTED_AREA_OCCUPATION |
| Metadata storage | SQLite |
| Evidence | Snapshot trên filesystem |
| Video clip | Later / optional |
| RTSP | Out of MVP |
| Plate OCR / face recognition | Out |
| Fine-tuning | Chỉ sau baseline và failure analysis nếu cần |
| Main evaluation | Event-level metrics |

Freeze áp dụng cho scope, module, interface và quyết định ở trên. Nguồn/số lượng dữ liệu, model weights và threshold thử nghiệm vẫn có thể TBD; acceptance thresholds được chốt sau Baseline Run 0. Thay đổi architecture phải cập nhật Requirements, Solution Design và các docs liên quan trước khi triển khai.

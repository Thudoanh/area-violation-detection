# Requirements

> Functional requirements source of truth — **Frozen MVP v0.1**. Technical architecture: [Solution Design](solution_design.md). Phase 1 xây dựng code skeleton; Phase 2 tích hợp detector pretrained và baseline.

## 1. Functional Requirements

### FR-01 — Video Input

Hệ thống phải đọc được video file từ camera fixed-view. RTSP ngoài MVP.

Phase sau có thể hỗ trợ RTSP.

### FR-02 — Object Detection

Hệ thống phải phát hiện các object class được cấu hình.

```text
DETECTION_CLASSES = {person, motorcycle, bicycle, car, bus, truck}
VEHICLE_CLASSES = VIOLATION_TARGET_CLASSES = {motorcycle, bicycle, car, bus, truck}
```

`person` được detect và track chỉ để cung cấp context, không đi vào violation evaluation. Config không được mở rộng violation targets sang `person` trong MVP.

Detector phải nằm sau một abstraction/interface để có thể đổi backend.

### FR-03 — Multi-Object Tracking

Hệ thống phải duy trì `track_id` xuyên suốt nhiều frame cho cùng một object.

Tracker MVP là ByteTrack; implementation phải đáp ứng yêu cầu license.

### FR-04 — ROI Configuration

Hệ thống phải cho phép định nghĩa polygon zone theo camera.

Web demo phải hỗ trợ hai cách tạo ROI: người dùng click để vẽ polygon và hệ thống
đề xuất polygon tự động cho vùng kẻ/sơn ổn định. Kết quả tự động phải được preview
và có thể chỉnh sửa trước khi chạy pipeline; không được tự động coi là ROI đã xác nhận.

Zone type:

- `SIDEWALK`
- `MONITORED`
- `ALLOWED`
- `IGNORE`

### FR-05 — Zone Membership

Hệ thống phải xác định object có nằm trong zone hay không.

MVP dùng bottom-center point; điểm trên biên polygon tính là inside.

Precedence: `IGNORE > ALLOWED > SIDEWALK / MONITORED`. Object trong `SIDEWALK + ALLOWED` không sinh event. Bbox/mask overlap để giai đoạn sau.

### FR-06 — Stationary Detection

Hệ thống phải xác định object đang di chuyển hay đứng yên từ trajectory theo thời gian.

### FR-07 — Dwell Time

Hệ thống phải đo thời gian object đứng yên trong monitored zone.

Dwell time phải dựa trên timestamp/seconds, không phụ thuộc trực tiếp vào số frame.

### FR-08 — State Machine

Mỗi vehicle track được đánh giá vi phạm phải có trạng thái độc lập (person chỉ giữ tracking context):

- OUTSIDE
- ENTERING
- INSIDE_MOVING
- STATIONARY
- SUSPECTED_VIOLATION
- ALERTED
- CLOSED

### FR-09 — Rule Engine

MVP rule:

```text
object.class ∈ VEHICLE_CLASSES
AND object is inside SIDEWALK or MONITORED zone
AND object is NOT inside ALLOWED zone
AND object is NOT inside IGNORE zone
AND object is stationary
AND stationary_duration >= min_dwell_time_sec
→ SUSPECTED_AREA_OCCUPATION
```

### FR-10 — Duplicate Suppression

Hệ thống phải tránh sinh nhiều alert cho cùng một event.

Cơ chế:

- per-track lock;
- cooldown;
- spatial deduplication;

Re-identification heuristic là hướng mở rộng sau MVP.

### FR-11 — Event Storage

Event vượt qua Rule Engine và dedup phải lưu metadata trong SQLite, snapshot trên filesystem. Candidate bị suppress không tạo event mới. Schema canonical nằm trong Solution Design.

### FR-12 — Evidence

Tối thiểu phải lưu snapshot.

Video clip optional ở giai đoạn sau, không phải điều kiện hoàn thành MVP.

### FR-13 — Configuration

Các threshold không hard-code.

Phải đọc từ config:

- detector confidence;
- detection classes và violation target classes (hai tập riêng);
- stationary window;
- stationary movement threshold;
- minimum dwell time;
- exit grace time;
- cooldown;
- spatial dedup distance.

### FR-14 — Evaluation

Hệ thống phải hỗ trợ đánh giá event prediction với manual ground truth.

---

## 2. Non-Functional Requirements

### NFR-01 — Modularity

Detection, tracking, zones, violation logic và storage phải tách module.

### NFR-02 — Replaceable Detector

Thay detector không được yêu cầu sửa business rule.

### NFR-03 — Reproducibility

Experiment phải ghi:

- git commit;
- dataset version;
- model;
- config;
- hardware;
- metrics.

### NFR-04 — Performance

MVP phải báo cáo:

- average FPS;
- per-frame latency;
- event detection delay.

Chưa chốt acceptance threshold trước baseline.

### NFR-05 — Maintainability

Business rule phải cấu hình được và không hard-code theo một camera.

### NFR-06 — Auditability

Mỗi event phải có đủ metadata để truy vết lại video, object và rule đã kích hoạt.

### NFR-07 — Privacy

MVP không xử lý khuôn mặt hoặc OCR biển số.

### NFR-08 — Licensing

Các dependency, model weights và dataset phải được audit license trước khi dùng trong sản phẩm thương mại.

Repository tham khảo không được copy source code nếu không có quyền sử dụng rõ ràng.

---

## 3. MVP Technical Parameters

Các giá trị dưới đây chỉ là initial values để thử nghiệm:

```yaml
detector:
  confidence_threshold: 0.4

stationary:
  window_sec: 3
  max_displacement_px: 15

violation:
  min_dwell_time_sec: 30
  exit_grace_sec: 3

dedup:
  cooldown_sec: 60
  spatial_distance_px: 50
```

Các threshold này phải được tune bằng experiment.

---

## 4. Scope và các giá trị còn TBD

MVP chỉ xét vehicle occupation: phương tiện đứng yên đủ lâu trong vùng hợp lệ. Input là video file camera cố định; ROI polygon thủ công; output là event và snapshot.

Ngoài MVP: RTSP, multi-camera đồng thời, allowed-zone schedule, street vending, bàn ghế, hàng hóa, biển quảng cáo, vật liệu xây dựng, behavior recognition, OCR biển số, nhận diện khuôn mặt, automatic sidewalk segmentation, PTZ và xử phạt tự động. Clip optional ở giai đoạn sau.

Còn TBD: nguồn video, resolution/FPS, số lượng/split dataset, backend/weights cụ thể trong YOLOX hoặc RTMDet cho Phase 2, threshold vận hành qua experiment. Acceptance thresholds được chốt sau Baseline Run 0.

## 5. Interface contracts

- Frame Provider trả frame và timestamp theo giây trên timeline video.
- Detector trả `list[Detection]`: bbox pixel xyxy, confidence và class chuẩn hóa.
- ByteTrack adapter nhận detections, frame, timestamp và trả `list[TrackedObject]` có ID, class, bbox, confidence, timestamp.
- Zone Manager trả membership và precedence; Stationary Detector trả motion; Dwell Timer trả stationary duration theo giây.
- State Machine quản lý trạng thái per-track; Rule Engine riêng trả candidate hoặc không có event; Deduplicator quyết định phát event; Storage lưu metadata và snapshot.

Data models và lifecycle chi tiết theo [Solution Design](solution_design.md).

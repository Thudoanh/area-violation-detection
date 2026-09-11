# Project Context

Tài liệu này cung cấp bối cảnh ngắn gọn cho người phát triển và coding agent
trước khi đọc code hoặc thực hiện thay đổi. Khi có khác biệt, yêu cầu mới nhất đã
được phê duyệt và `docs/solution_design.md` là nguồn thiết kế chi tiết; tài liệu
này là bản định hướng tổng quan.

## 1. Mục tiêu dự án

Hệ thống phân tích video từ camera cố định để:

- phát hiện người và phương tiện trong từng frame;
- theo dõi object qua nhiều frame bằng `track_id`;
- kiểm tra bottom-center của phương tiện có nằm trong ROI hay không;
- xác nhận suspected violation theo rule cấu hình;
- tạo video annotation, event metadata và snapshot phục vụ hậu kiểm.

Kết quả `SUSPECTED_AREA_OCCUPATION` là cảnh báo kỹ thuật, không phải kết luận vi
phạm pháp luật.

## 2. Architecture

```text
Video/Input
    ↓
Detection — YOLO11n pretrained COCO
    ↓
Tracking — ByteTrack
    ↓
ROI Engine — Shapely bottom-center × polygon
    ↓
N-frame Validation + State Machine
    ↓
Rule Engine
    ↓
Per-track Lock + Cooldown + Spatial Deduplication
    ↓
Violation Event
    ↓
SQLite + Snapshot + Annotated Video + Evaluation
```

Pipeline dùng chung cho CLI và web demo nằm trong `src/pipeline/`.

## 3. Nguyên tắc kiến trúc

Detection, Tracking, ROI, Rule Engine, deduplication và storage phải độc lập.

Có thể thay detector thông qua adapter `BaseDetector` mà không sửa:

- Tracking;
- ROI;
- N-frame validation;
- Rule Engine;
- event lifecycle;
- storage.

Các data contract giữa module dùng model chuẩn hóa trong `src/core/models.py`.
Business rule không được phụ thuộc class ID nội bộ của detector.

## 4. Detector hiện tại

Detector đang hoạt động: **YOLO11n pretrained COCO** qua Ultralytics.

```text
model: yolo11n
weights: weights/yolo11n.pt
input size mặc định: 640
```

Classes được detect:

- `person`;
- `motorcycle`;
- `car`;
- `bicycle`;
- `bus`;
- `truck`.

`person` chỉ cung cấp context, không được đưa vào violation evaluation. Baseline
hiện tại không dùng YOLOv8m, không dùng `best_v5.pt` và chưa fine-tune trên VIRAT.

## 5. Tracking

Tracker hiện tại: **ByteTrack**.

Mục tiêu:

- duy trì `track_id` cho object qua nhiều frame;
- liên kết detection hiện tại với track đang tồn tại;
- giữ bbox, class, confidence và video timestamp trong `TrackedObject`.

Track ID chỉ có phạm vi trong một video/run. ByteTrack hiện không có ReID; mất
detection, che khuất hoặc cảnh đông có thể gây mất track hay đổi ID.

## 6. ROI

ROI là polygon theo camera, do người dùng cấu hình thủ công hoặc chấp nhận từ đề
xuất auto-ROI của web demo.

Các loại zone:

- `SIDEWALK`;
- `MONITORED`;
- `ALLOWED`;
- `IGNORE`.

Thứ tự ưu tiên:

```text
IGNORE > ALLOWED > SIDEWALK / MONITORED
```

Membership được quyết định bằng:

```text
bottom_center = ((x1 + x2) / 2, y2)
Shapely Polygon.covers(Point(bottom_center))
```

Điểm trên biên được tính là inside. Bbox giao với ROI nhưng bottom-center nằm
ngoài polygon vẫn được tính là outside.

## 7. Rule Engine

Rule hiện tại:

```text
object.class ∈ {motorcycle, bicycle, car, bus, truck}
AND bottom-center nằm trong SIDEWALK hoặc MONITORED
AND object không nằm trong ALLOWED
AND object không nằm trong IGNORE
AND consecutive_inside_frames >= min_inside_frames
→ SUSPECTED_AREA_OCCUPATION
```

Ngưỡng mặc định:

```yaml
violation:
  min_inside_frames: 30
  exit_grace_sec: 3
```

Đây là validation theo số frame, không phải stationary/dwell-time theo giây. Bộ
đếm reset khi track ra khỏi target zone, vào vùng loại trừ, đổi zone hoặc bị mất.
Giá trị N phải được tune theo FPS và dữ liệu thực tế.

State lifecycle:

```text
OUTSIDE → ENTERING → INSIDE_PENDING
        → SUSPECTED_VIOLATION → ALERTED → CLOSED
```

Sau Rule Engine, event tiếp tục qua per-track lock, cooldown và spatial
deduplication. Mặc định cooldown là 60 giây và spatial distance là 50 px.

## 8. Dataset

Evaluation dataset hiện được định hướng sử dụng: **VIRAT**.

VIRAT dùng cho kiểm thử/evaluation pipeline, bao gồm detection, tracking, ROI và
event-level violation. Dataset này không được dùng để train hoặc fine-tune
YOLO11n trong baseline hiện tại.

Trước mỗi experiment phải ghi rõ:

- video và subset VIRAT sử dụng;
- phiên bản ground truth;
- ROI tương ứng;
- FPS/resolution;
- config và Git commit.

## 9. Evaluation

Đánh giá cần tách theo từng tầng:

- Detection: Precision, Recall, F1, mAP nếu có detection ground truth;
- Tracking: ID stability, ID switches, IDF1/HOTA nếu có annotation;
- ROI: đúng/sai của bottom-center membership, đặc biệt ở biên polygon;
- N-frame validation: reset và threshold theo từng track/zone;
- Event level: Event Precision, Recall, F1, False Alerts/hour, Detection Delay và
  Duplicate Alert Rate;
- vận hành: processing FPS và latency.

Automated tests chỉ xác nhận hành vi code, không thay thế baseline trên video và
ground truth thực tế.

## 10. Current Status

- Phase 1 — Project skeleton: **Done**.
- Phase 2 — YOLO11n detection baseline integration: **Done**.
- Phase 3 — ByteTrack integration: **Done**.
- Phase 4 — Shapely ROI/zone management: **Done**.
- Phase 5 — Consecutive N-frame validation: **Done**.
- Phase 6 — State machine và violation rule: **Done**.
- Phase 7 — Deduplication, SQLite và snapshot: **Done**.
- Phase 8 — VIRAT evaluation và Baseline Run 0: **In progress**.

Web demo và CLI đã dùng chung pipeline. Chưa có acceptance threshold chính thức
cho chất lượng event trước khi hoàn thành Baseline Run 0.

## 11. Repository Constraints

Không tự ý thực hiện các thay đổi sau nếu chưa có yêu cầu rõ ràng:

- thay architecture;
- đổi evaluation dataset;
- đổi detector/model weights;
- thay đổi tập violation target classes;
- sửa public interface hoặc event schema;
- thay đổi semantics ROI và zone precedence;
- thay đổi threshold mặc định;
- xóa hoặc làm mất dữ liệu event đã tồn tại.

Nếu cần thay đổi kiến trúc, phải cập nhật đồng bộ code, config, automated tests,
`project_context.md`, `docs/requirements.md`, `docs/solution_design.md` và
`docs/experiment_log.md`.

## 12. Cách làm việc

Khi sửa code:

1. Đọc `project_context.md`.
2. Đọc `docs/solution_design.md` và tài liệu liên quan.
3. Kiểm tra implementation và tests hiện tại trước khi kết luận.
4. Ưu tiên thay đổi nhỏ nhất đáp ứng đúng yêu cầu.
5. Không phá backward compatibility nếu không được yêu cầu rõ ràng.
6. Cập nhật test cho cả positive case, negative case và boundary case.
7. Chạy test liên quan, sau đó chạy toàn bộ test suite.
8. Ghi rõ thay đổi ảnh hưởng architecture hoặc metric vào experiment log.

Không được suy ra chất lượng thực tế chỉ từ unit test. Mọi tuyên bố về loại bỏ
false positive hoặc đạt accuracy phải có kết quả evaluation và ground truth.

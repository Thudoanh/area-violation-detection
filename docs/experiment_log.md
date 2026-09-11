# Experiment Log

Tài liệu này dùng để ghi lại mọi thay đổi có ảnh hưởng đến kết quả.

Không xóa experiment cũ. Mỗi run phải có ID riêng.

## Experiment naming

| ID | Tên | Trạng thái |
|---|---|---|
| EXP-000 | Project Initialization | Tài liệu MVP v0.1 đã cập nhật; chưa có kết quả chạy |
| EXP-001 | Detection Baseline | NOT STARTED |
| EXP-002 | Tracking Baseline | NOT STARTED |
| EXP-003 | ROI Integration | NOT STARTED |
| EXP-004 | Stationary Detection | NOT STARTED |
| EXP-005 | Violation Pipeline Baseline | NOT STARTED — Baseline Run 0 |

EXP-000 ghi khởi tạo project, không phải baseline end-to-end. EXP-001–005 là ID dành cho experiment dự kiến, chưa có kết quả. Run tiếp theo cấp ID tăng dần, không ghi đè kết quả cũ. Phase 1 là code skeleton; Phase 2 bắt đầu tích hợp detector pretrained và baseline. Acceptance thresholds chỉ chốt sau Baseline Run 0.

---

## Experiment Template

### EXP-XXX — <Tên experiment>

**Date:** YYYY-MM-DD  
**Owner:**  
**Git commit:**  
**Dataset version:**  
**Split:** dev / test  
**Hardware:**  
**Video set:**  

### Objective

Mục tiêu của experiment.

### Change

Thay đổi so với baseline trước.

### Configuration

```yaml
detector:
  backend:
  model:
  confidence_threshold:

tracking:
  backend:

stationary:
  window_sec:
  max_displacement_px:

violation:
  min_dwell_time_sec:
  exit_grace_sec:

dedup:
  cooldown_sec:
  spatial_distance_px:
```

### Results

| Metric | Value |
|---|---:|
| Event Precision | |
| Event Recall | |
| Event F1 | |
| False Alerts / Hour | |
| Median Detection Delay | |
| Duplicate Alert Rate | |
| Detector mAP (nếu có detection GT) | |
| FPS | |
| Per-frame Latency | |

### Failure Analysis

#### False Positives

- 

#### False Negatives

- 

#### Duplicate Events

- 

### Observation

-

### Decision

- Keep / Reject / Investigate

### Next Step

-

---

# Experiment History

## EXP-009 — Shapely + Consecutive N-frame Architecture

**Date:** 2026-09-11.
**Change:** Giữ YOLO11n pretrained COCO và ByteTrack; thay OpenCV
`pointPolygonTest` bằng Shapely `Polygon.covers(Point)`; thay stationary detector
và dwell-time threshold bằng bộ đếm `min_inside_frames` liên tiếp theo track/zone.
Bộ đếm reset khi ra ROI, vào ALLOWED/IGNORE, đổi zone hoặc mất track. Giữ
per-track lock, cooldown và spatial deduplication. Ngưỡng khởi tạo là 30 frame và
cần tune theo FPS/dataset thực tế. Các cột thời gian cũ trong SQLite được giữ để
tương thích dữ liệu.

## EXP-008 — Vietnamese Web Demo + ROI Suggestion

**Date:** 2026-09-10.  
**Change:** Thêm web demo Gradio tiếng Việt với upload/preview video, vẽ ROI bằng
click hoặc đề xuất ROI classical từ nhiều frame, điều chỉnh threshold theo run,
video annotate phát trên trình duyệt, bảng event, analytics và gallery snapshot.
Vòng xử lý video được đưa vào reusable runner để CLI và web dùng chung. SQLite giữ
ở `data/events/events.db`; kết quả UI được lọc bằng `run_id`.  
**Auto ROI scope:** Chỉ đề xuất vùng kẻ/sơn ổn định có đường chéo và tương phản rõ;
không phải semantic sidewalk segmentation. Temporal voting giảm chi tiết xuất hiện
tạm thời và người dùng phải kiểm tra polygon trên preview.  
**Verification:** 201 tests pass, 1 conditional smoke test skipped trong full suite;
YOLO11 real-model smoke 25 tests pass; Gradio 6.26 khởi động thành công tại
`127.0.0.1:7860` và trả HTTP 200. Chưa có video CCTV/ROI thật để đánh giá chất lượng
auto ROI hoặc event-level accuracy.

## EXP-007 — Detection migration: YOLO11n

**Date:** 2026-09-09.  
**Change:** CLI chuyển từ YOLOX-S ONNX sang YOLO11n COCO, `ultralytics==8.4.144`, PyTorch CPU. Chỉ đổi adapter detection, import/khởi tạo detector, config và dependency; các module downstream giữ nguyên. YOLOX source giữ để tham chiếu lịch sử, không nằm trong đường chạy hiện tại.  
**Weights:** `weights/yolo11n.pt`, nguồn `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt`. SHA-256: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.  
**Config:** input 640, confidence 0.4, NMS IoU 0.45, giữ 6 class và ID nội bộ. Ultralytics xử lý preprocessing/NMS; adapter chuẩn hóa xyxy ảnh gốc, confidence float, class_id int và class_name.  
**Verification:** unit tests adapter; integration output mô phỏng qua ByteTrack/ROI/rule/storage thật; smoke pretrained thật trên ảnh bus có sẵn trong package qua detection/tracking/ROI/rule. Lưu trữ kiểm thử nằm trong temporary directories. Hash toàn bộ source core/tracking/zones/violation/storage/pipeline không đổi.  
**Limitations:** chưa có video CCTV hoặc ground truth; chưa chạy Baseline Run 0, chưa đo chất lượng/FPS thực tế hoặc GPU. Không suy ra chất lượng cảnh báo từ smoke test ảnh mẫu.

## EXP-000 — Project Initialization

**Date:** TBD  
**Objective:** Chốt tài liệu, scope và Architecture Freeze v0.1 trước Phase 1.

### Planned configuration

- detector-agnostic; Phase 2 ưu tiên pretrained YOLOX hoặc RTMDet;
- ByteTrack;
- manual polygon ROI;
- bottom-center zone test;
- stationary window = TBD;
- dwell-time = TBD;
- per-track lock + cooldown + spatial dedup;
- person context-only; năm vehicle classes được xét violation;
- SUSPECTED_AREA_OCCUPATION; SQLite metadata + filesystem snapshot.

### Expected outcome

README và docs thống nhất canonical rule, class scope, zone precedence, pipeline, storage và framework evaluation. Việc chạy baseline end-to-end và thu thập failure modes thuộc EXP-005.

### Status

`DOCUMENTATION UPDATED — NO RUN RESULTS`

Architecture Freeze v0.1 đã ghi trong Solution Design. Chưa có code skeleton, baseline metrics hoặc kết quả thực nghiệm.

---

## Rules for Experiment Logging

1. Mỗi thay đổi threshold quan trọng phải tạo experiment mới.
2. Không overwrite metric cũ.
3. Luôn ghi dataset version.
4. Luôn ghi git commit.
5. Không so sánh hai run dùng test set khác nhau mà không ghi chú.
6. Model weights phải có version/hash nếu có thể.
7. Nếu có manual correction, phải ghi rõ.
8. Khi chọn một config làm baseline mới, ghi rõ lý do.

---

## EXP-001 — Detection Baseline

**Detector:** YOLOX qua ONNX Runtime, CPU.  
**Pretrained model:** YOLOX-S COCO, ONNX release `0.1.1rc0`, input 640×640; nguồn [chính thức](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx).  
**Weights SHA-256 (file đã tải):** `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`.  
**Confidence threshold:** 0.4; NMS IoU 0.45, giá trị kỹ thuật ban đầu.  
**Detection classes:** person, motorcycle, car, bicycle, bus, truck.  
**Video test:** TBD — repo chưa có video thực tế; không tải video thay thế.  
**Hardware:** TBD — chưa chạy baseline video.  
**Qualitative observation / Result:** TBD — unit/import tests không chứng minh chất lượng detector; chưa có annotated video baseline hoặc FPS thực nghiệm.

---

## EXP-002 — Tracking Baseline

**Tracker backend:** ByteTrack qua `supervision==0.25.0` (MIT); một instance cho mỗi video/run.  
**Config:** `track_thresh=0.5`, `track_buffer=30`, `match_thresh=0.8`; FPS lấy từ video, buffer thực tế là `int(FPS / 30 * 30)` frame. Detector YOLOX-S và confidence 0.4 giữ nguyên từ EXP-001.  
**Video:** TBD — chưa có input thực tế; không download hoặc generate video giả.  
**Hardware:** TBD — chưa chạy baseline video.  
**Unique track IDs:** TBD.  
**ID stability observation:** TBD — kiểm tra bằng dữ liệu bbox trong tests không thay thế đánh giá video thực tế.  
**FPS:** TBD.  
**Known issues:** Association không phân biệt class, nên nhãn có thể đổi theo detection. Ngưỡng detector 0.4 giới hạn detection confidence thấp đến tracker; track mới cần confidence ít nhất 0.6 theo backend này. Occlusion dài/crossing có thể gây ID switch. Summary theo class quan sát đầu tiên của ID; unique IDs không phải ground-truth object count. Timestamp của `TrackedObject` lấy trực tiếp từ frame; ByteTrack nội bộ vẫn cập nhật theo frame.

---

## EXP-003 — ROI / Zone Integration

**Video:** TBD — chưa có video thật; chưa chạy selector GUI hoặc pipeline end-to-end trên video.  
**camera_id:** `CAM_001` theo config mặc định.  
**Zone types:** SIDEWALK / MONITORED / ALLOWED / IGNORE.  
**Polygon config:** `configs/zones/cam_001.json`, danh sách rỗng chờ ROI được vẽ từ camera thật.  
**Anchor method:** bottom-center bbox; geometry dùng OpenCV pointPolygonTest.  
**Qualitative observation:** TBD — unit tests xác minh membership, chồng vùng và precedence; chưa có nhận xét trên scene thực tế.  
**Boundary issues:** Điểm trên biên tính inside; jitter thực tế TBD. Vùng hiệu lực ưu tiên IGNORE, ALLOWED, rồi target đầu tiên theo JSON. Polygon phải đơn giản, tọa độ đúng resolution/view video gốc; không tự scale. Summary là membership của unique track IDs, không phải violation.

---

## EXP-004 — Stationary Detection / Dwell Baseline

**Configuration:** bottom-center, cửa sổ 3 giây, displacement đầu/cuối ≤15 px. Dwell theo giây stationary trong effective target zone, tách residence time. Reset khi moving/excluded/lost/đổi zone hoặc class.  
**Video / hardware / observations:** TBD — chưa có video thật. Unit tests không thay thế baseline camera. Hạn chế đầu/cuối: object đi rồi quay về có thể bị coi stationary.

## EXP-005 — Violation Pipeline Baseline

Giữ tên và vai trò **Baseline Run 0** đã chốt trong Evaluation; phần triển khai rule hoàn thành nhưng chưa thực hiện baseline run.  
**Configuration:** vehicle-only, IGNORE > ALLOWED > SIDEWALK/MONITORED, stationary dwell ≥30s, exit grace 3s; lifecycle gồm ENTERING/ALERTED/CLOSED theo docs.  
**Video / hardware / results:** TBD. Chưa tính metric Phase 8, chưa tune threshold.

## EXP-006 — Dedup + Event Storage

**Configuration:** per-track OPEN lock; cooldown 60s từ đóng event; spatial <50px cùng run/camera/zone/class, bảo vệ hai ID từng quan sát đồng thời. Snapshot JPEG trước SQLite insert; EOF đóng event với left_at nullable. Schema canonical của Solution Design, không tạo schema event thứ hai.  
**Verification scope:** unit/integration tests dùng bbox, timestamp và ảnh bộ nhớ; SQLite/JPEG trong temporary directories. Không download/generate video.  
**Video / hardware / candidates / events / snapshots / baseline observations:** TBD — chưa có video CCTV thật. Counts trong tests không phải baseline metrics. Grace chỉ giữ lifecycle, không cho phát candidate khi thiếu quan sát.

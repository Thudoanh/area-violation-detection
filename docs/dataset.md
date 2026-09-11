# Dataset

> Strategy và scenario theo Frozen MVP v0.1; dataset cụ thể còn Draft/TBD. Scope theo [Requirements](requirements.md).

## 1. Mục tiêu dữ liệu

Project cần hai loại dữ liệu khác nhau:

1. dữ liệu để đánh giá/phát triển object detector;
2. dữ liệu video để đánh giá toàn pipeline violation detection.

Không nên chỉ đánh giá detector bằng mAP rồi kết luận hệ thống violation hoạt động tốt.

---

## 2. Dataset Layers

### Layer A — Detection Dataset

Dùng để kiểm tra/fine-tune detector.

Detection classes (MVP):

- person
- motorcycle
- car
- bicycle
- bus
- truck

`person` là context-only; event ground truth chỉ gán cho `motorcycle`, `bicycle`, `car`, `bus`, `truck` theo canonical rule.

Có thể bắt đầu bằng pretrained weights trên dataset phổ biến nếu license cho phép.

Sau khi chạy baseline trên camera Việt Nam, quyết định có cần custom dataset hay không.

### Layer B — Pipeline Evaluation Videos

Đây là dataset quan trọng nhất cho MVP.

Mỗi video phải có:

- camera ID;
- resolution;
- FPS;
- scene;
- lighting condition;
- sidewalk/monitoring polygon;
- ground-truth event timestamp;
- object type;
- positive/negative label.

---

## 3. Required Scenarios

Dataset pipeline cần cover:

### Positive

- motorcycle parked on sidewalk;
- car parked on sidewalk;
- bicycle stopped in monitored zone;
- multiple vehicles occupying zone;
- vehicle enters then stops;
- violation near ROI boundary.

### Negative

- vehicle crosses ROI;
- vehicle stops briefly;
- person walking;
- vehicle outside polygon;
- allowed-zone occupation;
- ignore-zone occupation;
- bbox/ROI overlap nhỏ hơn ngưỡng 0.2;
- tracker ID switch (không được tạo alert trùng);

Occlusion, low light, rain và crowded scene là điều kiện khó cần có cả positive và negative; không tự động coi chúng là negative. Case gần biên gán nhãn theo tỷ lệ bbox/ROI giao nhau, số frame liên tiếp và precedence `IGNORE > ALLOWED > SIDEWALK / MONITORED`.

---

## 4. Ground Truth Event Schema

Ví dụ:

```json
{
  "video_id": "cam01_001",
  "camera_id": "CAM01",
  "events": [
    {
      "event_id": "GT_001",
      "object_class": "motorcycle",
      "start_time_sec": 32.5,
      "min_inside_frames": 30,
      "violation_time_sec": 62.5,
      "end_time_sec": 84.0,
      "zone_id": "SW01",
      "zone_type": "SIDEWALK",
      "label": "SUSPECTED_AREA_OCCUPATION"
    }
  ]
}
```

Timestamp là giây từ đầu video: `start_time_sec` là thời điểm vào vùng và
`violation_time_sec` là frame đạt ngưỡng theo config annotation. Ví dụ dùng 30
frame, không phải threshold nghiệm thu. GT ID độc lập với tracker ID; dùng
bbox/keyframes hoặc track hint để phân biệt nhiều phương tiện cùng zone.

Nếu có thể, lưu thêm:

```text
track_hint
bbox/keyframes
notes
reviewer
review_status
```

---

## 5. Suggested Data Layout

```text
data/
├── videos/
│   ├── raw/
│   ├── dev/
│   └── test/
├── datasets/
│   └── detection/
├── ground_truth/
│   ├── cam01_001.json
│   └── ...
└── events/
```

---

Zone config dùng chung tại `configs/zones/` theo Solution Design.

## 6. Data Split

Pipeline evaluation nên split theo **video/camera/session**, không split frame ngẫu nhiên.

Ví dụ:

```text
development set
  → tune threshold

test set
  → final evaluation
```

Không tune threshold trên test set.

Nếu có nhiều camera:

- dev và test nên có camera khác nhau nếu muốn đo generalization;
- hoặc ít nhất session/time khác nhau.

---

## 7. Annotation Guidelines

Một ground-truth event cần xác định:

1. thời điểm object đi vào monitored zone;
2. frame object bắt đầu chuỗi quan sát liên tiếp trong ROI;
3. thời điểm đủ điều kiện violation theo annotation policy;
4. thời điểm object rời zone;
5. class;
6. zone;
7. ambiguity note.

Để tránh subjective labeling, annotation policy phải được chốt trước khi annotate.

---

## 8. Dataset Quality Checklist

Mỗi video phải kiểm tra:

- file mở được;
- timestamp/FPS hợp lệ;
- polygon khớp scene;
- không đổi camera view giữa video;
- ground truth được review;
- không duplicate video giữa dev/test;
- license/source được ghi rõ;
- không đưa dữ liệu nhạy cảm không cần thiết.

---

## 9. Dataset Registry

Nên có file metadata CSV:

```text
video_id,
camera_id,
source,
license,
duration_sec,
fps,
resolution,
lighting,
weather,
scene_type,
num_positive_events,
num_negative_scenarios,
split,
annotation_status
```

---

## 10. License Tracking

Mỗi dataset phải ghi:

- source URL/path;
- owner;
- license;
- commercial use allowed?;
- modification allowed?;
- attribution required?;
- internal-only?;
- review date.

Nếu license không rõ, dữ liệu chỉ dùng nghiên cứu nội bộ cho đến khi được xác nhận.

---

## 11. Initial Dataset Strategy

```text
Pretrained detector
        ↓
Collect representative videos
        ↓
Run baseline
        ↓
Failure analysis
        ↓
Fine-tune only if needed
```

Phase 1 của project là code skeleton; chuẩn bị dữ liệu có thể tiến hành song song. Phase 2 tích hợp pretrained detector (ưu tiên YOLOX hoặc RTMDet), annotate event và chạy baseline trước khi cân nhắc training.

Chỉ thu thêm frame/fine-tune nếu failure analysis cho thấy detector chưa đáp ứng các scene như motorcycle nhỏ, camera cao, đêm hoặc occlusion. Mở rộng sang goods, stall, chair/table, signboard hay construction materials nằm ngoài MVP và cần dataset riêng.

| Dataset decision | Status |
|---|---|
| Dataset source | TBD |
| Number of videos | TBD |
| Train/dev/test size | TBD |
| Annotation policy chi tiết và version | Chốt trước annotation theo canonical rule |

Không yêu cầu chốt dataset final trước Phase 1; các scenario ở mục 3 là yêu cầu coverage.

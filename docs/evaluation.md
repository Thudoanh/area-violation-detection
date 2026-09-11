# Evaluation

## 1. Evaluation Goal

Đánh giá phải trả lời hai câu hỏi độc lập:

1. detector có nhìn thấy object đúng không?
2. toàn pipeline có tạo đúng violation event không?

Metric chính của project là **event-level metrics** cho `SUSPECTED_AREA_OCCUPATION` trên năm vehicle classes. `person` chỉ đánh giá ở detection/context, không có GT violation event.

| Nhóm | Metrics |
|---|---|
| Primary | Event Precision, Event Recall, Event F1, False Alerts / Hour, Detection Delay, Duplicate Alert Rate |
| Secondary | Detector mAP, FPS, Latency |

Các detector/tracking metrics khác bên dưới dùng chẩn đoán bổ sung. Framework đã chốt cho MVP v0.1; kết quả và acceptance thresholds còn TBD.

---

## 2. Detector Metrics

Nếu có detection ground truth:

- Precision
- Recall
- F1
- mAP@50
- mAP@50:95
- per-class AP

Các metric này chỉ đánh giá detection layer.

---

## 3. Tracking Metrics

Nếu có tracking ground truth:

- IDF1
- HOTA
- ID switches

Nếu chưa có tracking annotation đầy đủ, MVP có thể dùng sanity metrics:

- track fragmentation count;
- average track duration;
- ID switch observed on critical cases.

---

## 4. Event Matching

Một predicted event được match với ground truth khi:

- cùng video/run được đánh giá và camera;
- cùng zone;
- cùng event type và object class;
- predicted violation timestamp nằm trong matching window;
- không match cùng GT hai lần.

Ví dụ:

```text
|predicted_time - gt_violation_time| <= tolerance_sec
```

`tolerance_sec` và policy đối chiếu bbox/keyframes để phân biệt nhiều xe cùng zone phải được chốt trước run và ghi trong experiment. Matching một-một; khi có nhiều ứng viên hợp lệ, ưu tiên cặp có sai lệch timestamp nhỏ nhất, tie-break bằng event ID. GT không phụ thuộc tracker ID.

---

## 5. Event Metrics

### True Positive

Predicted violation match một GT violation.

### False Positive

Predicted violation không match GT.

### False Negative

GT violation không có prediction.

### Precision

```text
TP / (TP + FP)
```

### Recall

```text
TP / (TP + FN)
```

### F1

```text
2 * TP / (2 * TP + FP + FN)
```

---

## 6. Operational Metrics

### False Alerts per Hour

```text
FP / total_video_hours
```

Đây là metric rất quan trọng với camera surveillance.

### Detection Delay

```text
predicted_violation_time - gt_violation_time
```

Tính trên các cặp TP, giữ dấu để thấy alert sớm; không gán delay bằng 0 cho FN.

Báo cáo:

- mean;
- median;
- p95.

### Duplicate Alert Rate

```text
duplicate_events / total_predicted_events
```

Duplicate là prediction bổ sung cho cùng một GT episode đã có prediction được match, xác nhận bằng identity/spatial evidence. Duplicate vẫn tính FP; không match lại GT để tăng TP. Trường hợp không có prediction báo rate N/A. Với các tỷ lệ khác, mẫu số bằng 0 cũng báo N/A kèm TP/FP/FN; không tự coi là đạt yêu cầu.

### FPS

Average processing FPS.

### Latency

Per-frame inference + pipeline latency.

---

## 7. Scenario-Level Evaluation

Báo cáo metric theo nhóm:

- motorcycle;
- car;
- daytime;
- nighttime;
- crowded;
- boundary;
- occlusion;
- allowed zone;
- short stop;
- pass-through.

Điều này giúp xác định failure mode.

---

## 8. Baseline Run 0

Baseline Run 0 là baseline end-to-end đầu tiên, tương ứng `EXP-005 — Violation Pipeline Baseline` (không phải EXP-000 khởi tạo tài liệu).

Baseline run phải freeze:

```text
model
tracker config
zone
minimum consecutive inside frames
dedup threshold
dataset version
code commit
```

Output tối thiểu:

| Metric | Value |
|---|---:|
| Event Precision | TBD |
| Event Recall | TBD |
| Event F1 | TBD |
| False Alerts / Hour | TBD |
| Median Detection Delay | TBD |
| Duplicate Alert Rate | TBD |
| Detector mAP (nếu có detection GT) | TBD |
| FPS | TBD |
| Per-frame Latency | TBD |

---

## 9. Threshold Tuning

Các tham số cần tune:

```text
detector confidence
minimum consecutive inside frames
exit grace
dedup cooldown
spatial dedup distance
```

Quy trình:

```text
dev set
→ baseline
→ failure analysis
→ change one/few variables
→ rerun
→ compare
→ freeze
→ final test
```

Không tune trực tiếp trên test set.

---

## 10. Acceptance Criteria

> Acceptance thresholds will be defined after Baseline Run 0.

Threshold kỹ thuật khởi tạo trong config không phải acceptance thresholds. Matching tolerance phải được chốt trước mỗi run để metrics có thể tái lập.

Sau baseline, acceptance threshold cần được mentor/PO duyệt dựa trên:

- mức false alarm chấp nhận được;
- recall yêu cầu;
- camera environment;
- hardware;
- realtime/offline requirement.

---

## 11. Error Taxonomy

Mỗi FP/FN nên gán một cause:

### Detection error

- missed object;
- wrong class;
- low confidence.

### Tracking error

- ID switch;
- track fragmentation;
- lost track.

### Zone error

- wrong ROI;
- boundary issue.

### Consecutive-frame error

- inside sequence reset by missed detection or track-ID change;
- frame threshold too short/long for the source FPS.

### Dedup error

- duplicate event;
- valid re-entry suppressed.

### Business-rule error

- allowed case flagged;
- violation ignored.

---

## 12. Evaluation Output Files

```text
results/
└── <experiment_id>/
    ├── metrics.json
    ├── predictions.json
    ├── errors.csv
    ├── config.yaml
    └── summary.md
```

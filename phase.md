Phase 1 — Project Skeleton
Tạo cấu trúc repo, config, core models, video input, test cơ bản.
Phase 2 — Detection Baseline
Tích hợp detector permissive như YOLOX hoặc RTMDet, detect person, motorcycle, car, bicycle, bus, truck.
Phase 3 — Tracking
Tích hợp ByteTrack, duy trì track_id, kiểm tra ID stability.
Phase 4 — ROI / Zone Management
Tạo polygon ROI và hỗ trợ SIDEWALK, MONITORED, ALLOWED, IGNORE.
Phase 5 — Stationary Detection + Dwell Time
Xác định moving/stationary từ trajectory và tính thời gian chiếm dụng theo giây.
Phase 6 — State Machine + Violation Rule
Ghép logic trạng thái và rule để sinh SUSPECTED_AREA_OCCUPATION.
Phase 7 — Deduplication + Event Storage
Chống duplicate alert, cooldown, spatial dedup, lưu event + snapshot.
Phase 8 — Evaluation & Baseline Run
Chuẩn bị ground truth, tính Event Precision/Recall/F1, False Alerts/hour, Detection Delay, FPS; sau đó tune threshold.
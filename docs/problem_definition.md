# Problem Definition

> Phạm vi nghiệp vụ đã chốt — Frozen MVP v0.1.

## 1. Bối cảnh

Vỉa hè và các khu vực công cộng có thể bị chiếm dụng bởi phương tiện hoặc vật thể, gây cản trở lưu thông và làm giảm hiệu quả quản lý đô thị.

Trong hệ thống camera giám sát, bài toán không chỉ là phát hiện một object xuất hiện trong ảnh. Hệ thống phải xác định:

1. object nào đang được theo dõi;
2. object có nằm trong vùng cần giám sát hay không;
3. object đang di chuyển hay đứng yên;
4. object tồn tại trong vùng bao lâu;
5. object có thuộc vùng được phép hoặc vùng bỏ qua hay không;
6. khi nào một chuỗi quan sát đủ điều kiện để tạo một sự kiện nghi ngờ vi phạm.

Business problem: phát hiện object chiếm dụng trái phép/không mong muốn trong vùng giám sát để phục vụ hậu kiểm.

```text
Video Event Detection
= Object Detection + Tracking + Spatial Reasoning
  + Temporal Reasoning + Rule-based Event Detection
```

---

## 2. Problem Statement

Xây dựng hệ thống AI có khả năng:

> Phân tích video từ camera cố định để phát hiện phương tiện chiếm dụng vỉa hè hoặc một khu vực được cấu hình để giám sát, dựa trên vị trí trong ROI, trạng thái di chuyển và thời gian tồn tại trong vùng; sau đó tạo `suspected violation event` kèm bằng chứng.

---

## 3. Định nghĩa "lấn chiếm" trong phạm vi MVP

Trong MVP, "lấn chiếm" được định nghĩa ở mức kỹ thuật:

```text
Phương tiện thuộc nhóm cần giám sát
+ nằm trong ROI
+ đứng yên
+ đủ lâu
+ không thuộc allowed/ignore zone
= suspected violation
```

Đây **không phải định nghĩa pháp lý cuối cùng**.

Ngưỡng số frame liên tiếp chỉ phục vụ nhận diện kỹ thuật và phải được hiệu chỉnh theo FPS/môi trường triển khai.

---

## 4. Đối tượng mục tiêu

### MVP — Vehicle occupation only

- motorcycle
- car
- bicycle
- bus
- truck

`person` được detect và track chỉ để cung cấp context; không thuộc đối tượng đánh giá vi phạm.

### Future scope (ngoài MVP)

- table
- chair
- cart
- stall
- goods
- signboard
- construction material
- street vending
- behavior recognition

Các class future có thể yêu cầu custom dataset và fine-tuning.

---

## 5. Zone

Hệ thống hỗ trợ bốn loại zone:

### `SIDEWALK`

Vùng vỉa hè.

### `MONITORED`

Vùng bất kỳ do người dùng thiết lập để giám sát.

### `ALLOWED`

Vùng nằm trong hoặc giao với monitored area nhưng được phép sử dụng.

### `IGNORE`

Vùng hệ thống bỏ qua do camera, scene hoặc nghiệp vụ.

---

## 6. Event Definition

Tên event duy nhất: `SUSPECTED_AREA_OCCUPATION`. `zone_type = SIDEWALK` hoặc `MONITORED` phân biệt khu vực xảy ra sự kiện.

Event xác định phương tiện, camera/video nguồn, zone, các mốc thời gian và snapshot để hậu kiểm. Metadata lưu SQLite, snapshot lưu filesystem. Schema chi tiết nằm trong [Solution Design](solution_design.md).

Zone precedence: `IGNORE > ALLOWED > SIDEWALK / MONITORED`. Object trong `SIDEWALK + ALLOWED` không sinh event.

---

## 7. Positive Case

Ví dụ:

- Xe máy đi vào vỉa hè.
- Xe dừng.
- Xe đứng trong ROI trên threshold.
- Xe không thuộc allowed/ignore zone.
- Hệ thống tạo đúng một event.

---

## 8. Negative Cases quan trọng

Hệ thống không được báo vi phạm trong các tình huống:

- xe chạy ngang ROI;
- xe chạm mép ROI trong thời gian rất ngắn;
- object nằm trong allowed zone;
- object nằm trong ignore zone;
- tracker đổi ID nhưng vẫn là cùng một object: không sinh thêm alert trùng;
- bbox jitter ở mép polygon;
- người đi bộ bình thường trên vỉa hè;
- object xuất hiện một vài frame rồi mất;
- camera drop frame ngắn hạn.

---

## 9. Input

### Video

- local video file;
- RTSP stream trong phase sau;
- camera fixed-view.

### Zone configuration

ROI polygon cấu hình thủ công theo từng camera. Chi tiết cấu hình nằm trong [Solution Design](solution_design.md).

---

## 10. Output

Hệ thống xuất:

1. event metadata;
2. annotated frame;
3. snapshot;
4. optional video clip ở giai đoạn sau;
5. event log.

Ví dụ:

```json
{
  "event_type": "SUSPECTED_AREA_OCCUPATION",
  "camera_id": "CAM_001",
  "zone_id": "SW_01",
  "track_id": 27,
  "object_class": "motorcycle",
  "inside_frame_count": 30,
  "confidence": 0.91,
  "status": "OPEN"
}
```

---

## 11. Success Criteria

MVP được xem là thành công khi:

- xử lý được video end-to-end;
- theo dõi được object bằng persistent ID;
- ROI hoạt động đúng;
- đếm đúng số frame liên tiếp mà track nằm trong ROI;
- tạo event theo ngưỡng `min_inside_frames`;
- không tạo duplicate alert liên tục;
- có ground truth để tính event precision/recall/F1;
- có log experiment để tune threshold.

Acceptance threshold định lượng chỉ được chốt sau Baseline Run 0 trên dữ liệu đại diện.

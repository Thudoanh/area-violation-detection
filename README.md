# Area Violation Detection

Hệ thống thị giác máy tính phát hiện phương tiện nghi ngờ chiếm dụng vỉa hè hoặc
khu vực được cấu hình để giám sát trong video camera cố định.

Hệ thống tạo video đã đánh dấu, sự kiện và ảnh bằng chứng để phục vụ cảnh báo,
kiểm tra lại. Kết quả không phải kết luận vi phạm pháp luật.

> **Trạng thái:** Pipeline MVP và web demo đã hoạt động. Dự án đang chờ video thực
> tế và ground truth để chạy baseline đánh giá chất lượng.

## Hệ thống làm được gì?

```text
Video → Nhận diện phương tiện → Theo dõi → Kiểm tra ROI
      → Xác định dừng lâu → Tạo sự kiện → Video + SQLite + Snapshot
```

1. Nhận diện người và các phương tiện xuất hiện trong video.
2. Theo dõi từng phương tiện bằng một ID trong suốt quá trình di chuyển.
3. Kiểm tra phương tiện có nằm trong vùng giám sát hay không.
4. Chỉ tạo sự kiện khi phương tiện đứng trong vùng đủ lâu.
5. Hạn chế cảnh báo trùng và lưu lại bằng chứng để hậu kiểm.

Các phương tiện được xét gồm xe máy, xe đạp, ô tô, xe buýt và xe tải. Người chỉ
được hiển thị để cung cấp ngữ cảnh, không tạo sự kiện chiếm dụng.

## Web demo

Web demo sử dụng giao diện tiếng Việt và hỗ trợ:

- tải video qua file picker và tự chuyển sang H.264 để xem trước trên trình duyệt;
- vẽ ROI trực tiếp bằng cách click lên khung hình;
- tự động đề xuất ROI cho vùng kẻ/sơn có độ tương phản rõ;
- điều chỉnh ngưỡng nhận diện và thời gian dừng;
- xem video đã chạy detection ngay trên trình duyệt;
- xem bảng sự kiện, biểu đồ tổng quan và ảnh bằng chứng.

ROI tự động chỉ là đề xuất. Người dùng cần kiểm tra polygon trên preview và có thể
chỉnh tọa độ hoặc vẽ lại trước khi chạy phát hiện.

## Cài đặt

Yêu cầu Python 3.10 trở lên.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Tải YOLO11n weights chính thức:

```powershell
New-Item -ItemType Directory -Force weights
Invoke-WebRequest `
  -Uri https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt `
  -OutFile weights/yolo11n.pt
```

## Chạy web demo

```powershell
python app.py
```

Ứng dụng ưu tiên địa chỉ `http://127.0.0.1:7860`. Nếu cổng này đang được sử dụng,
Gradio sẽ chọn cổng kế tiếp và in địa chỉ chính xác trong terminal.

Quy trình sử dụng:

1. Tải video lên.
2. Vẽ ROI thủ công hoặc chọn **Tự động đề xuất ROI**.
3. Kiểm tra polygon trên khung hình preview.
4. Chọn loại khu vực và các thiết lập cần thiết.
5. Nhấn **Chạy phát hiện**.
6. Xem video kết quả, bảng sự kiện, biểu đồ và snapshot.

## Chạy bằng command line

Tạo ROI thủ công trên frame đầu tiên:

```powershell
python scripts/select_roi.py `
  --video data/videos/example.mp4 `
  --zone-type SIDEWALK `
  --zone-id SW_01 `
  --output configs/zones/cam_001.json
```

Chạy pipeline:

```powershell
python scripts/run_video.py --video data/videos/example.mp4
```

## Kết quả đầu ra

| Dữ liệu | Vị trí mặc định |
|---|---|
| Sự kiện | `data/events/events.db` |
| Ảnh bằng chứng | `data/events/snapshots/` |
| Video từ CLI | `data/events/annotated_<video>.mp4` |
| Video từ web | `data/events/web/<run_id>/` |

Mỗi lần xử lý có một `run_id` riêng. Web chỉ hiển thị các sự kiện thuộc lần chạy
hiện tại, trong khi SQLite vẫn giữ lịch sử của các lần chạy trước.

## Công nghệ chính

| Thành phần | Công nghệ |
|---|---|
| Nhận diện | YOLO11n pretrained COCO |
| Theo dõi | ByteTrack |
| Xử lý video | OpenCV |
| Web demo | Gradio + Plotly |
| Lưu sự kiện | SQLite + ảnh JPEG |

## Cấu trúc dự án

```text
app.py              Web demo tiếng Việt
configs/             Cấu hình và ROI theo camera
data/                Video, ground truth và kết quả
docs/                Đặc tả, thiết kế và đánh giá
scripts/             Công cụ chạy video, chọn ROI và evaluation
src/                 Detection, tracking, zones, rules và storage
tests/               Automated tests
weights/             Model weights cục bộ
```

## Trạng thái đánh giá

| Hạng mục | Trạng thái |
|---|---|
| Pipeline detection → event | Đã triển khai |
| Web demo | Đã triển khai |
| Automated tests | 201 passed, 1 conditional test skipped |
| Video CCTV thực tế | Chưa có trong repository |
| Event Precision / Recall / F1 | Chưa có baseline |
| Acceptance thresholds | Chốt sau Baseline Run 0 |

Automated tests xác nhận hành vi của code, nhưng không thay thế việc đánh giá trên
video camera thực tế. Các chỉ số chất lượng chỉ được công bố sau khi có dataset,
ground truth và Baseline Run 0.

## Phạm vi hiện tại

Dự án tập trung vào video file từ camera cố định và sự kiện phương tiện đứng lâu
trong vùng giám sát. Phiên bản hiện tại chưa hỗ trợ RTSP, nhận diện biển số, nhận
diện khuôn mặt, nhiều camera đồng thời hoặc semantic segmentation vỉa hè.

Auto ROI hiện phù hợp nhất với vùng kẻ/sơn hoặc vùng có đường biên tương phản rõ.
Vùng giám sát tùy ý vẫn nên được xác định thủ công.

## Tài liệu

- [Problem Definition](docs/problem_definition.md) — bài toán và phạm vi nghiệp vụ.
- [Requirements](docs/requirements.md) — yêu cầu chức năng.
- [Solution Design](docs/solution_design.md) — kiến trúc và quyết định kỹ thuật.
- [Dataset](docs/dataset.md) — kế hoạch dữ liệu và scenario.
- [Evaluation](docs/evaluation.md) — metrics và quy trình baseline.
- [Experiment Log](docs/experiment_log.md) — lịch sử thay đổi và thực nghiệm.

## Nguồn tham khảo

Dự án được phát triển theo hướng clean-room reimplementation và tham khảo cách tổ
chức pipeline của
[sertacakalin/hatched-area-violation-detection](https://github.com/sertacakalin/hatched-area-violation-detection).
Không sao chép source code của repository tham chiếu.

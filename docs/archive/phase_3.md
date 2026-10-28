# Phase 3: Hoàn thiện Backend API & Tích hợp Vector Database

## 1. Tổng quan (Overview)

Giai đoạn này đã kết nối thành công luồng xử lý Computer Vision ngoại tuyến (Offline CV Pipeline ở Phase 1) với Vector Database (Stage 4) và đưa các tính năng ra ngoài thông qua REST API hiệu năng cao sử dụng FastAPI (Stage 5). Hệ thống hiện tại đã hoàn toàn sẵn sàng nhận các truy vấn bằng ngôn ngữ tự nhiên (natural language queries), thực hiện semantic search và stream video động (dynamic video streaming).

## 2. Các mốc hoàn thành chính (Key Milestones)

### A. Tích hợp Core CV & Database (Stage 3 & Stage 4)

* **Kết nối Pipeline (Pipeline Bridging):** Cập nhật script `src/vision/embed_crops.py` đóng vai trò cầu nối giữa xử lý offline và cơ sở dữ liệu. Sau khi tạo các 512-D L2-normalized embeddings bằng CLIP, script sẽ tự động map dữ liệu và gọi hàm `ingest_video_data()` để thực hiện thao tác UPSERT các bản ghi vào PostgreSQL một cách an toàn (idempotent).
* **Sửa lỗi & Môi trường (Environment & Bug Resolution):**
  * Xử lý triệt để lỗi `ModuleNotFoundError` (`cv2`, `torch`) bằng cách chuẩn hóa môi trường chạy test (`python3 -m pytest`).
  * Khắc phục lỗi SQLAlchemy `CompileError` (unconsumed columns) để đảm bảo quá trình DB ingestion hoạt động mượt mà.
  * Toàn bộ 37 unit tests cho CV pipeline đã pass thành công trong chưa tới 9 giây, đảm bảo pipeline xử lý dữ liệu cực kỳ ổn định.

### B. Lớp Backend REST API - FastAPI (Stage 5)

* **Data Validation (`src/api/schemas.py`):** Triển khai các Pydantic models khắt khe (`SearchRequest`, `SearchResponse`, `SearchResultItem`) để đảm bảo type safety cho request đầu vào và cấu trúc JSON payload đầu ra.
* **Quản lý Model Lifespan (`src/api/main.py`):** Sử dụng tính năng `@asynccontextmanager` của FastAPI để load model CLIP (\~500MB) vào bộ nhớ RAM một lần duy nhất khi khởi động server (server startup). Kỹ thuật này giúp loại bỏ hoàn toàn độ trễ (inference overhead) khi người dùng gửi request tìm kiếm.
* **Semantic Search Endpoint (`POST /api/search`):**
  * Tiếp nhận text queries từ người dùng.
  * Chuyển đổi text thành 512-D vector on the fly (ngay lập tức).
  * Truy vấn cơ sở dữ liệu PostgreSQL (`pgvector`) sử dụng thuật toán Approximate Nearest Neighbors với chỉ mục HNSW và khoảng cách Cosine Distance (`<=>`).
  * Lọc trùng lặp (deduplicate) kết quả theo `track_id` trực tiếp ở tầng SQL để tối ưu tốc độ.
* **Video Streaming Endpoint (`GET /api/videos/{video_id}/stream`):**
  * Triển khai thuật toán đọc file theo chunk (byte-range chunking) hỗ trợ chuẩn HTTP 206 Partial Content.
  * Cho phép client tua nhanh (seek/fast-forward) đến một thời điểm cụ thể (`timestamp_sec`) ngay lập tức mà không cần tải toàn bộ dung lượng file video.

## 3. Kiểm thử toàn trình (End-to-End Verification)

Toàn bộ luồng hoạt động đã được verify trực tiếp trên terminal một cách hoàn hảo:

1. **Ingestion:** Lệnh `python3 -m src.vision.embed_crops --all` đã xử lý thành công `metadata.jsonl`, xuất `.npy` backups và đẩy dữ liệu (push data) vào PostgreSQL thành công.
2. **Server Startup:** Lệnh `python3 -m uvicorn src.api.main:app` khởi động trơn tru, load CLIP model vào memory và mở cổng lắng nghe.
3. **Search Execution:** Lệnh `curl` giả lập truy vấn *"a person walking"* đã truy xuất đúng cơ sở dữ liệu và trả về chính xác JSON array chứa `video_path`, tọa độ `bbox_xyxy`, thời gian `timestamp_sec`, và điểm số `cosine_distance`.

## 4. Bước tiếp theo: Phase 4 (Frontend)

"Bộ não" (AI) và "Trái tim" (Backend API) đã vận hành hoàn chỉnh. Giai đoạn tiếp theo (Stage 6) là xây dựng "Gương mặt" (Frontend GUI) sử dụng React/Next.js để cung cấp thanh tìm kiếm (search bar) trực quan, hiển thị kết quả (thumbnails) và một video player tương tác có khả năng vẽ bounding boxes bám theo các đối tượng đã được tìm kiếm.
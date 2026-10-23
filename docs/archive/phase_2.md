# Phase 2: Vector Database & Storage Layer (Stage 4)

## 1. Mục tiêu
Xây dựng lớp lưu trữ dữ liệu (Database Layer) để kết nối luồng xử lý AI (Offline) với hệ thống tìm kiếm (Online). Hệ thống cần lưu trữ cả dữ liệu quan hệ (metadata của video, tọa độ bounding box) và dữ liệu phi cấu trúc (vector 512 chiều từ CLIP) trên cùng một hệ quản trị cơ sở dữ liệu PostgreSQL.

## 2. Các thành phần đã triển khai (`src/db/`)

*   **`src/utils/config.py`**: Thiết lập module quản lý cấu hình, đọc biến môi trường `DATABASE_URL` để bảo mật thông tin kết nối.
*   **`src/db/database.py`**: 
    *   Sử dụng SQLAlchemy 2.0 tạo `Engine` và `SessionLocal`.
    *   Viết hàm `init_db()` tự động kích hoạt tính năng vector bằng lệnh `CREATE EXTENSION IF NOT EXISTS vector;` và tạo bảng tự động qua `Base.metadata.create_all`.
*   **`src/db/models.py`**: Thiết kế 3 bảng ORM chuẩn hóa:
    *   `Video`: Chứa metadata của video gốc.
    *   `Track`: Chứa các đối tượng duy nhất trong video (sử dụng `UniqueConstraint` trên `video_id` và `track_id` để chống trùng lặp).
    *   `Crop`: Chứa từng khung hình cắt, tọa độ `bbox_xyxy`, thời gian, và đặc biệt là cột `embedding` kiểu `Vector(512)`.
    *   *Tối ưu hóa:* Đánh chỉ mục **HNSW** (`m=16, ef_construction=64`) trên cột `embedding` với toán tử `vector_cosine_ops` để truy vấn K-NN siêu tốc.
*   **`src/db/ingest.py`**: Viết luồng nạp dữ liệu thông minh (Idempotent UPSERT). Tự động bỏ qua nếu video/track đã tồn tại và dùng `bulk_save_objects` để chèn hàng nghìn ảnh crop cùng lúc.
*   **`src/db/search.py`**: Xây dựng thuật toán tìm kiếm lai. Sử dụng subquery kết hợp `DISTINCT ON (track_id)` của PostgreSQL để gom nhóm kết quả, đảm bảo khi query không trả về nhiều ảnh của cùng một vật thể, đồng thời dùng toán tử `<=>` để tính khoảng cách Cosine.

## 3. Nhật ký Xử lý Sự cố (Troubleshooting Log)

Trong quá trình khởi tạo container, hệ thống gặp 2 sự cố hạ tầng lớn và đã được giải quyết triệt để:

1.  **Lỗi `KeyError: 'ContainerConfig'` (Bất đồng bộ phiên bản Docker):**
    *   *Nguyên nhân:* Ubuntu đang sử dụng bản `docker-compose` cũ (v1.29 - viết bằng Python), không hiểu được metadata của Image `pgvector` đời mới (chuẩn BuildKit).
    *   *Giải pháp:* Xóa bản cũ, tải trực tiếp binary của Docker Compose V2 (v2.29.2 - viết bằng Go) vào `~/.docker/cli-plugins/`. Đổi lệnh từ `docker-compose` thành `docker compose`.
2.  **Lỗi `database files are incompatible with server` (Xung đột phiên bản Postgres):**
    *   *Nguyên nhân:* Volume cũ của Docker đang chứa dữ liệu của Postgres 15, nhưng image mới lại kéo bản Postgres 17.
    *   *Giải pháp:* Chạy `docker compose down -v` để ép Docker xóa bỏ ổ cứng ảo (volume) cũ bị lỗi, cho phép Postgres 17 khởi tạo data dir hoàn toàn mới.
3.  **Lỗi `password authentication failed` (Khóa mật khẩu khởi tạo):**
    *   *Nguyên nhân:* Docker chốt mật khẩu khởi tạo từ file `.env` ở lần chạy đầu tiên. Sửa file `.env` sau đó không làm thay đổi mật khẩu trong DB đã tạo.
    *   *Giải pháp:* Hardcode tạm thời `DATABASE_URL` trong config, xóa volume và chạy `up -d` lại để đồng bộ hoàn toàn tài khoản `root` và mật khẩu `secret`.

## 4. Kết quả Kiểm thử (Verification)

Viết script `run_test_ingest.py` sinh vector giả (dummy L2-normalized 512-D vectors) và đẩy vào DB. 
Truy vấn trực tiếp qua `psql` cho thấy kết quả thành công 100%:
*   Bảng `videos` nhận 1 video.
*   Bảng `tracks` tự động gom 3 bức ảnh thành đúng 2 đối tượng duy nhất (car ID 3, person ID 5).
*   Cột `embedding` trong bảng `crops` được xác nhận có đúng 512 chiều bằng hàm `vector_dims()`.

## 5. Bước tiếp theo (Next Step)
Gắn hàm `ingest_video_data` vào luồng CV offline (`embed_crops.py`) và chuyển sang **Stage 5**: Khởi tạo FastAPI xây dựng các endpoint REST API phục vụ tìm kiếm và streaming video.
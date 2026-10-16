# Tài Liệu Kỹ Thuật (Phase 1) - Dự Án Multimodal Video Search Engine

Tài liệu này ghi chú lại quá trình khởi tạo môi trường chuẩn MLOps, xây dựng luồng xử lý video (Computer Vision) và nhúng vector ngữ nghĩa (Multimodal Embedding).

## 1. Khởi Tạo Không Gian Dự Án

Xây dựng cấu trúc thư mục (Folder Structure) tách biệt rõ ràng các module logic:

```bash
# 1.1 Tạo thư mục gốc dự án
mkdir multimodal-video-search && cd multimodal-video-search

# 1.2 Tạo cấu trúc thư mục con và các file rỗng
mkdir -p src/{vision,db,api,utils} data/{raw_videos,crops} notebooks docker configs
touch src/__init__.py src/vision/__init__.py src/db/__init__.py src/api/__init__.py
touch Makefile README.md .gitignore
```

## 2. Thiết Lập Môi Trường (Anaconda & Python)

Cô lập môi trường thư viện để tránh xung đột hệ thống trên Ubuntu.

```bash
# 2.1 Khởi tạo môi trường ảo với Python 3.10
conda create -n video_rag python=3.10 -y

# 2.2 Kích hoạt môi trường (chạy lệnh này mỗi khi mở terminal mới)
conda activate video_rag

# 2.3 Cài đặt PyTorch chuẩn với CUDA 12.1 của NVIDIA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 2.4 Cài đặt các thư viện lõi cho AI & Computer Vision
pip install ultralytics transformers opencv-python fastapi uvicorn[standard] psycopg2-binary sqlalchemy pgvector

# 2.5 Cài đặt MKL tương thích (Sửa lỗi "undefined symbol: iJIT_NotifyEvent" của libtorch)
conda install mkl=2024.0.0 -c conda-forge -y
```

## 3. Thiết Lập Cơ Sở Dữ Liệu pgvector (Docker)

Dựng cơ sở hạ tầng lưu trữ Vector độc lập, quản lý bằng Docker Compose.

```bash
# 3.1 Mở/Tạo file docker-compose
code docker/docker-compose.yml
```

**Nội dung `docker/docker-compose.yml`:**

```yaml
version: '3.8'
services:
  db:
    image: ankane/pgvector:latest
    container_name: video_rag_db
    restart: always
    environment:
      POSTGRES_USER: root
      POSTGRES_PASSWORD: password123
      POSTGRES_DB: videorag
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
```

```bash
# 3.2 Kích hoạt Database chạy ngầm dưới nền
cd docker && docker-compose up -d && cd ..
```

## 4. Cấu Hình Version Control (Git)

Loại trừ các file môi trường, dữ liệu nặng hoặc mô hình ra khỏi Source Code.

```bash
# 4.1 Khởi tạo git repository
git init

# 4.2 Thiết lập bộ lọc ignore
code .gitignore
```

**Nội dung file `.gitignore`:**

```text
__pycache__/
*.pyc
.env
data/
*.mp4
*.avi
*.pt
*.onnx
build/
*.o
```

## 5. Xây Dựng Luồng Xử Lý Hình Ảnh & Tracking (Stage 1 & 2)

### Mục tiêu:

Sử dụng YOLOv10 và thuật toán ByteTrack đọc file video `.mp4`, khoanh vùng các vật thể, gán ID theo dõi và cắt ảnh tự động (Smart Cropping) với tần suất `sample_rate`.

### Thực hiện:

1. **Chuẩn bị:** Tải 1 file `test_video.mp4` thả vào thư mục `data/raw_videos/`.

2. **Tải trọng số AI:**

   ```bash
   wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt -O data/yolov10s.pt
   ```

3. **Viết mã nguồn:** Tạo file `src/vision/tracker.py`

**Mã nguồn `src/vision/tracker.py`:**

```python
import cv2
import os
from ultralytics import YOLO

def process_video(video_path, output_dir, sample_rate=15):
    """Đọc video, tracking đối tượng và cắt ảnh lưu ra file theo sample_rate"""
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO('data/yolov10s.pt')
    cap = cv2.VideoCapture(video_path)
    
    frame_idx = 0
    last_crop_frame = {} 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Gọi thuật toán ByteTrack nhận diện Người(0) và Xe Cộ(2)
        results = model.track(frame, persist=True, classes=[0, 2], tracker="bytetrack.yaml", verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, track_ids):
                # Logic lấy mẫu động: Cắt ảnh nếu ID này xuất hiện lần đầu, hoặc đã vượt quá sample_rate
                if track_id is not None and (track_id not in last_crop_frame or frame_idx - last_crop_frame[track_id] >= sample_rate):
                    x1, y1, x2, y2 = box
                    y1, y2 = max(0, y1), min(frame.shape[0], y2)
                    x1, x2 = max(0, x1), min(frame.shape[1], x2)
                    
                    crop_img = frame[y1:y2, x1:x2]
                    
                    if crop_img.size > 0:
                        crop_filename = f"{output_dir}/id_{track_id}_frame_{frame_idx}.jpg"
                        cv2.imwrite(crop_filename, crop_img)
                        print(f"[Lưu] ID: {track_id} | Frame: {frame_idx} | File: {crop_filename}")
                        last_crop_frame[track_id] = frame_idx
        frame_idx += 1
    cap.release()
    print("Hoàn tất xử lý video.")

if __name__ == "__main__":
    process_video("data/raw_videos/test_video.mp4", "data/crops", sample_rate=15)
```

**Cách chạy kiểm thử và Kết quả thực tế:**

```bash
python3 src/vision/tracker.py
```

**Log Terminal:**
```text
requirements: Ultralytics requirement ['lap>=0.5.12'] not found, attempting AutoUpdate...
Using Python 3.10.12 environment at: /home/thanh-trung/miniconda3/envs/video_rag
Resolved 2 packages in 590ms
Downloading lap (1.6MiB)
Downloaded lap
Prepared 1 package in 332ms
Installed 1 package in 6ms
 + lap==0.5.13
requirements: AutoUpdate success ✅ 1.1s
WARNING ⚠️ requirements: Restart runtime or rerun command for updates to take effect

[Lưu] ID: 3 | Frame: 69 | File: data/crops/id_3_frame_69.jpg
[Lưu] ID: 3 | Frame: 86 | File: data/crops/id_3_frame_86.jpg
[Lưu] ID: 3 | Frame: 101 | File: data/crops/id_3_frame_101.jpg
[Lưu] ID: 6 | Frame: 185 | File: data/crops/id_6_frame_185.jpg
[Lưu] ID: 8 | Frame: 198 | File: data/crops/id_8_frame_198.jpg
[Lưu] ID: 6 | Frame: 200 | File: data/crops/id_6_frame_200.jpg
[Lưu] ID: 9 | Frame: 202 | File: data/crops/id_9_frame_202.jpg
[Lưu] ID: 10 | Frame: 219 | File: data/crops/id_10_frame_219.jpg
[Lưu] ID: 11 | Frame: 223 | File: data/crops/id_11_frame_223.jpg
[Lưu] ID: 10 | Frame: 234 | File: data/crops/id_10_frame_234.jpg
[Lưu] ID: 12 | Frame: 325 | File: data/crops/id_12_frame_325.jpg
[Lưu] ID: 14 | Frame: 345 | File: data/crops/id_14_frame_345.jpg
Hoàn tất xử lý video.
```

*(Thư mục `data/crops/` hiện đã chứa thành công các ảnh tĩnh vật thể được cắt tự động bởi YOLO và ByteTrack).*

## 6. Xây Dựng Logic Nhúng Đa Phương Thức (Stage 3 - Baseline)

### Mục tiêu:

Mở các file ảnh đã cắt bằng OpenCV/PIL, đẩy qua mô hình Vision-Language `CLIP` của OpenAI và nén bức ảnh đó thành một Vector số thực đặc trưng đa chiều để tiến hành tìm kiếm sau này.

### Thực hiện:

1. **Viết mã nguồn:** Tạo file `src/vision/embedder.py`

**Mã nguồn `src/vision/embedder.py`:**

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        # Auto-detect GPU Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Khởi tạo CLIP Model trên {self.device.upper()}...")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def embed_image(self, image_path):
        """Kéo ảnh từ ổ cứng, chuẩn hóa và chuyển thành Vector đặc trưng"""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                
            # Chuẩn hóa L2-Norm cho thuật toán đo khoảng cách Cosine Similarity
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu().numpy()[0].tolist()
            
        except Exception as e:
            print(f"Lỗi khi embed ảnh {image_path}: {e}")
            return None

    def embed_text(self, text):
        """Hàm dự phòng: Dùng cho Luồng Người Dùng gõ tìm kiếm văn bản"""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0].tolist()

if __name__ == "__main__":
    import os
    embedder = CLIPEmbedder()
    crop_dir = "data/crops/"
    if os.path.exists(crop_dir) and len(os.listdir(crop_dir)) > 0:
        sample_img = os.path.join(crop_dir, os.listdir(crop_dir)[0])
        print(f"Tạo Vector cho mẫu: {sample_img}")
        
        vector = embedder.embed_image(sample_img)
        if vector:
            print(f"✅ Kích thước Tensor: {len(vector)} chiều. Output chuẩn.")
```

**Cách chạy kiểm thử:**

```bash
python3 src/vision/embedder.py
```

*(Kết quả mong đợi: In ra dòng log xác nhận kích thước tensor là 512).*
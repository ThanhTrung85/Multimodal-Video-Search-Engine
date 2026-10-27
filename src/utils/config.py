# Configuration file: Used to configure shared components
# File: src/utils/config.py
from __future__ import annotations

# Ghi cứng chuỗi kết nối để đảm bảo không lấy nhầm biến môi trường cũ
DATABASE_URL = "postgresql+psycopg2://root:secret@localhost:5432/videorag"
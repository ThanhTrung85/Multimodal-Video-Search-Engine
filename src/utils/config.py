# Configuration file: Used to configure shared components
from __future__ import annotations

import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://root:secret@localhost:5432/videorag",
)

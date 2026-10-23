# Define struct for table
# Coding SQLAIchemy define 3 tables have Foreig Keys relationship together

# Video: Saving video's information (id, name, file_path, duration)
# Track: Saving all objects in video (iid, video_id. track_id, class_name)
# Crop: Saving the cropping image of each object 
# (Must have a column that name is embedding with the data type is Vector(512))
from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.database import Base


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    tracks: Mapped[list["Track"]] = relationship(back_populates="video", cascade="all, delete-orphan")
    crops: Mapped[list["Crop"]] = relationship(back_populates="video", cascade="all, delete-orphan")


class Track(Base):
    __tablename__ = "tracks"
    __table_args__ = (UniqueConstraint("video_id", "track_id", name="uq_tracks_video_track"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"), nullable=False, index=True)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    class_name: Mapped[str] = mapped_column(String(128), nullable=False)
    start_frame: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_frame: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    video: Mapped["Video"] = relationship(back_populates="tracks")
    crops: Mapped[list["Crop"]] = relationship(back_populates="track", cascade="all, delete-orphan")


class Crop(Base):
    __tablename__ = "crops"
    __table_args__ = (
        Index(
            "ix_crops_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"), nullable=False, index=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id"), nullable=False, index=True)
    frame_idx: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    timestamp_sec: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_xyxy: Mapped[dict] = mapped_column(JSON, nullable=False)
    crop_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(512), nullable=False)

    video: Mapped["Video"] = relationship(back_populates="crops")
    track: Mapped["Track"] = relationship(back_populates="crops")

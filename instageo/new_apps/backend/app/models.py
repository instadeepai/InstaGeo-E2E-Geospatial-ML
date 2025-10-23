"""Database models module for the InstaGeo backend."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for database models."""

    pass


class User(Base):
    """User model for database."""

    __tablename__ = "users"

    sub = Column(String, primary_key=True)
    email = Column(String, nullable=True)
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    last_seen_at = Column(DateTime, default=datetime.now(timezone.utc))

    tasks = relationship("Task", back_populates="user")


class Task(Base):
    """Task model for database."""

    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_sub = Column(String, ForeignKey("users.sub"), nullable=False)
    task_id = Column(String, unique=True, nullable=False)
    bboxes = Column(Text, nullable=True)
    parameters = Column(Text, nullable=True)
    status = Column(String, default="data_processing")
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    stages = Column(Text, nullable=True)
    model_short_name = Column(String, nullable=True)
    model_type = Column(String, nullable=True)
    model_name = Column(String, nullable=True)
    model_size = Column(String, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="tasks")

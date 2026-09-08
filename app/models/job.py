"""SQLAlchemy model for job descriptions."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from app.core.database import Base

class JobModel(Base):
    __tablename__ = "job_descriptions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String(255), nullable=True, default="Job Description")
    filename = Column(String(255), nullable=True)
    raw_text = Column(Text, nullable=False)
    clean_text = Column(Text, nullable=True)
    required_skills = Column(Text, nullable=True)  # JSON-serialized list
    created_at = Column(DateTime, default=datetime.utcnow)

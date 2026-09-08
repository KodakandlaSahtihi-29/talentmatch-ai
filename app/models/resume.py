"""SQLAlchemy model for uploaded resumes."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from app.core.database import Base

class ResumeModel(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), default="pdf")
    raw_text = Column(Text, nullable=False)
    clean_text = Column(Text, nullable=True)
    extracted_skills = Column(Text, nullable=True)  # JSON-serialized list
    created_at = Column(DateTime, default=datetime.utcnow)

"""Schemas for resume upload and representation."""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict

class ResumeUploadResponse(BaseModel):
    id: int
    filename: str
    text_length: int
    extracted_skills: List[str]
    message: str

class ResumeDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    raw_text: str
    clean_text: Optional[str] = None
    extracted_skills: List[str] = []
    created_at: datetime

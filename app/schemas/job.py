"""Schemas for job description upload and representation."""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict

class JobUploadRequest(BaseModel):
    title: Optional[str] = "Target Job Description"
    job_text: str

class JobUploadResponse(BaseModel):
    id: int
    title: str
    filename: Optional[str] = None
    text_length: int
    required_skills: List[str]
    message: str

class JobDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    filename: Optional[str] = None
    raw_text: str
    clean_text: Optional[str] = None
    required_skills: List[str] = []
    created_at: datetime

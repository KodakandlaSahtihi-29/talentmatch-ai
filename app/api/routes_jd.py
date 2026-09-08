"""Job Description upload and parsing API routes."""
import json
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.job import JobModel
from app.schemas.job import JobUploadRequest, JobUploadResponse, JobDetail
from app.services.pdf_extractor import extract_text_from_file
from app.services.preprocessing import preprocess_text
from app.services.skill_extractor import extract_skills_from_text

router = APIRouter(prefix="/job", tags=["Job Descriptions"])
logger = logging.getLogger(__name__)

@router.post("/upload-file", response_model=JobUploadResponse)
async def upload_job_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form("Job Description"),
    db: Session = Depends(get_db)
):
    """
    Upload and parse a Job Description from PDF or TXT file.
    """
    filename = file.filename or "job_description.txt"
    content = await file.read()
    
    if not content or len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded JD file is empty (0 bytes).")
        
    try:
        raw_text, _ = extract_text_from_file(content, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error parsing JD {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract text from JD: {str(e)}")
        
    doc = preprocess_text(raw_text)
    skills_data = extract_skills_from_text(raw_text)
    
    job_entry = JobModel(
        title=title or filename,
        filename=filename,
        raw_text=raw_text,
        clean_text=doc.clean_text,
        required_skills=json.dumps(skills_data["all_skills"])
    )
    
    db.add(job_entry)
    db.commit()
    db.refresh(job_entry)
    
    return JobUploadResponse(
        id=job_entry.id,
        title=job_entry.title,
        filename=filename,
        text_length=len(raw_text),
        required_skills=skills_data["all_skills"],
        message=f"Job Description processed successfully ({len(skills_data['all_skills'])} requirements identified)."
    )

@router.post("/upload-text", response_model=JobUploadResponse)
def upload_job_text(
    payload: JobUploadRequest,
    db: Session = Depends(get_db)
):
    """
    Accepts pasted or raw text for a Job Description.
    """
    if not payload.job_text or not payload.job_text.strip():
        raise HTTPException(status_code=400, detail="Job Description text cannot be empty.")
        
    raw_text = payload.job_text.strip()
    doc = preprocess_text(raw_text)
    skills_data = extract_skills_from_text(raw_text)
    
    job_entry = JobModel(
        title=payload.title or "Job Description",
        filename="pasted_text.txt",
        raw_text=raw_text,
        clean_text=doc.clean_text,
        required_skills=json.dumps(skills_data["all_skills"])
    )
    
    db.add(job_entry)
    db.commit()
    db.refresh(job_entry)
    
    return JobUploadResponse(
        id=job_entry.id,
        title=job_entry.title,
        filename="pasted_text.txt",
        text_length=len(raw_text),
        required_skills=skills_data["all_skills"],
        message=f"Job Description text parsed successfully ({len(skills_data['all_skills'])} requirements identified)."
    )

@router.get("/{job_id}", response_model=JobDetail)
def get_job(job_id: int, db: Session = Depends(get_db)):
    """Fetches details of a stored job description by ID."""
    job = db.query(JobModel).filter(JobModel.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found.")
        
    required_skills = []
    if job.required_skills:
        try:
            required_skills = json.loads(job.required_skills)
        except Exception:
            pass
            
    return JobDetail(
        id=job.id,
        title=job.title,
        filename=job.filename,
        raw_text=job.raw_text,
        clean_text=job.clean_text,
        required_skills=required_skills,
        created_at=job.created_at
    )

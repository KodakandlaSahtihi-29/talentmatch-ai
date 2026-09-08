"""Resume upload and processing API routes."""
import json
import logging
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.resume import ResumeModel
from app.schemas.resume import ResumeUploadResponse, ResumeDetail
from app.services.pdf_extractor import extract_text_from_file
from app.services.preprocessing import preprocess_text
from app.services.skill_extractor import extract_skills_from_text

router = APIRouter(prefix="/resume", tags=["Resumes"])
logger = logging.getLogger(__name__)

@router.post("/upload", response_model=ResumeUploadResponse)
async def upload_resume(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload and parse a candidate resume in PDF or TXT format.
    Extracts raw text, performs NLP preprocessing, and extracts candidate skills.
    """
    filename = file.filename or "resume.pdf"
    content = await file.read()
    
    if not content or len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")
        
    try:
        raw_text, page_count = extract_text_from_file(content, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error parsing resume {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract text from document: {str(e)}")
        
    # Preprocess & Extract Skills
    doc = preprocess_text(raw_text)
    skills_data = extract_skills_from_text(raw_text)
    
    resume_entry = ResumeModel(
        filename=filename,
        file_type=filename.split(".")[-1].lower() if "." in filename else "pdf",
        raw_text=raw_text,
        clean_text=doc.clean_text,
        extracted_skills=json.dumps(skills_data["all_skills"])
    )
    
    db.add(resume_entry)
    db.commit()
    db.refresh(resume_entry)
    
    return ResumeUploadResponse(
        id=resume_entry.id,
        filename=filename,
        text_length=len(raw_text),
        extracted_skills=skills_data["all_skills"],
        message=f"Resume successfully extracted ({page_count} pages/sections parsed, {len(skills_data['all_skills'])} skills detected)."
    )

@router.get("/{resume_id}", response_model=ResumeDetail)
def get_resume(resume_id: int, db: Session = Depends(get_db)):
    """Fetches details of an uploaded resume by ID."""
    resume = db.query(ResumeModel).filter(ResumeModel.id == resume_id).first()
    if not resume:
        raise HTTPException(status_code=404, detail=f"Resume with ID {resume_id} not found.")
    
    extracted_skills = []
    if resume.extracted_skills:
        try:
            extracted_skills = json.loads(resume.extracted_skills)
        except Exception:
            pass
            
    return ResumeDetail(
        id=resume.id,
        filename=resume.filename,
        raw_text=resume.raw_text,
        clean_text=resume.clean_text,
        extracted_skills=extracted_skills,
        created_at=resume.created_at
    )

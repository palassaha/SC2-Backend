from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import shutil
import os
from pathlib import Path
from interview.test import get_interview_questions
from onboarding.college_gpa import extract_gpa_from_image
from onboarding.school import extract_marks_from_marksheet
from summarizer.test import test_extraction
from skills.skills_matcher import analyze_resume_skills
from eligibility.eligibility_checker import check_detailed_eligibility

app = FastAPI()

# Pydantic model for skills matching request
class SkillsMatchRequest(BaseModel):
    company_skills: List[str]

# Pydantic model for eligibility check request
class EligibilityRequest(BaseModel):
    user: Dict[str, Any]
    post: Dict[str, Any]

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/student/extract-gpa/")
async def extract_gpa(file: UploadFile = File(...)):
    try:
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / file.filename

        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = extract_gpa_from_image(str(temp_file))

        os.remove(temp_file)

        return {"GPA": result}

    except Exception as e:
        return {"error": str(e)}

@app.post("/student/extract-percent/")
async def extract_percent(file: UploadFile = File(...)):
    try:
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / file.filename

        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        result = extract_marks_from_marksheet(str(temp_file))

        os.remove(temp_file)

        return {"Percent": result}

    except Exception as e:
        return {"error": str(e)}

@app.post("/job/summarize")
async def summarize_job(
    title: str = Body(...),
    description: str = Body(...)
):
    try:
        text = title + "\n" + description
        result = test_extraction(text)
        return {"summary": result}
    except Exception as e:
        return {"error": str(e)}

@app.post("/interview/questions")
async def get_questions(company: str, position: str):
    try:
        questions = get_interview_questions(company, position)
        return {"questions": questions}
    except Exception as e:
        return {"error": str(e)}

@app.post("/skills/match-resume")
async def match_resume_skills(
    file: UploadFile = File(...),
    company_skills: str = None
):
    """
    Match skills from resume with company required skills.
    
    Args:
        file: Resume file (PDF or image)
        company_skills: Comma-separated string of required skills
    """
    if not company_skills:
        raise HTTPException(status_code=400, detail="company_skills parameter is required")
    
    try:
        # Parse company skills
        skills_list = [skill.strip() for skill in company_skills.split(",") if skill.strip()]
        
        if not skills_list:
            raise HTTPException(status_code=400, detail="At least one skill must be provided")
        
        # Save uploaded file temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / file.filename

        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Analyze resume and match skills
        result = analyze_resume_skills(str(temp_file), skills_list)

        # Clean up temp file
        os.remove(temp_file)

        return {
            "filename": file.filename,
            "skills_analysis": result
        }

    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        return {"error": str(e)}

@app.post("/eligibility/check")
async def check_eligibility(request: EligibilityRequest):
    """
    Check eligibility for a job posting based on user profile and job criteria.
    
    Args:
        request: EligibilityRequest containing user and post data
    """
    try:
        result = check_detailed_eligibility(request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

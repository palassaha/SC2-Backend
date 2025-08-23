from fastapi import FastAPI, UploadFile, File
import uvicorn
import shutil
import os
from pathlib import Path
from onboarding.college_gpa import extract_gpa_from_image
from onboarding.school import extract_marks_from_marksheet

app = FastAPI()

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

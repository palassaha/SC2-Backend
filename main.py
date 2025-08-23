from onboarding.college_gpa import extract_student_scores
from fastapi import FastAPI, UploadFile, File
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/extract-scores/")
async def extract_scores(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = extract_student_scores(contents)  # pass raw bytes
        return result
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

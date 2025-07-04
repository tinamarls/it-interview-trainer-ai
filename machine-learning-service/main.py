from fastapi import FastAPI
from app.routers import resume, vacancy, video_analyze

app = FastAPI(title="Resume & Vacancy Parser")

app.include_router(resume.router, prefix="/resume", tags=["Resume Parsing"])
app.include_router(vacancy.router, prefix="/vacancy", tags=["Vacancy Parsing"])
app.include_router(video_analyze.router, prefix="/video", tags=["Video Analysis"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Resume & Vacancy Parser API"}
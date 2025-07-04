from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from typing import Dict
import logging

from app.services.comparasion_service import ComparasionService
from app.services.html_parser import extract_text_from_url

router = APIRouter()

@router.get("/parse_vacancy")
async def parse_vacancy(url: str):
    try:
        extracted_text = extract_text_from_url(url)
        return JSONResponse(content={"extracted_text": extracted_text}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.post("/compare")
async def compare(data: Dict[str, str] = Body(...)):
    resume_text = data.get("resume_text", "")
    vacancy_text = data.get("vacancy_text", "")
    comparasion_service = ComparasionService(resume_text=resume_text, vacancy_text=vacancy_text)

    logging.info(
        f"Получен запрос на сравнение текстов. Размер резюме: {len(resume_text)}, размер вакансии: {len(vacancy_text)}")

    return comparasion_service.compare_texts()

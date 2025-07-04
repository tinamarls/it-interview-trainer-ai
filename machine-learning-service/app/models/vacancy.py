from pydantic import BaseModel

class VacancyRequest(BaseModel):
    text: str

class VacancyResponse(BaseModel):
    required_skills: list
